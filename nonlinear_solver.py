import torch
from torch.func import vmap, jacrev
import numpy as np
from scipy.optimize import root

class IFTNewtonSolver(torch.autograd.Function):
    # differntiable Newton's method solver for G(u, *args) = 0 using implicit function theorem for backward pass.
    # can handler models-as-arguments by manually updating the shared module parameters in the backward pass,
    # which may be fragile compared to a functional approach but allows more flexibility in how the model is used inside G.
 
    # notes:
    # only works for batched inputs (or non tensor args)
    # not implemented line search or trust region
    # assumes G: (B, N) -> (B, N)
    # computes full dense Jacobians, may be slow for large N

    @staticmethod
    def forward(ctx, G, u_init, solver_params, *args):
        """
        Solves batched system G(u, *args) = 0 by Newton's method.
        u: (B, N)
        G(u, *args): (B, N)
        """
        B = u_init.shape[0]
        u = u_init.clone()
        max_iters = solver_params.get("max_iters", 20)
        tol = solver_params.get("tol", 1e-6)
        exit_ratio = solver_params.get("exit_ratio", 0.80)

        # need these for backward to handle non-tensor arguments
        tensor_mask = [torch.is_tensor(a) for a in args]
        tensor_args = [a for a in args if torch.is_tensor(a)]
        non_tensor_args = [a for a in args if not torch.is_tensor(a)]
        in_mask = [0 if torch.is_tensor(a) else None for a in args]

        def G_single(u_b, *args_b):
            processed_args = [
                a_b.unsqueeze(0) if torch.is_tensor(a_b) else a_b for a_b in args_b
            ]
            return G(u_b.unsqueeze(0), *processed_args).squeeze(0)
        
        r_mag = []

        # Newton iterations loop
        with torch.no_grad():
            for solve_itr in range(max_iters):
                r = G(u, *args)
                r_mag.append(torch.sqrt((r**2).mean()).item())
                if torch.mean((torch.sqrt((r**2).mean(dim=-1)) < tol).float()) >= exit_ratio:
                    break

                def jac_single(u_b, *args_b):
                    return jacrev(lambda x: G_single(x, *args_b))(u_b)

                J = vmap(jac_single, in_dims=(0, *in_mask))(u, *args)

                delta_u = torch.linalg.solve(J, -r.unsqueeze(-1)).squeeze(-1)
                u = u + delta_u
                
        with torch.no_grad():
            converge_flag = torch.sqrt((r**2).mean(dim=-1)) < tol

        # save data to ctx for backward
        ctx.G = G
        ctx.tensor_mask = tensor_mask
        ctx.non_tensor_args = non_tensor_args
        ctx.args_require_grad = [torch.is_tensor(a) and a.requires_grad for a in args]
        ctx.save_for_backward(u.detach(), *[a.detach() for a in tensor_args])

        # TODO make this a dict with the number of iterations...
        out_dict = { "n_iters": solve_itr + 1, "r_mag": r_mag, "converged": converge_flag}
        return u.detach(), out_dict
        # return u.detach(), converge_flag

    @staticmethod
    def backward(ctx, grad_output, _):
        # underscore _ is for converge_flag which has no grad
        G = ctx.G
        saved = ctx.saved_tensors
        u_star = saved[0]
        tensor_args = list(saved[1:])
        non_tensor_args = ctx.non_tensor_args
        tensor_mask = ctx.tensor_mask
        requires_grad_flags = ctx.args_require_grad

        # reconstruct original args in call order (tensors + non-tensors)
        all_args = []
        t_iter = iter(tensor_args)
        nt_iter = iter(non_tensor_args)
        for is_tensor in tensor_mask:
            all_args.append(next(t_iter) if is_tensor else next(nt_iter))

        # Upstream grad on the solved state u*
        grad_output = grad_output.detach()
        B, N = grad_output.shape

        # batch of adjoints lambda via IFT
        def G_single(u, *a):
            proc = [x.unsqueeze(0) if torch.is_tensor(x) else x for x in a]
            return G(u.unsqueeze(0), *proc).squeeze(0)

        # determine which args are batched at dim 0 (so vmap knows what to map over)
        in_mask = []
        for a in all_args:
            if torch.is_tensor(a) and a.ndim > 0 and a.shape[0] == B:
                in_mask.append(0)
            else:
                in_mask.append(None)

        # J_Gu[b] = dG/du evaluated at (u_star[b], args_b)
        def JGu_of_sample(u_b, *args_b):
            return jacrev(lambda x: G_single(x, *args_b))(u_b)

        J_Gu = vmap(JGu_of_sample, in_dims=(0, *in_mask))(u_star, *all_args)

        # solve adjoint system: (J_Gu^T) lambda = dL/du*
        lam = torch.linalg.solve(J_Gu.transpose(-1, -2), grad_output)

        # rebuild a batched G(u*, args) under grad
        with torch.enable_grad():
            u_req = u_star.detach().requires_grad_(True)

            rebuilt_args = []
            tensor_slots = []
            for i, a in enumerate(all_args):
                if torch.is_tensor(a):
                    need = bool(requires_grad_flags[i])
                    # DON'T detach! Just use the original tensor
                    if need:
                        a_req = a.requires_grad_(True)
                        rebuilt_args.append(a_req)
                        tensor_slots.append(i)
                    else:
                        rebuilt_args.append(a)
                else:
                    rebuilt_args.append(a)

            G_out_full = G(u_req, *rebuilt_args)

        # one autograd.grad to get VJPs for tensors & module params
        # collect shared module parameters (not inputs to the Function; we’ll add .grad manually)
        param_list = []
        for a in all_args:
            if isinstance(a, torch.nn.Module):
                for p in a.parameters():
                    if p.requires_grad:
                        param_list.append(p)

        # Inputs for VJP: u*, tensor args that require grad, and shared params
        inputs_for_grad = [u_req]
        inputs_for_grad += [rebuilt_args[i] for i in tensor_slots]
        inputs_for_grad += param_list

        vjps = torch.autograd.grad(
            outputs=G_out_full,
            inputs=inputs_for_grad,
            grad_outputs=-lam,
            retain_graph=True,
            allow_unused=True,
        )

        # split VJPs
        vjp_u = vjps[0]
        vjp_tensor_args = vjps[1:1 + len(tensor_slots)]
        vjp_params = vjps[1 + len(tensor_slots):]

        # build return list of grads for *args only (match Function inputs)
        # return grads for (G, u_init, solver_params, *args)
        n_args = len(all_args)
        grad_args = [None] * n_args

        # fill tensor arg grads into their original slots
        for slot, g in zip(tensor_slots, vjp_tensor_args):
            grad_args[slot] = g

        # convert exact zeros to None (may not be needed / clean up step)
        for i, g in enumerate(grad_args):
            if torch.is_tensor(g) and g.numel() > 0 and torch.all(g == 0):
                grad_args[i] = None

        # for nn.Module params, accumulate grads onto shared module parameters
        for p, gp in zip(param_list, vjp_params):
            if gp is None:
                continue
            if p.grad is None:
                p.grad = gp.detach().clone()
            else:
                p.grad = p.grad + gp.detach()

        # return grads aligned with Function inputs
        # None for G, u_init, solver_params, then grads for *args
        return (None, None, None, *grad_args)



class NonlinearSolver(torch.nn.Module):
    """
    General-purpose batched implicit Newton solver.
    This class is a simple wrapper for the IFTNewtonSolver autograd Function with some settings.
    Use this to call forward and backward passes easily.
    G: callable that accepts batched inputs (B, N, ...) and returns (B, N, ...).
    Example:
        solver = NonlinearSolver(G)
        u_star = solver(u_init, *args)
    """

    def __init__(self, G, max_iters=20, tol=1e-6, exit_ratio=0.80):
        super().__init__()
        self.G = G
        self.max_iters = max_iters
        self.tol = tol
        self.exit_ratio = exit_ratio

    def forward(self, u_init, *args):
        solver_params = {
            "max_iters": self.max_iters,
            "tol": self.tol,
            "exit_ratio": self.exit_ratio}
        return IFTNewtonSolver.apply(self.G, u_init, solver_params, *args)


if __name__ == "__main__":
    # Simple nonlinear function with some parameters and a model input
    def G(u, a, b, c_model):
        return u**2 + a * u + b + c_model(u)

    B = 5
    u_init = torch.randn(B, 1, requires_grad=True)
    a = torch.randn(B, 1, requires_grad=True)
    b = torch.randn(B, 1, requires_grad=True)
    c_model = torch.nn.Sequential(
        torch.nn.Linear(1, 1),
        torch.nn.ReLU()
    )

    # solves a function of the type G(u|meta_params) = 0
    # in a differentiable way using implicit function theorem
    # implements newton's method for root finding in forward pass
    # with batched support and flexible input types (to an extent)
    solver = NonlinearSolver(G)

    # call the solve with some batched tensor and some functions
    u_star = solver(u_init, a, b, c_model)
