import torch

import src.utils as utils
from src.nonlinear_solver import NonlinearSolver


# wrapper class to hold all components of the GeoNew model, and implement the forward pass and residual computation for the nonlinear solver
class GeoNew(torch.nn.Module):

    def __init__(
        self,
        encoder,
        source_model,
        boundary_w,
        flux_model,
        n_pou=16,
        n_fields=3,
        device="cpu",
        max_iters=50,
        tol=1e-5,
        solve_exit_ratio=0.8,
        source_mode="zero",  # "zero", "zero_mean", "constant", "unconstrained"
    ):
        super().__init__()
        self.device = device
        self.n_pou = n_pou
        self.n_fields = n_fields
        self.source_mode = source_mode

        self.n_bc_pou = 3
        self.full_n_pou = n_pou + self.n_bc_pou
        self.n_one_forms = self.full_n_pou * (self.full_n_pou - 1) // 2

        self.encoder = encoder
        self.source_model = source_model
        self.boundary_w = boundary_w
        self.flux_model = flux_model

        self.setup_templates()
       
        # nonlinear solver just needs the residual function and solver parameters
        self.solver = NonlinearSolver(self.G_residual,
            max_iters=max_iters, tol=tol, exit_ratio=solve_exit_ratio)

        self.register_buffer("_lipschitz_ema", torch.tensor(0.0))
        self.register_buffer("_lipschitz_ema_initialized", torch.tensor(False))
        self._lipschitz_ema_alpha = 0.6

    def setup_templates(self):
        # precompute delta_0 and masks for BCs
        device = self.device
        self.delta_0_template = utils.construct_delta0(self.full_n_pou).to(device)
        _, _, self.edge_pairs = utils.extract_edge_endpoints(self.delta_0_template)
        self.edge_pairs = self.edge_pairs.to(device)

        self.bilinear_boundary_masks_template = torch.zeros(
            self.full_n_pou, dtype=torch.bool, device=device
        )
        self.bilinear_boundary_masks_template[-self.n_bc_pou :] = True
        self.latent_boundary_masks_template = torch.zeros(
            self.full_n_pou, self.n_fields, dtype=torch.bool, device=device
        )
        self.latent_boundary_masks_template[-self.n_bc_pou :] = True
        self.latent_boundary_vals_template = torch.zeros(
            self.full_n_pou, self.n_fields, device=device
        )
        self.latent_boundary_vals_template[-self.n_bc_pou, :] = 1.0

    def expand_masks(self, B):
        delta_0 = self.delta_0_template.unsqueeze(0).expand(B, -1, -1)
        bilinear_mask = self.bilinear_boundary_masks_template.unsqueeze(0).expand(B, -1)
        latent_mask = self.latent_boundary_masks_template.unsqueeze(0).expand(B, -1, -1)
        latent_vals = self.latent_boundary_vals_template.unsqueeze(0).expand(B, -1, -1)
        return delta_0, bilinear_mask, latent_mask, latent_vals

    def get_full_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # define the residual function for the nonlinear solver
    # for our implementation, this is G(u) := Ku - Mf - D F(u), with dirichlet BCs.
    # and we solve for u such that G(u) = 0
    def G_residual(
        self, u_hat_flat, flux_model, z_latents, f_latent, K_hat, M_hat, delta_0, bc_mask, bc_vals,
    ):
        B = u_hat_flat.shape[0]
        u_hat = u_hat_flat.view(B, self.full_n_pou, self.n_fields)

        diff_term = torch.einsum("bijf,bjf->bif", K_hat, u_hat)
        src_term = torch.einsum("bijf,bjf->bif", M_hat, f_latent)
        flux = flux_model(z_latents, u_hat, self.edge_pairs)
        flux_term = torch.einsum("bji,bjf->bif", delta_0, flux)

        r = diff_term - src_term - flux_term
        r = utils.apply_dirichlet_residual(r, u_hat, bc_mask, bc_vals)
        return r.reshape(B, -1).contiguous()

    def project_and_bc(self, K_list, M_list, W, bilinear_mask, latent_scaling=1.0):
        K_hat = utils.project_bilinear_per_field(K_list, W, latent_scaling)
        M_hat = utils.project_bilinear_per_field(M_list, W, latent_scaling)
        K_hat_bc = utils.apply_dirichlet_per_field(K_hat, bilinear_mask)
        M_hat_bc = utils.apply_dirichlet_per_field(M_hat, bilinear_mask)
        return K_hat_bc, M_hat_bc
    
    def forward(
        self, in_tokens, K_list, M_list, dirichlet_nodes, boundary_vals # query_tokens, adj,
    ):
        B, N = in_tokens.shape[:2]
        delta_0, bilinear_mask, latent_mask, latent_boundary_vals = self.expand_masks(B)
        latent_scaling = N / self.full_n_pou # for solver stability, uniform energy scaling

        z_unpooled = self.encoder(in_tokens)[0]

        W = self.boundary_w(z_unpooled, z_unpooled, dirichlet_nodes, boundary_vals)

        f_full = self.source_model(z_unpooled).view(B, N, self.n_fields)
        f_latent = torch.einsum("bjif,bif->bjf", W, f_full) / latent_scaling

        # Apply source mode constraint
        f_latent = utils.process_source(f_latent, self.source_mode)

        K_hat_bc, M_hat_bc = self.project_and_bc(
            K_list, M_list, W, bilinear_mask, latent_scaling
        )

        u_init = torch.randn(
            B, self.full_n_pou * self.n_fields, device=in_tokens.device
        )

        u_hat, solve_out = self.solver(
            u_init,
            self.flux_model,
            z_unpooled,
            f_latent,
            K_hat_bc,
            M_hat_bc,
            delta_0,
            latent_mask,
            latent_boundary_vals,
        )

        u_hat_coarse = u_hat.view(B, self.full_n_pou, self.n_fields)
        u_fine = torch.einsum("bjif,bjf->bif", W, u_hat_coarse)

        return {
            "u_fine": u_fine,
            "u_coarse": u_hat_coarse,
            "residual_flat": solve_out["r_mag"], # will need to be adjusted later to expect this output format
            "W": W,
            "converged": solve_out["converged"],
            "n_iters": solve_out["n_iters"],
        }

    # if we want a minimal forward interface which just returns the solution on original mesh
    def simple_forward(self, *args, **kwargs):
        return self.forward(*args, **kwargs)["u_fine"]


if __name__ == "__main__":
    pass
