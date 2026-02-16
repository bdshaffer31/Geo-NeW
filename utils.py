import torch
import numpy as np
from scipy.sparse import lil_matrix


def make_attention_mask_batched(
    adjacency_mask, n_global_connects=16, n_hops=1, nhead=1
):
    B, N, _ = adjacency_mask.size()
    allow_mask = adjacency_mask.clone()
    for i in range(n_hops - 1):
        allow_mask = allow_mask | (allow_mask @ allow_mask).bool()
    for b in range(B):
        allow_mask[b].fill_diagonal_(True)

    # add random global connectors
    global_connects = torch.rand((B, N, N), device=adjacency_mask.device) < (
        n_global_connects / N
    )
    allow_mask = allow_mask | global_connects

    attn_mask = torch.zeros(
        (B, N, N), device=adjacency_mask.device, dtype=torch.float32
    )
    attn_mask[~allow_mask] = float("-inf")

    if nhead > 1:
        attn_masks_full = attn_mask.unsqueeze(1).repeat(1, nhead, 1, 1)
        attn_mask = attn_masks_full.reshape(-1, attn_mask.size(1), attn_mask.size(2))

    return attn_mask


def apply_dirichlet_bilinear(K, boundary_mask):
    K_bc = K.clone()

    # zero rows whereon boudnary POUs and add identity on diagonal
    interior_mask = (~boundary_mask).unsqueeze(-1).float()
    K_bc = K_bc * interior_mask

    I = torch.eye(K.shape[-1], device=K.device).unsqueeze(0)
    boundary_diag = boundary_mask.unsqueeze(-1).float() * I
    K_bc = K_bc + boundary_diag

    return K_bc



def construct_delta0(Npou, device="cpu"):
    """Incidence matrix for 0 to 1 forms (complete graph)."""
    n1 = Npou * (Npou - 1) // 2
    delta0 = torch.zeros(n1, Npou, device=device)
    k = 0
    for i in range(Npou):
        for j in range(i + 1, Npou):
            delta0[k, i] = -1.0
            delta0[k, j] = 1.0
            k += 1
    return delta0


# from the delta_0 incidence matrix, extract edge endpoints
def extract_edge_endpoints(delta_0):
    nz_pos = delta_0 == 1.0
    nz_neg = delta_0 == -1.0
    pos_idx = nz_pos.float().argmax(dim=1)
    neg_idx = nz_neg.float().argmax(dim=1)
    edge_pairs = torch.stack([pos_idx, neg_idx], dim=1)
    return pos_idx, neg_idx, edge_pairs


def compute_adj_sparse(mesh):
    N = mesh.p.shape[1]
    adj = lil_matrix((N, N), dtype=bool)
    for elem in mesh.t.T:
        for i in range(len(elem)):
            for j in range(len(elem)):
                adj[elem[i], elem[j]] = True
    return adj.tocsr()


def scipy_sparse_to_torch_sparse(sparse_mtx, dtype=torch.float32):
    sparse_mtx = sparse_mtx.tocoo()
    indices = torch.tensor(np.array([sparse_mtx.row, sparse_mtx.col]), dtype=torch.long)
    values = torch.tensor(sparse_mtx.data, dtype=dtype)
    shape = torch.Size(sparse_mtx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def make_masked_outer_W(W, boundary_mask, boundary_vals):
    # Add three boundary POUs
    # shape, constant, complement
    B, Npou, N = W.shape
    new_W = torch.zeros(B, Npou + 3, N, device=W.device, dtype=W.dtype)
    new_W[:, :Npou, :] = W
    mask = boundary_mask.unsqueeze(1).expand(B, Npou, N)
    new_W[:, :Npou, :] = new_W[:, :Npou, :].masked_fill(mask, 0.0)

    for b in range(B):
        bm = boundary_mask[b]
        new_W[b, -3, bm] = boundary_vals[b, bm]
    new_W[:, -2, :] = boundary_mask.float()    
    comp = 1.0 - new_W[:, -3, :] - new_W[:, -2, :]
    comp = torch.clamp(comp, min=0.0)
    new_W[:, -1, :] = comp * boundary_mask.float()

    return new_W



def apply_dirichlet_residual(r, u, boundary_mask, u_D=None):
    if u_D is None:
        u_D = torch.zeros_like(u)
    if len(u.shape) == 2:
        u = u.unsqueeze(-1)
        u_D = u_D.unsqueeze(-1)
    boundary_mask = boundary_mask.expand_as(u)
    mask = boundary_mask.float()
    return (1.0 - mask) * r + mask * (u - u_D)


def apply_dirichlet_latent(u_full, boundary_mask, u_b=None):
    if u_b is None:
        u_b = torch.zeros_like(u_full)
    if len(u_full.shape) == 2:
        u_full = u_full.unsqueeze(-1)
        u_b = u_b.unsqueeze(-1)
    boundary_mask = boundary_mask.expand_as(u_full)
    mask = boundary_mask.float()
    return (1.0 - mask) * u_full + mask * (u_b)


def project_bilinear_form(A, W):
    # use sparse dense mm over batched A and W
    A_hat = []
    for b in range(W.shape[0]):
        tmp = torch.sparse.mm(A[b], W[b].T)
        A_hat.append(W[b] @ tmp)
    A_hat = torch.stack(A_hat)
    return A_hat


def relative_l2_error(u, u_pred):
    """
    Compute relative L2 error between u and u_pred.
    """
    error = u - u_pred
    error_norm = torch.norm(error)
    value_norm = torch.norm(u)
    rel_l2 = error_norm / value_norm.clamp(min=1e-6)
    return rel_l2


def relative_l2_error_per_field_vector(u, u_pred, eps: float = 1e-6):
    """Compute per-field relative L2 error.

    Returns:
        Tensor of shape (F,) where F = u.shape[-1]
    """
    rel_l2_per_field = []
    for f in range(u.shape[-1]):
        error_f = u[..., f] - u_pred[..., f]
        error_norm = torch.norm(error_f)
        value_norm = torch.norm(u[..., f])
        rel_l2 = error_norm / value_norm.clamp(min=eps)
        rel_l2_per_field.append(rel_l2)
    return torch.stack(rel_l2_per_field)

def relative_l2_error_per_field(u, u_pred):
    """
    Compute relative L2 error between u and u_pred per field.
    """
    return relative_l2_error_per_field_vector(u, u_pred).mean()


def relative_l2_error_unpadded(u, u_pred, n_original_nodes, eps: float = 1e-6) -> torch.Tensor:
    """Relative L2 error excluding padded nodes.

    Computes:
        ||u - u_pred|| / ||u||
    but only over nodes 0:n_original_nodes[b] for each sample b.

    This matches the usual (global) relative L2 definition, just with padding
    excluded from numerator and denominator.

    Args:
        u: (B,N,...) or (N,...)
        u_pred: same shape as u
        n_original_nodes: (B,) or scalar; number of valid nodes per sample
        eps: denominator clamp

    Returns:
        Scalar tensor (on u.device)
    """

    if u.shape != u_pred.shape:
        raise ValueError(f"u and u_pred must have same shape, got {tuple(u.shape)} vs {tuple(u_pred.shape)}")

    if u.ndim == 1:
        # (N,)
        n0 = int(n_original_nodes.item()) if torch.is_tensor(n_original_nodes) else int(n_original_nodes)
        n0 = max(0, min(n0, int(u.shape[0])))
        if n0 == 0:
            return torch.tensor(float("nan"), device=u.device, dtype=u.dtype)
        err = u[:n0] - u_pred[:n0]
        return torch.norm(err) / torch.norm(u[:n0]).clamp(min=eps)

    if u.ndim == 2:
        # (N,D)
        n0 = int(n_original_nodes.item()) if torch.is_tensor(n_original_nodes) else int(n_original_nodes)
        n0 = max(0, min(n0, int(u.shape[0])))
        if n0 == 0:
            return torch.tensor(float("nan"), device=u.device, dtype=u.dtype)
        err = u[:n0] - u_pred[:n0]
        return torch.norm(err) / torch.norm(u[:n0]).clamp(min=eps)

    # Batched: assume node dimension is dim=1.
    if u.ndim < 3:
        raise ValueError(f"Unsupported u dims for unpadded metric: {u.ndim}")

    B = int(u.shape[0])
    if torch.is_tensor(n_original_nodes):
        n_list = n_original_nodes.detach().reshape(-1).tolist()
    else:
        n_list = [int(n_original_nodes)] * B

    num_sq = torch.zeros((), device=u.device, dtype=torch.float32)
    den_sq = torch.zeros((), device=u.device, dtype=torch.float32)
    for b in range(B):
        n0 = int(n_list[b])
        n0 = max(0, min(n0, int(u.shape[1])))
        if n0 == 0:
            continue
        ub = u[b, :n0]
        upb = u_pred[b, :n0]
        err = (ub - upb).to(dtype=torch.float32)
        ub_f = ub.to(dtype=torch.float32)
        num_sq = num_sq + torch.sum(err * err)
        den_sq = den_sq + torch.sum(ub_f * ub_f)

    return torch.sqrt(num_sq) / torch.sqrt(den_sq).clamp(min=eps)


def relative_l2_error_per_field_unpadded(u, u_pred, n_original_nodes, eps: float = 1e-6) -> torch.Tensor:
    """Per-field relative L2 error excluding padded nodes.

    Args:
        u: (B,N,F) or (N,F)
        u_pred: same shape
        n_original_nodes: (B,) or scalar

    Returns:
        Tensor of shape (F,)
    """

    if u.shape != u_pred.shape:
        raise ValueError(f"u and u_pred must have same shape, got {tuple(u.shape)} vs {tuple(u_pred.shape)}")
    if u.ndim == 2:
        n0 = int(n_original_nodes.item()) if torch.is_tensor(n_original_nodes) else int(n_original_nodes)
        n0 = max(0, min(n0, int(u.shape[0])))
        if n0 == 0:
            return torch.full((int(u.shape[-1]),), float("nan"), device=u.device, dtype=u.dtype)
        return relative_l2_error_per_field_vector(u[:n0], u_pred[:n0], eps=eps)
    if u.ndim != 3:
        raise ValueError(f"Expected (B,N,F) or (N,F), got {tuple(u.shape)}")

    B, N, F = int(u.shape[0]), int(u.shape[1]), int(u.shape[2])
    if torch.is_tensor(n_original_nodes):
        n_list = n_original_nodes.detach().reshape(-1).tolist()
    else:
        n_list = [int(n_original_nodes)] * B

    num_sq = torch.zeros((F,), device=u.device, dtype=torch.float32)
    den_sq = torch.zeros((F,), device=u.device, dtype=torch.float32)
    for b in range(B):
        n0 = int(n_list[b])
        n0 = max(0, min(n0, N))
        if n0 == 0:
            continue
        ub = u[b, :n0].to(dtype=torch.float32)      # (n0,F)
        upb = u_pred[b, :n0].to(dtype=torch.float32)
        err = ub - upb
        num_sq = num_sq + torch.sum(err * err, dim=0)
        den_sq = den_sq + torch.sum(ub * ub, dim=0)

    return torch.sqrt(num_sq) / torch.sqrt(den_sq).clamp(min=eps)

def process_source(source, source_mode):
    """
    Apply source mode constraint.
        Processed source based on source_mode:
        - "zero": returns zeros
        - "zero_mean": subtracts spatial mean per field
        - "constant": returns spatial mean per field (constant everywhere)
        - "unconstrained": returns source unchanged
    """
    if source_mode == "zero":
        return torch.zeros_like(source)
    elif source_mode == "zero_mean":
        mean = source.mean(dim=1, keepdim=True)
        return source - mean
    elif source_mode == "constant":
        mean = source.mean(dim=1, keepdim=True)
        return mean.expand_as(source)
    elif source_mode == "unconstrained":
        return source
    else:
        raise ValueError(f"Unknown source_mode: {source_mode}")


def perturb_cnwf_weights(cwnf_model, noise_scale=0.01):
    modules_to_perturb = [
        cwnf_model.encoder,
        cwnf_model.flux_model,
        cwnf_model.source_model,
        cwnf_model.w_inner,
    ]
    
    with torch.no_grad():
        for module in modules_to_perturb:
            for param in module.parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * noise_scale
                    param.add_(noise)
    
    print(f"  → Perturbed model weights (noise_scale={noise_scale})")

def project_bilinear_per_field(A_list, W, latent_scaling):
    B, P, N, F = W.shape
    A_hats = []
    for f in range(F):
        Wf = W[..., f]
        Ah = project_bilinear_form(A_list, Wf) / latent_scaling
        A_hats.append(Ah.unsqueeze(-1))
    return torch.cat(A_hats, dim=-1)

def apply_dirichlet_per_field(A_hat, bilinear_mask):
    B, P, _, F = A_hat.shape
    out = []
    for f in range(F):
        Af = A_hat[..., f]
        Af_bc = apply_dirichlet_bilinear(Af, bilinear_mask)
        out.append(Af_bc.unsqueeze(-1))
    return torch.cat(out, dim=-1)

def interior_nmse(u, u_pred, boundary_mask):
    """
    Compute normalized MSE on interior nodes.
    """
    error = u - u_pred
    interior_error = error[~boundary_mask]
    interior_vals = u[~boundary_mask]
    # nmse = ((interior_error) ** 2).mean(dim=0) / (interior_vals**2).mean(dim=0).clamp(min=1e-6)
    # print(nmse)
    error_norm = torch.norm(interior_error)
    value_norm = torch.norm(interior_vals)
    rel_l2 = error_norm / value_norm.clamp(min=1e-6)
    return rel_l2


def interior_nmse_per_field(u, u_pred, dirichlet_nodes, return_per_field=False):
    """
    Compute NMSE on interior (non-Dirichlet) nodes per field.
    
    Args:
        u: (B, N, F) ground truth fields
        u_pred: (B, N, F) predicted fields
        dirichlet_nodes: (B, N, F) boolean mask of Dirichlet nodes per field
        
    Returns:
        nmse: scalar mean NMSE across all fields
    """

    error = u - u_pred
    # Compute NMSE for each field separately
    nmse_per_field = []
    for f in range(u.shape[-1]):
        interior_mask_f = ~dirichlet_nodes[..., f]
        interior_error_f = error[..., f][interior_mask_f]
        interior_vals_f = u[..., f][interior_mask_f]

        # nmse_f = (interior_error_f ** 2).mean() / (interior_vals_f ** 2).mean().clamp(min=1e-6)
        error_norm = torch.norm(interior_error_f)
        value_norm = torch.norm(interior_vals_f)
        rel_l2 = error_norm / value_norm.clamp(min=1e-6)
        nmse_per_field.append(rel_l2)

    nmse_per_field = torch.stack(nmse_per_field)
    
    if return_per_field:
        return nmse_per_field
    else:
        return nmse_per_field.mean()


def boundary_relative_l2_per_field(u, u_pred, dirichlet_nodes, return_per_field=False):
    """
    Compute normalized L2 error on the Dirichlet (boundary) nodes per field.

    The numerator is restricted to Dirichlet nodes, but the normalization
    denominator uses the full-field L2 norm (all nodes). This avoids degenerate
    normalization when the boundary truth is (near) zero.

    Args:
        u: (B, N, F) ground truth
        u_pred: (B, N, F) prediction
        dirichlet_nodes: (B, N, F) boolean mask of Dirichlet nodes per field
        return_per_field: if True return a tensor of shape (F,) with per-field errors,
                          otherwise return the mean across fields.

    Returns:
        Tensor of shape (F,) or scalar mean across fields.
    """

    error = u - u_pred
    rel_per_field = []
    F = u.shape[-1]
    for f in range(F):
        mask_f = dirichlet_nodes[..., f]
        # select all boundary entries across the batch
        err_f = error[..., f][mask_f]
        # normalize by full-field magnitude (across all nodes)
        vals_f_full = u[..., f]
        if err_f.numel() == 0:
            # No boundary nodes for this field in the batch; return 0.0 to avoid NaNs
            rel = torch.tensor(0.0, device=u.device)
        else:
            rel = torch.norm(err_f) / torch.norm(vals_f_full).clamp(min=1e-6)
        rel_per_field.append(rel)

    rel_per_field = torch.stack(rel_per_field)
    if return_per_field:
        return rel_per_field
    return rel_per_field.mean()

