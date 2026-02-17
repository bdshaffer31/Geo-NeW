import torch
from torch.utils.data import Dataset, DataLoader

from src.geo_new import GeoNew
from src.models import (
    BoundaryWPerField,
    InducingPointEncoder,
    MLPLearnableW,
    MLPSourceModel,
    StableHyperFlux,
)


def setup_GeoNew_model(
        n_pou=16,
        n_fields=1,
        encoder_in_dim=2,
        model_dim=64,
        n_heads=2,
        n_layers=2,
        encoder_dropout=0.0,
        resample_anchors=False,
        w_alpha_init=0.5,
        w_dropout=0.0,
        w_temperature=1.0,
        flux_dim=128,
        flux_layers=4,
        flux_lipschitz_max=1.0,
        flux_gamma_init=1.0,
        flux_mean_features=False,
        flux_easein=False,
        use_base_ops=False,
        id_init_base=False,
        zero_init_ops=False,
        spectral_upper_bound=None,
        no_flux_conditioning=False,
):

    n_bc_pou = 3
    full_n_pou = n_pou + n_bc_pou

    delta_0_template = construct_delta0(full_n_pou)
    # extract_edge_endpoints returns (pos_idx, neg_idx, edge_pairs)
    edge_pairs = extract_edge_endpoints(delta_0_template)[2]

    encoder = InducingPointEncoder(
        in_dim=encoder_in_dim,
        d_model=model_dim,
        nhead=n_heads,
        num_anchor_layers=n_layers,
        dropout=encoder_dropout,
        resample_anchors=resample_anchors,
    )

    source_model = MLPSourceModel(model_dim=model_dim, n_fields=n_fields)

    w_inner = MLPLearnableW(
        n_pou=n_pou,
        n_fields=n_fields,
        in_dim=model_dim,
        model_dim=model_dim,
        alpha_init=w_alpha_init,
        dropout=w_dropout,
    )
    boundary_w = BoundaryWPerField(w_inner, temperature=w_temperature)

    flux_model = StableHyperFlux(
        n_fields=n_fields,
        latent_dim=model_dim,
        hidden=flux_dim,
        n_layers=flux_layers,
        lipschitz_max=flux_lipschitz_max,
        gamma_init=flux_gamma_init,
        use_mean_features=flux_mean_features,
        easein=flux_easein,
        n_pous=full_n_pou,
        edge_pairs=edge_pairs,
        use_base_ops=use_base_ops,
        id_init_base=id_init_base,
        zero_init_ops=zero_init_ops,
        spectral_upper_bound=spectral_upper_bound,
        no_flux_conditioning=no_flux_conditioning,
    )

    geo_new = GeoNew(
        encoder=encoder,
        source_model=source_model,
        boundary_w=boundary_w,
        flux_model=flux_model,
        n_pou=n_pou,
        n_fields=n_fields,
    )
    return geo_new


class MeshDataset(Dataset):
    """Dataset wrapper for the preprocessed .pt mesh data."""
    
    def __init__(
        self,
        file_name,
        idx_range=None,
        device="cpu",
        hks_steps=8,
        eps=1e-12,
        use_coords=False,
        use_poisson=False,
        use_harmonic=True,
        # excl_boundary_groups=None,
    ):
        self.device = device
        self.hks_steps = hks_steps
        self.eps = eps
        self.use_coords = use_coords
        self.use_poisson = use_poisson
        self.use_harmonic = use_harmonic
        
        # Load data (this file is data, not a model checkpoint).
        try:
            self.data = torch.load(file_name, weights_only=False)
        except TypeError:
            self.data = torch.load(file_name)
        if idx_range is not None:
            self.data = self.data[idx_range[0]:idx_range[1]]
        
        # HKS time scales (log-spaced)
        self.hks_times = torch.logspace(-4, -1, steps=self.hks_steps)

        # self.excl_boundary_groups = excl_boundary_groups
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]

        # Expected layout is the fixed 19-tuple plus optional trailing `details`.
        details = None
        if isinstance(sample, (list, tuple)) and len(sample) >= 19:
            base = sample[:19]
            if len(sample) > 19:
                details = sample[19]
        else:
            raise ValueError(
                f"Unexpected sample structure at idx={idx}: type={type(sample)} len={getattr(sample, '__len__', lambda: 'NA')()}"
            )

        (
            dist_field_t,
            boundary_flags_t,
            dirichlet_nodes_t,
            fields_t,
            global_cond_t,
            coords_t,
            adj_t,
            elements_t,
            mass_matrix,
            stiffness_matrix,
            Gx_matrix,
            Gy_matrix,
            eigvals_t,
            eigvecs_t,
            poisson_t,
            harmonic_t,
            n_original_nodes_t,
            norm_offset_t,
            norm_scale_t,
        ) = base
        
        # boundary mask from flags
        boundary_mask = boundary_flags_t.squeeze(-1).to(torch.bool)

        # Tokens from spectral + geometric features
        tokens = self._compute_tokens(
            coords_t, eigvecs_t, eigvals_t, dist_field_t,
            poisson_t, harmonic_t, Gx_matrix, Gy_matrix, mass_matrix
        )
        
        # Move to device
        tokens = tokens.to(torch.float32).to(self.device)
        fields_t = fields_t.to(torch.float32).to(self.device)
        coords_t = coords_t.to(torch.float32).to(self.device)
        adj_t = adj_t.to(self.device).to_dense()
        M = mass_matrix.to(self.device)
        K = stiffness_matrix.to(self.device)
        boundary_mask = boundary_mask.to(self.device)
        dirichlet_nodes_t = dirichlet_nodes_t.to(self.device)
        n_original_nodes_t = n_original_nodes_t.to(self.device)

        return {
            "coords": coords_t,
            "tokens": tokens,
            "K": K,
            "M": M,
            "adj": adj_t,
            "boundary_mask": boundary_mask,
            "dirichlet_nodes": dirichlet_nodes_t,
            "fields": fields_t,
            "elements": elements_t,
            "n_original_nodes": n_original_nodes_t,
            "details": details,
        }
    
    def _compute_tokens(
        self, coords, eigvecs, eigvals, dist_field, poisson, harmonic,
        Gx_matrix, Gy_matrix, mass_matrix
    ):
        N = eigvecs.shape[0]
        K_evec = eigvecs.shape[1]
        
        phi = eigvecs[:, :K_evec]
        lam = eigvals[:K_evec].clamp_min(self.eps)
        
        # Lumped mass inverse
        m_lumped = torch.sparse.sum(mass_matrix, dim=1).to_dense().reshape(-1)
        inv_m = 1.0 / (m_lumped + self.eps)
        
        # Gradient of eigenvectors
        Gx_phi = inv_m.unsqueeze(-1) * torch.sparse.mm(Gx_matrix, phi)
        Gy_phi = inv_m.unsqueeze(-1) * torch.sparse.mm(Gy_matrix, phi)
        
        # Heat Kernel Signature
        decay = torch.exp(-torch.outer(self.hks_times, lam))
        HKS = torch.einsum("nk,tk->nt", phi * phi, decay)
        HKS = HKS / (HKS.max(dim=0, keepdim=True)[0] + self.eps)
        
        # Gradient features
        grad_x = torch.einsum("nk,nk,tk->nt", phi, Gx_phi, decay)
        grad_y = torch.einsum("nk,nk,tk->nt", phi, Gy_phi, decay)
        
        grad_x = grad_x / (grad_x.abs().max(dim=0, keepdim=True)[0] + self.eps)
        grad_y = grad_y / (grad_y.abs().max(dim=0, keepdim=True)[0] + self.eps)
        
        token_list = [
            HKS,           # (N, hks_steps)
            grad_x,        # (N, hks_steps)
            grad_y,        # (N, hks_steps)
            10 * dist_field,  # (N, 1) - scaled distance to boundary
        ]
        
        if self.use_poisson:
            token_list.append(100 * poisson)  # (N, 1)
        
        if self.use_coords:
            token_list.append(coords)  # (N, 2)
        
        if self.use_harmonic:
            token_list.append(harmonic)  # (N, n_harmonic)
        
        tokens = torch.cat(token_list, dim=-1)
        return tokens


def collate_fn(batch):
    """Batch collator; keeps sparse K/M and elements as Python lists."""
    coords = torch.stack([item["coords"] for item in batch], dim=0)
    tokens = torch.stack([item["tokens"] for item in batch], dim=0)
    K = [item["K"] for item in batch]
    M = [item["M"] for item in batch]
    adj = torch.stack([item["adj"] for item in batch], dim=0)
    boundary_mask = torch.stack([item["boundary_mask"] for item in batch], dim=0)
    dirichlet_nodes = torch.stack([item["dirichlet_nodes"] for item in batch], dim=0)
    fields = torch.stack([item["fields"] for item in batch], dim=0)
    elements = [item["elements"] for item in batch]  # List of tensors (variable size)
    n_original_nodes = torch.stack([item["n_original_nodes"] for item in batch], dim=0)
    details = [item.get("details", None) for item in batch]

    return {
        "coords": coords,
        "tokens": tokens,
        "K": K,
        "M": M,
        "adj": adj,
        "boundary_mask": boundary_mask,
        "dirichlet_nodes": dirichlet_nodes,
        "fields": fields,
        "elements": elements,
        "n_original_nodes": n_original_nodes,
        "details": details,
    }


def create_dataloaders(
    train_file,
    val_file=None,
    train_range=None,
    val_range=None,
    batch_size=32,
    device="cpu",
    hks_steps=8,
    num_workers=0,
    **dataset_kwargs
):
    """Create train/val dataloaders for a processed dataset."""
    train_dataset = MeshDataset(
        train_file,
        idx_range=train_range,
        device=device,
        hks_steps=hks_steps,
        **dataset_kwargs
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    
    if val_file is None and val_range is None:
        return train_dataloader
    
    val_dataset = MeshDataset(
        val_file if val_file is not None else train_file,
        idx_range=val_range,
        device=device,
        hks_steps=hks_steps,
        **dataset_kwargs
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    
    return train_dataloader, val_dataloader


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


def extract_edge_endpoints(delta_0):
    nz_pos = delta_0 == 1.0
    nz_neg = delta_0 == -1.0
    pos_idx = nz_pos.float().argmax(dim=1)
    neg_idx = nz_neg.float().argmax(dim=1)
    edge_pairs = torch.stack([pos_idx, neg_idx], dim=1)
    return pos_idx, neg_idx, edge_pairs


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


def project_bilinear_form(A, W):
    # use sparse dense mm over batched A and W
    A_hat = []
    for b in range(W.shape[0]):
        tmp = torch.sparse.mm(A[b], W[b].T)
        A_hat.append(W[b] @ tmp)
    A_hat = torch.stack(A_hat)
    return A_hat


def relative_l2_error_unpadded(u, u_pred, n_original_nodes, eps=1e-6):
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

