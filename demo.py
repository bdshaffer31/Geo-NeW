
import torch

from models import InducingPointEncoder, MLPSourceModel, MLPLearnableW, BoundaryWPerField, StableHyperFlux
from geo_new import GeoNew

import utils


def _make_sparse_identity_list(B: int, N: int, device=None, dtype=torch.float32):
    device = device or "cpu"
    idx = torch.arange(N, device=device)
    indices = torch.stack([idx, idx], dim=0)
    values = torch.ones((N,), device=device, dtype=dtype)
    I = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()
    return [I for _ in range(B)]

# revisit these args from original implementation
def setup_GeoNew_model(
        n_pou=16,
        n_fields=3,
        encoder_in_dim=2,
        model_dim=128,
        n_heads=4,
        n_layers=4,
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

    delta_0_template = utils.construct_delta0(full_n_pou)
    # extract_edge_endpoints returns (pos_idx, neg_idx, edge_pairs)
    edge_pairs = utils.extract_edge_endpoints(delta_0_template)[2]

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
    

def main():
    geo_new = setup_GeoNew_model()

    # Minimal smoke test to validate forward I/O contracts.
    B, N = 2, 32
    n_fields = geo_new.n_fields
    in_dim = geo_new.encoder.input_proj.in_features
    device = next(geo_new.parameters()).device

    in_tokens = torch.randn(B, N, in_dim, device=device)
    query_tokens = in_tokens  # currently unused by GeoNew.forward
    adj = None  # currently unused by GeoNew.forward

    # Per-sample sparse operators (identity is enough for a wiring test)
    K_list = _make_sparse_identity_list(B, N, device=device)
    M_list = _make_sparse_identity_list(B, N, device=device)

    # Dirichlet masks/values per node and field
    dirichlet_nodes = (torch.rand(B, N, n_fields, device=device) < 0.1)
    boundary_vals = torch.zeros(B, N, n_fields, device=device)

    out = geo_new(
        in_tokens=in_tokens,
        query_tokens=query_tokens,
        adj=adj,
        K_list=K_list,
        M_list=M_list,
        dirichlet_nodes=dirichlet_nodes,
        boundary_vals=boundary_vals,
    )

    print("u_fine:", tuple(out["u_fine"].shape))
    print("u_coarse:", tuple(out["u_coarse"].shape))
    print("converged:", out["converged"].float().mean().item(), "mean")
    print("n_iters:", out["n_iters"])

if __name__ == "__main__":
    main()