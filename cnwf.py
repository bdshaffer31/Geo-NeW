import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import utils

from geometry_encoders.helpers import build_geometry_encoder
from nonlinear_solver import NonlinearSolver


# define Flux, W, and Source models.
# then combine them with the geometry encoders and poolers in the CNWF class
# use these to define the residual and solve with the nonlinear solver.

# # TODO verify the lipschitz regularity computation is correct
class StableHyperFlux(nn.Module):
    """
    Solvability-safe antisymmetric flux with task-specific token processing:
    """

    def __init__(
        self,
        n_fields: int,
        latent_dim: int,
        proj_dim: int = 16,
        hidden: int = 128,
        n_layers: int = 2,
        proj_init_gain = 0.1,
        lipschitz_max: float = None,
        # lipschitz_scale: float = None,
        n_flux_queries: int = 8,  # number of learnable queries for cross-attention
        gamma_init: float = -2.0,  # initial value for gamma_raw parameter
        beta_init: float = 0.0,  # initial value for beta_raw parameter
        use_mean_features: bool = True, # use difference and mean of u_i u_j as features
        # anti_symmetric: bool = True,  # enforce antisymmetric flux, actually not needed, always antisymettric
        easein: int = 0,  # number of epochs to ease in nonlinearity multiplier
        # make the args options
        use_base_ops: bool = True,
        id_init_base: bool = True,
        zero_init_ops: bool = True,
        spectral_upper_bound: bool = False, # only enforce spectral normalization upper bound on operators
        no_flux_conditioning: bool = False,
        n_pous: int = 32,
        edge_pairs = None,
        lumped_L: bool = True, # if lumped learn only diagonal of L factor
    ):
        super().__init__()
        self.n_fields = n_fields
        self.proj_dim = proj_dim

        # initialize lipshitz buffers and parameters, clunky for now.
        init_lmax = float(lipschitz_max) if lipschitz_max is not None else 0.0
        self.register_buffer("lipschitz_max_buf", torch.tensor(init_lmax))
        self.register_buffer("lipschitz_scale_buf", torch.tensor(init_lmax))
        self.register_buffer("lipschitz_scale_initialized", torch.tensor(lipschitz_max is not None))
        self.lipschitz_max = float(lipschitz_max) if lipschitz_max is not None else None
        self.lipschitz_scale = None

        self.easein = easein
        init_easein_scale = 1 / easein if easein > 0 else 1.0
        self.register_buffer("easein_scale", torch.tensor(float(init_easein_scale)))


        self.latent_dim = latent_dim
        self.n_flux_queries = n_flux_queries
        self.use_mean_features = use_mean_features
        self.call_count = 0

        self.n_pous = n_pous
        self.edge_pairs = edge_pairs
        self.n_edges = edge_pairs.shape[0]

        # various settings
        self.spectral_upper_bound = spectral_upper_bound
        self.no_flux_conditioning = no_flux_conditioning
        self.use_base_ops = use_base_ops
        self.id_init_base = id_init_base
        self.zero_init_ops = zero_init_ops
        self.lumped_L = lumped_L

        print("edge_pairs:", self.edge_pairs.shape)

        if self.use_mean_features:
            self.op_dim = proj_dim * 2  # difference + mean
        else:
            self.op_dim = proj_dim  # just difference

        self.proj = nn.Parameter(torch.randn(n_fields, proj_dim))
        self.proj_back = nn.Parameter(torch.randn(self.op_dim, n_fields))
        
        nn.init.orthogonal_(self.proj)
        nn.init.orthogonal_(self.proj_back)

        # Learnable queries for extracting flux-relevant information
        # These learn to attend to different aspects of the geometry/field
        self.flux_queries = nn.Parameter(torch.randn(n_flux_queries, latent_dim))
        std = 1 / math.sqrt(latent_dim)
        nn.init.normal_(self.flux_queries, std=std)
        
        # Cross-attention: queries attend to token sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=4,
            batch_first=True,
        )
            
        # After cross-attention, we have n_flux_queries features
        # Flatten and use as conditioning (more expressive than single mean-pooled vector)
        flux_context_dim = n_flux_queries * latent_dim

        self.A0 = nn.Linear(self.op_dim, self.op_dim, bias=False)
        self.B0 = nn.Linear(self.op_dim, self.op_dim, bias=False)
        self.C0 = nn.Linear(self.op_dim, self.op_dim, bias=False)
        # these can actually just be parameters, not linear layers
        # self.A0 = nn.Parameter(torch.eye(self.op_dim))
        # self.B0 = nn.Parameter(torch.eye(self.op_dim))
        # self.C0 = nn.Parameter(torch.eye(self.op_dim))
        if self.id_init_base:
            nn.init.eye_(self.A0.weight)
            nn.init.eye_(self.B0.weight)
            nn.init.eye_(self.C0.weight)

        # Cholesky parameterization for SPD L: L = (L_factor @ L_factor^T) + eps*I
        self.L_factor0 = nn.Parameter(torch.eye(self.n_edges))
        self.L_eps = 1e-6

        self.meta_A = self.create_meta_network(
            flux_context_dim,
            self.op_dim,
            hidden,
            n_layers,
            zero_last=self.zero_init_ops,
        )
        self.meta_B = self.create_meta_network(
            flux_context_dim,
            self.op_dim,
            hidden,
            n_layers,
            zero_last=self.zero_init_ops,
        )
        self.meta_C = self.create_meta_network(
            flux_context_dim,
            self.op_dim,
            hidden,
            n_layers,
            zero_last=self.zero_init_ops,
        )
        self.meta_L = self.create_meta_network(
            flux_context_dim,
            self.n_edges,  # number of coarse edges / POU pairs
            hidden,
            n_layers,
            zero_last=True,
        )

        self.m_max_net = nn.Sequential(
            nn.Linear(flux_context_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        # gamma is a single learnable scalar; initialized at zero (after sigmoid = 0.5)
        # actually initialize to -2.0 so initial sigmoid(gamma) = ~0.12 for more stability (could even init near zero (post sigmoid))
        self.gamma_raw = nn.Parameter(torch.ones(()) * gamma_init)
        self.beta_raw = nn.Parameter(torch.ones(()) * beta_init)
        
    def create_meta_network(self, input_dim, op_dim, hidden=128, n_layers=4, zero_last=False):
        meta_net = []
        for k in range(n_layers - 1):
            meta_net.append(nn.Linear(input_dim if k == 0 else hidden, hidden))
            meta_net.append(nn.GELU())
        last_layer = nn.Linear(hidden, op_dim * op_dim)
        if zero_last:
            nn.init.zeros_(last_layer.weight)
            if last_layer.bias is not None:
                nn.init.zeros_(last_layer.bias)
        meta_net.append(last_layer)
        return nn.Sequential(*meta_net)

    def extract_flux_context(self, z_latents):
        """
        Extract flux-specific context from shared latent tokens using cross-attention.
        """
        B = z_latents.shape[0]
        
        queries = self.flux_queries.unsqueeze(0).expand(B, -1, -1)

        flux_features, _ = self.cross_attn(
            query=queries,
            key=z_latents,
            value=z_latents,
        ) 
        flux_context = flux_features.reshape(B, -1)  # (B, n_flux_queries * latent_dim)
        return flux_context


    # if we want to regularize the flux Lipschitz constant, pass in u_hat from solver
    def get_flux_regularity_loss(self, z_latents, u_hat, edge_pairs):
        flux = self.forward(z_latents, u_hat, edge_pairs)
        d_flux_du = torch.autograd.grad(
            outputs=flux,
            inputs=u_hat,
            grad_outputs=torch.ones_like(flux),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return torch.mean(d_flux_du**2)

    def spectral_normalize(self, M):
        slim = 1e-6
        if self.spectral_upper_bound:
            slim = 1.0 # only enforce upper bound, don't scale up small values
        col_sum = M.abs().sum(dim=-2).max(dim=-1).values
        col_sum = col_sum.clamp(min=slim).view(-1, 1, 1)
        return M / col_sum
    
    def easein_scaling(self, gamma):
        if self.easein > 0:
            return gamma * self.easein_scale
        return gamma

    def update_easin_scale(self, current_epoch):
        if self.easein > 0:
            self.easein_scale.fill_(float(1 / max(1, self.easein - current_epoch)))

    def compute_normalizations(self, B):
        # NOTE: This must not mutate buffers/params because the nonlinear solver
        # uses torch.func transforms (jacrev/vmap).
        if self.lipschitz_max is None:
            beta_scale = torch.sigmoid(self.beta_raw)
            gamma_scale = torch.sigmoid(self.gamma_raw)
            return beta_scale, gamma_scale

        # Use buffer values as the source of truth in forward.
        lmax = self.lipschitz_max_buf
        lscale = self.lipschitz_scale_buf

        # If scale was never initialized (older checkpoints / first run), fall
        # back to using lmax as scale *without* mutating buffers.
        if not bool(self.lipschitz_scale_initialized.item()):
            lscale = lmax

        beta_scale = lscale * torch.sigmoid(self.beta_raw)
        gamma_scale = lscale * torch.sigmoid(self.gamma_raw)
        gamma_scale = self.easein_scaling(gamma_scale)

        # Apply caps (purely functional; do not update lscale here).
        beta_scale = torch.minimum(beta_scale, lmax)
        gamma_scale = torch.minimum(gamma_scale, lmax)

        return beta_scale, gamma_scale

    def compute_edge_features(self, u_tilde, edge_pairs):
        i_idx, j_idx = edge_pairs[:, 0], edge_pairs[:, 1]
        u_i = u_tilde[:, i_idx, :]
        u_j = u_tilde[:, j_idx, :]
        du = u_i - u_j
        mean_ij = 0.5 * (u_i + u_j)
        return du, mean_ij
    
    def construct_L(self, flux_context, B_dim):
        dL_eval = self.meta_L(flux_context).view(B_dim, self.n_edges, self.n_edges)
        if self.lumped_L:
            dL = torch.diag_embed(torch.diagonal(dL_eval, dim1=-2, dim2=-1))
        else:
            dL = dL_eval
        L_factor = dL + self.L_factor0.unsqueeze(0).to(dL.device)
        Lmat_spd = L_factor @ L_factor.transpose(-1, -2)
        Lmat_spd = Lmat_spd + self.L_eps * torch.eye(self.n_edges, device=Lmat_spd.device).unsqueeze(0)
        return Lmat_spd

    def construct_hyper_op(self, meta_network, base_op, B_dim, flux_context, op_dim=None):
        if op_dim is None:
            op_dim = self.op_dim
        # if self.no_flux_conditioning: # just return base operators
        #     return base_op.weight.unsqueeze(0).to(flux_context.device).expand(B_dim, -1, -1)
        op_res = (meta_network(flux_context)).view(B_dim, op_dim, op_dim)
        if self.use_base_ops:
            op_mat = op_res + base_op.weight.unsqueeze(0).to(op_res.device)
        else:
            op_mat = op_res
        op_mat = self.spectral_normalize(op_mat)
        return op_mat
    
    def construct_hyper_network(self, A, B, C, gamma, beta):
        def hyper_net(u_in):
            # F_ij = alpha A (ui-uj) + beta C tanh(B (ui-uj))

            # linear trunk: m * A @ du (linear layer)
            A_du = torch.einsum("bij,bej->bei", A, u_in)
            F_lin = beta * A_du

            # nonlinear trunk: gamma * (C @ du + tanh(B @ du))
            B_du = torch.einsum("bij,bej->bei", B, u_in)
            tanh_B_du = torch.tanh(B_du)
            C_x = torch.einsum("bij,bej->bei", C, tanh_B_du)
            F_nonlin = gamma * C_x
            return F_lin + F_nonlin
        return hyper_net



    def forward(self, z_latents, u_hat, edge_pairs):
        B_dim, P, F = u_hat.shape
        
        # Extract flux-specific context via learned cross-attention
        flux_context = self.extract_flux_context(z_latents)

        beta_scale, gamma_scale = self.compute_normalizations(B_dim)

        # Determine operator dimension based on whether we use mean features
        op_dim = self.proj_dim * 2 if self.use_mean_features else self.proj_dim

        # Build operators A(z), B(z), C(z)
        Amat = self.construct_hyper_op(self.meta_A, self.A0, B_dim, flux_context, op_dim)
        Bmat = self.construct_hyper_op(self.meta_B, self.B0, B_dim, flux_context, op_dim)
        Cmat = self.construct_hyper_op(self.meta_C, self.C0, B_dim, flux_context, op_dim)

        # Build SPD L matrix
        Lmat_spd = self.construct_L(flux_context, B_dim)
        Lmat_spd = self.spectral_normalize(Lmat_spd)

        proj_op = self.spectral_normalize(self.proj).squeeze(0)
        u_tilde = torch.einsum("kf,bpk->bpf", proj_op, u_hat)
    
        # compute u difference and mean
        edge_feat, mean_feat = self.compute_edge_features(u_tilde, edge_pairs)
        in_feat = edge_feat
        if self.use_mean_features:
            in_feat = torch.cat([edge_feat, mean_feat], dim=-1)
        
        hyper_net = self.construct_hyper_network(Amat, Bmat, Cmat, gamma_scale, beta_scale)

        # per-edge fluxes in op space
        flux_op = hyper_net(in_feat)

        # mass like scaling with SPD L matrix
        flux_op = torch.einsum("bij,bjd->bid", Lmat_spd, flux_op)

        # map back to physical fields
        proj_back_op = self.spectral_normalize(self.proj_back).squeeze(0)
        flux = torch.einsum("kf,bpk->bpf", proj_back_op, flux_op)

        return flux


# can be any model that produces per-field W
# for now just a linear function of query features
# in general this should be a cross attention transformer
class LinearLearnableW(nn.Module):
    def __init__(
        self,
        n_pou,
        n_fields,
        in_dim,
        model_dim=128,
    ):
        super().__init__()
        self.n_pou = n_pou
        self.n_fields = n_fields
        self.final_projection = nn.Linear(model_dim, n_pou * n_fields)

    def forward(self, query_feats, z_latents):
        logits = self.final_projection(query_feats)
        W = logits.view(logits.shape[0], logits.shape[1], self.n_pou, self.n_fields)
        W = W.permute(0, 2, 1, 3).contiguous()
        return W


class MLPLearnableW(nn.Module):
    """
    MLP-based W predictor with local node features + global latent conditioning.
    
    Each node gets:
    - Local features (query_feats per node)
    - Global context (pooled z_latents broadcast to all nodes)
    
    Weight-shared MLP processes all nodes in parallel.
    """
    def __init__(
        self,
        n_pou,
        n_fields,
        in_dim,
        model_dim=128,
        hidden_dim=None,
        n_queries=1,
        alpha_init=3.0,  # initial value for gated skip connection
        dropout=0.0,
    ):
        super().__init__()
        self.n_pou = n_pou
        self.n_fields = n_fields
        self.n_queries = n_queries
        hidden_dim = hidden_dim or 4*model_dim
        context_dim = n_queries * model_dim
        
        # Learnable queries for W-specific information extraction
        self.queries = nn.Parameter(torch.randn(n_queries, model_dim))
        nn.init.normal_(self.queries, std=0.02)
        
        # Cross-attention: queries attend to token sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=4,
            batch_first=True,
            dropout=dropout,
        )

        # Gated skip connection with learnable interpolation
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.linear_skip = nn.Linear(model_dim + model_dim, n_pou * n_fields)
        
        # Weight-shared MLP: processes (local + global) per node
        # self.mlp = nn.Sequential(
        #     nn.Linear(model_dim + model_dim, hidden_dim),  # local + global
        #     nn.Dropout(dropout),
        #     nn.LayerNorm(hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Dropout(dropout),
        #     nn.LayerNorm(hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, n_pou * n_fields)
        # )
        self.mlp = nn.Sequential(
            nn.Linear(model_dim + model_dim, hidden_dim),  # local + global
            nn.Dropout(dropout),
            # nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            # nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, n_pou * n_fields)
        )

    def cross_attention_pooling(self, z_latents):
        """
        Extract W-specific context from shared latent tokens using cross-attention.
        """
        B = z_latents.shape[0]
        
        # Expand learnable queries for batch
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention: queries attend to latent tokens
        W_features, _ = self.cross_attn(
            query=queries,
            key=z_latents,
            value=z_latents,
        )
        
        return W_features.mean(dim=1)
    
    def forward(self, query_feats, z_latents=None):
        # query_feats: (B, N, model_dim) - local per-node features
        # z_latents: (B, n_latent, model_dim) - latent representation (optional)

        if z_latents is None:
            z_latents = query_feats

        # Extract W-specific context via cross-attention
        z_global = self.cross_attention_pooling(z_latents)
        
        # Broadcast global context to all nodes
        z_broadcast = z_global.unsqueeze(1).expand(-1, query_feats.size(1), -1)
        
        # Concatenate local + global features
        combined = torch.cat([query_feats, z_broadcast], dim=-1)
        
        # Weight-shared MLP
        logits = self.mlp(combined)

        # Linear skip with gated interpolation
        linear_skip_out = self.linear_skip(combined)
        logits = logits * torch.sigmoid(self.alpha) + linear_skip_out * (1 - torch.sigmoid(self.alpha))
        
        # Reshape to (B, n_pou, N, n_fields)
        W = logits.view(logits.shape[0], logits.shape[1], self.n_pou, self.n_fields)
        W = W.permute(0, 2, 1, 3).contiguous()
        return W


class MLPSourceModel(nn.Module):
    """
    MLP-based source term predictor with local + global conditioning.
    
    Same architecture as MLPLearnableW but outputs n_fields per node.
    """
    def __init__(self, model_dim=128, n_fields=3, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or model_dim
        
        # Global latent pooling and projection
        self.global_pool = nn.Linear(model_dim, model_dim)
        
        # Weight-shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(model_dim + model_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_fields)
        )
    
    def forward(self, query_feats, z_latents=None):
        # query_feats: (B, N, model_dim)
        # z_latents: (B, n_latent, model_dim)
        if z_latents is None:
            z_latents = query_feats
        
        # Global pooling
        z_global = z_latents.mean(dim=1, keepdim=True)  # (B, 1, model_dim)
        z_global = self.global_pool(z_global)
        
        # Broadcast to all nodes
        z_broadcast = z_global.expand(-1, query_feats.size(1), -1)
        
        # Concatenate and process
        combined = torch.cat([query_feats, z_broadcast], dim=-1)
        output = self.mlp(combined)  # (B, N, n_fields)
        
        return output


# wrapper for any model that produces per-field W
class BoundaryWPerField(nn.Module):

    def __init__(self, learnable_w_per_field, temperature=1.0):
        super().__init__()
        self.learnable = learnable_w_per_field
        self.n_pou = learnable_w_per_field.n_pou
        self.n_fields = learnable_w_per_field.n_fields
        self.temperature = temperature

    def per_field_boundary_outer(self, W_inner, dirichlet_nodes, boundary_vals):
        B, P, N, F = W_inner.shape
        W_masked = []
        for f in range(F):
            Wf = W_inner[..., f]
            dirichlet_mask_f = dirichlet_nodes[..., f]
            bvf = boundary_vals[..., f]
            Wf = utils.make_masked_outer_W(Wf, dirichlet_mask_f, bvf)
            W_masked.append(Wf.unsqueeze(-1))
        W_out = torch.cat(W_masked, dim=-1)
        return W_out

    def forward(self, query_feats, z, dirichlet_nodes, boundary_vals):
        logits = self.learnable(query_feats, z)
        softmax_W = torch.softmax(logits / max(self.temperature, 1e-6), dim=1)
        W = self.per_field_boundary_outer(softmax_W, dirichlet_nodes, boundary_vals)
        return W

class SourceWrapper(nn.Module):
    def __init__(self, source_model, source_mode="zero"):
        super().__init__()
        self.source_model = source_model

    def forward(self, *args, **kwargs):
        return self.source_model(*args, **kwargs)



class CNWF_multifieldPOU(nn.Module):

    # TODO take an encoder as an argument, and take flux_args as a dict
    def __init__(
        self,
        # all
        n_pou=16,
        n_fields=3,
        model_dim=64,
        diffusion_only=False,
        device="cpu",
        encoder_in_dim=None,
        decoder_in_dim=None,
        # encoder settings (or some for all)?
        encoder_mode="upt",  # "upt" or "transolver"
        n_anchors=512,
        n_latent=256,
        n_heads=2,
        n_layers=4,
        encoder_dropout=0.01,
        resample_anchors=True,
        # begin flux args
        flux_mode="adaptive",  # "adaptive" or "fixed" NO LONGER USED
        flux_weight=1.0,
        flux_dim=8,
        flux_layers=2,
        flux_lipschitz_max=0.2,
        flux_gamma_init=-2.0,
        flux_mean_features=True,
        use_base_ops=True,
        id_init_base=True,
        zero_init_ops=True,
        spectral_upper_bound=False,
        no_flux_conditioning=False,
        flux_easein=0,
        # begin w args
        w_dropout=0.01,
        w_temperature=1.0,
        w_alpha_init=3.0,
        # solve args
        max_iters=50,
        tol=1e-5,
        solve_exit_ratio=0.8,
        use_line_search=False,
        # source settings
        source_mode="zero",  # "zero", "zero_mean", "constant", "unconstrained"
        *args,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.n_pou = n_pou
        self.n_fields = n_fields
        self.n_anchors = n_anchors
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.source_mode = source_mode
        self.encoder_mode = encoder_mode
        self.diffusion_only = diffusion_only

        self.n_bc_pou = 3
        self.full_n_pou = n_pou + self.n_bc_pou
        self.n_one_forms = self.full_n_pou * (self.full_n_pou - 1) // 2

        self.setup_templates()

        self.flux_weight = flux_weight
        self.decoder_in_dim = (
            decoder_in_dim if decoder_in_dim is not None else encoder_in_dim
        )

        # TODO I would prefer to use this
        # self.encoder = setup_encoder(**encoder_params)


        # if encoder_mode == "transolver":
        #     self.encoder = TransolverEncoder(
        #         in_dim=encoder_in_dim,
        #         out_dim=model_dim,
        #         d_model=model_dim,
        #         nhead=n_heads,
        #         n_slices=64,
        #         num_layers=n_layers,
        #         dropout=encoder_dropout,
        #         ffn_mult=4,
        #         return_anchors=True,
        #         use_output_residual=False,
        #     )
        # Build geometry encoders via the shared factory (per-file encoders).
        enc_cfg = {
            "encoder_mode": encoder_mode,
            "model_dim": model_dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "encoder_dropout": encoder_dropout,
            "n_anchors": n_anchors,
            "resample_anchors": resample_anchors,
            "n_slices": 64,
        }

        self.encoder = build_geometry_encoder(
            enc_cfg,
            in_dim=int(encoder_in_dim),
            out_dim=int(model_dim),
            return_anchors=True,
        )

        self.w_encoder = build_geometry_encoder(
            enc_cfg,
            in_dim=int(encoder_in_dim),
            out_dim=int(model_dim),
            return_anchors=True,
            n_layers=int(n_layers),
        )

        self.source_model = MLPSourceModel(model_dim=model_dim, n_fields=n_fields)

        self.w_inner = MLPLearnableW(
            n_pou=n_pou,
            n_fields=n_fields,
            in_dim=model_dim,
            model_dim=model_dim,
            alpha_init=w_alpha_init,
            dropout=w_dropout,
        )
        self.boundary_w = BoundaryWPerField(self.w_inner, temperature=w_temperature)

        self.flux_model = StableHyperFlux(
            n_fields=n_fields,
            latent_dim=model_dim,
            hidden=flux_dim,
            n_layers=flux_layers,
            lipschitz_max=flux_lipschitz_max,
            gamma_init=flux_gamma_init,
            use_mean_features=flux_mean_features,
            easein=flux_easein,
            n_pous=self.full_n_pou,
            edge_pairs=self.edge_pairs,
            use_base_ops=use_base_ops,
            id_init_base=id_init_base,
            zero_init_ops=zero_init_ops,
            spectral_upper_bound=spectral_upper_bound,
            no_flux_conditioning=no_flux_conditioning,
        )


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

    def G_residual(
        self,
        u_hat_flat,
        flux_model,
        z_latents,
        f_latent,
        K_hat,
        M_hat,
        delta_0,
        latent_boundary_masks,
        latent_boundary_vals,
    ):
        B = u_hat_flat.shape[0]
        u_hat = u_hat_flat.view(B, self.full_n_pou, self.n_fields)


        diff_term = torch.einsum("bijf,bjf->bif", K_hat, u_hat)
        src_term = torch.einsum("bijf,bjf->bif", M_hat, f_latent)

        if self.diffusion_only:
            r = diff_term - src_term
        else:
            flux = flux_model(z_latents, u_hat, self.edge_pairs)
            flux_term = torch.einsum("bji,bjf->bif", delta_0, flux)

            r = diff_term - src_term - self.flux_weight * flux_term

        r = utils.apply_dirichlet_residual(
            r, u_hat, latent_boundary_masks, latent_boundary_vals
        )
        return r.reshape(B, -1).contiguous()

    def project_and_bc(self, K_list, M_list, W, bilinear_mask, latent_scaling=1.0):
        K_hat = utils.project_bilinear_per_field(K_list, W, latent_scaling)
        M_hat = utils.project_bilinear_per_field(M_list, W, latent_scaling)
        K_hat_bc = utils.apply_dirichlet_per_field(K_hat, bilinear_mask)
        M_hat_bc = utils.apply_dirichlet_per_field(M_hat, bilinear_mask)
        return K_hat_bc, M_hat_bc
    
    def update_lipschitz_limits(
        self, in_tokens, query_tokens, adj, K_list, M_list, dirichlet_nodes, boundary_vals, margin=0.1,
    ):
        """
        Update Lipschitz limit using exponential moving average for stability.
        """
        B, N = in_tokens.shape[:2]
        delta_0, bilinear_mask, latent_mask, latent_boundary_vals = self.expand_masks(B)
        latent_scaling = N / self.full_n_pou

        
        z_unpooled = self.encoder(in_tokens)[0]

        enc_tokens = self.w_encoder(in_tokens)[0]
        W = self.boundary_w(enc_tokens, z_unpooled, dirichlet_nodes, boundary_vals)
        # W = self.boundary_w(z_unpooled, z_unpooled, dirichlet_nodes, boundary_vals)

        K_hat_bc, M_hat_bc = self.project_and_bc(
            K_list, M_list, W, bilinear_mask, latent_scaling
        )

        interior_mask = ~bilinear_mask[0]
        n_interior = interior_mask.sum().item()
        K_interior = K_hat_bc[0, interior_mask, :, 0][:, interior_mask]  # (n_interior, n_interior)
        
        try:
            lambda_min = torch.lobpcg(K_interior, k=1)[0].item()
            norm_delta0 = torch.linalg.svdvals(delta_0[0]).max().item()
            lipschitz_limit = margin * lambda_min / norm_delta0

            if not bool(self._lipschitz_ema_initialized.item()):
                self._lipschitz_ema.fill_(float(lipschitz_limit))
                self._lipschitz_ema_initialized.fill_(True)
            else:
                alpha = float(self._lipschitz_ema_alpha)
                new_val = alpha * float(lipschitz_limit) + (1 - alpha) * float(self._lipschitz_ema.item())
                self._lipschitz_ema.fill_(float(new_val))
            
            # Use smoothed value
            new_lmax = float(self._lipschitz_ema.item())
            # Keep backwards-compatible attribute and persisted buffer in sync.
            self.flux_model.lipschitz_max = new_lmax
            with torch.no_grad():
                if hasattr(self.flux_model, "lipschitz_max_buf"):
                    self.flux_model.lipschitz_max_buf.fill_(new_lmax)
            
            print(f"  Lipschitz - Current: {lipschitz_limit:.6f}, EMA: {float(self._lipschitz_ema.item()):.6f}")
            print(f"    λ_min: {lambda_min:.6f}, ||δ₀||: {norm_delta0:.6f}")
            
        except Exception as e:
            print(f"  Lipschitz limit computation failed: {e}")
            # Keep current value (no update)

    def forward(
        self, in_tokens, query_tokens, adj, K_list, M_list, dirichlet_nodes, boundary_vals
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
        converged = solve_out["converged"]
        r_mag = solve_out["r_mag"]
        n_iters = solve_out["n_iters"]

        u_hat_coarse = u_hat.view(B, self.full_n_pou, self.n_fields)
        u_fine = torch.einsum("bjif,bjf->bif", W, u_hat_coarse)

        with torch.no_grad():
            residual = self.G_residual(
                u_hat,
                self.flux_model,
                z_unpooled,
                f_latent,
                K_hat_bc,
                M_hat_bc,
                delta_0,
                latent_mask,
                latent_boundary_vals,
            )
        return {
            "u_fine": u_fine,
            "u_coarse": u_hat_coarse,
            "residual_flat": residual,
            "W": W,
            "converged": converged,
            "n_iters": n_iters,
        }

    # if we want a minimal forward interface which just returns the solution on original mesh
    def simple_forward(self, *args, **kwargs):
        return self.forward(*args, **kwargs)["u_fine"]


if __name__ == "__main__":
    pass
