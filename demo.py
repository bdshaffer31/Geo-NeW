
import torch
import src.utils as utils


def run_epoch(
    geo_new,
    loader,
    optimizer,
    train,
):
    """Run one epoch and return metrics."""
    if train:
        geo_new.train()
    else:
        geo_new.eval()

    total_rel = 0.0
    total_conv = 0.0
    n_batches = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            tokens = batch["tokens"]
            K_list = batch["K"]
            M_list = batch["M"]
            dirichlet_nodes = batch["dirichlet_nodes"]
            u_true = batch["fields"]
            n_original_nodes = batch["n_original_nodes"]

            boundary_vals = u_true * dirichlet_nodes.to(u_true.dtype)

            if train:
                optimizer.zero_grad(set_to_none=True)

            out = geo_new(
                in_tokens=tokens,
                K_list=K_list,
                M_list=M_list,
                dirichlet_nodes=dirichlet_nodes,
                boundary_vals=boundary_vals,
            )
            u_pred = out["u_fine"]

            loss = utils.relative_l2_error_unpadded(u_true, u_pred, n_original_nodes)
            if train:
                loss.backward()
                optimizer.step()

            conv = out["converged"].float().mean()

            total_rel += float(loss.detach().cpu())
            total_conv += float(conv.detach().cpu())
            n_batches += 1

    denom = max(1, n_batches)
    return total_rel / denom, total_conv / denom


def main():
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load in pre-processed data and create dataloaders
    data_file = "data/processed_polypoisson_id.pt"
    train_loader, val_loader = utils.create_dataloaders(
        train_file=data_file,
        train_range=(0, 400),
        val_range=(400, 500),
        batch_size=8,
        device=str(device),
        hks_steps=8,
        use_coords=False,
        use_poisson=False,
        use_harmonic=True,
    )

    encoder_in_dim = next(iter(train_loader))["tokens"].shape[-1]
    geo_new = utils.setup_GeoNew_model(encoder_in_dim=encoder_in_dim).to(device)

    print("Model parameters:", geo_new.get_full_parameter_count())
    print("Train samples:", len(train_loader.dataset), "| Val samples:", len(val_loader.dataset))

    optimizer = torch.optim.Adam(geo_new.parameters(), lr=1e-3)

    # train for 500 epochs, should be sub 1% error on train and validation by 50 
    n_epochs = 500
    for epoch in range(1, n_epochs + 1):
        train_rel, train_conv = run_epoch(
            geo_new=geo_new,
            loader=train_loader,
            optimizer=optimizer,
            train=True,
        )
        val_rel, val_conv = run_epoch(
            geo_new=geo_new,
            loader=val_loader,
            optimizer=optimizer,
            train=False,
        )

        print(
            f"Epoch {epoch:04d} | "
            f"train rel={train_rel:.4e} conv={train_conv:.3f} | "
            f"val rel={val_rel:.4e} conv={val_conv:.3f}"
        )

if __name__ == "__main__":
    main()