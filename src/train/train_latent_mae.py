"""
阶段 2.3: 训练隐空间特征提取器 (Latent-MAE φ).

自监督预训练：在 VAE latent cache 上做掩码自编码 (MAE)。
训练完成后，仅保留 Encoder 作为 φ 并冻结参数。

Usage:
    python src/train/train_latent_mae.py configs/qm9_latent_mae.yaml
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.models.latent_mae import LatentMAE, vicreg_variance_loss, vicreg_covariance_loss, multi_prop_supcon_loss
from src.utils import build_lr_scheduler

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required.") from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load latent cache ──
    cache = torch.load(cfg["latent_cache_path"], map_location="cpu", weights_only=False)
    z_train = cache["train"].to(device)
    z_val = cache["val"].to(device)
    latent_dim = int(cache["latent_dim"])
    print(f"[data] z_train: {z_train.shape}, z_val: {z_val.shape}, latent_dim={latent_dim}")

    # ── Load properties (optional) ──
    num_properties = int(cfg.get("num_properties", 0))
    lambda_prop = float(cfg.get("lambda_prop", 0.0))
    lambda_recon = float(cfg.get("lambda_recon", 1.0))  # NEW: weight for reconstruction loss (0 to disable)
    prop_grad_through_encoder = bool(cfg.get("prop_grad_through_encoder", False))
    prop_indices = cfg.get("prop_indices", None)  # e.g. [0, 2] for QED and LogP only

    # ── Contrastive config ──
    contrastive_mode = str(cfg.get("contrastive_mode", "none"))
    contrastive_property_idx = int(cfg.get("contrastive_property_idx", 0))
    contrastive_temperature = float(cfg.get("contrastive_temperature", 0.1))
    contrastive_sigma = float(cfg.get("contrastive_sigma", 1.0))
    lambda_contrastive = float(cfg.get("lambda_contrastive", 1.0))
    contrastive_warmup_epochs = int(cfg.get("contrastive_warmup_epochs", 0))

    # ── VICReg config ──
    lambda_vicreg_var = float(cfg.get("lambda_vicreg_var", 0.0))
    lambda_vicreg_cov = float(cfg.get("lambda_vicreg_cov", 0.0))
    vicreg_gamma = float(cfg.get("vicreg_gamma", 1.0))  # target std per dimension

    # ── Multi-property contrastive config ──
    multi_prop_contrastive = bool(cfg.get("multi_prop_contrastive", False))
    multi_prop_sigma = float(cfg.get("multi_prop_sigma", 1.0))

    props_train = None
    props_val = None

    if num_properties > 0 and "train_props" in cache:
        raw_train_props = cache["train_props"]  # [N_train, K]
        raw_val_props = cache["val_props"]      # [N_val, K]
        prop_names = cache.get("prop_names", [f"prop_{i}" for i in range(raw_train_props.shape[1])])

        # Select specific property indices if specified
        if prop_indices is not None:
            raw_train_props = raw_train_props[:, prop_indices]
            raw_val_props = raw_val_props[:, prop_indices]
            prop_names = [prop_names[i] for i in prop_indices]
            num_properties = len(prop_indices)

        # Handle NaN: replace with column mean
        for col in range(raw_train_props.shape[1]):
            col_data = raw_train_props[:, col]
            valid = ~torch.isnan(col_data)
            if valid.any():
                col_mean = col_data[valid].mean()
                raw_train_props[~valid, col] = col_mean
                # Use train mean for val NaN replacement
                val_col = raw_val_props[:, col]
                val_nan = torch.isnan(val_col)
                raw_val_props[val_nan, col] = col_mean

        # Normalize properties to zero-mean unit-variance
        prop_mean = raw_train_props.mean(dim=0, keepdim=True)
        prop_std = raw_train_props.std(dim=0, keepdim=True).clamp(min=1e-6)
        props_train = ((raw_train_props - prop_mean) / prop_std).to(device)
        props_val = ((raw_val_props - prop_mean) / prop_std).to(device)

        print(f"[props] Using {num_properties} properties: {prop_names}")
        print(f"[props] lambda_prop={lambda_prop}, grad_through_encoder={prop_grad_through_encoder}")
        for i, name in enumerate(prop_names):
            print(f"  {name}: mean={prop_mean[0, i]:.4f}, std={prop_std[0, i]:.4f}")
    elif num_properties > 0:
        print(f"[WARN] num_properties={num_properties} but no properties in cache. Disabling property head.")
        num_properties = 0

    # Contrastive mode requires properties in cache even if num_properties=0
    if contrastive_mode != "none" and props_train is None:
        if "train_props" in cache:
            raw_train_props = cache["train_props"]
            raw_val_props = cache["val_props"]
            prop_names = cache.get("prop_names", [f"prop_{i}" for i in range(raw_train_props.shape[1])])
            # Handle NaN
            for col in range(raw_train_props.shape[1]):
                col_data = raw_train_props[:, col]
                valid = ~torch.isnan(col_data)
                if valid.any():
                    col_mean = col_data[valid].mean()
                    raw_train_props[~valid, col] = col_mean
                    val_col = raw_val_props[:, col]
                    raw_val_props[torch.isnan(val_col), col] = col_mean
            prop_mean = raw_train_props.mean(dim=0, keepdim=True)
            prop_std = raw_train_props.std(dim=0, keepdim=True).clamp(min=1e-6)
            props_train = ((raw_train_props - prop_mean) / prop_std).to(device)
            props_val = ((raw_val_props - prop_mean) / prop_std).to(device)
            print(f"[contrastive] Loaded properties for contrastive learning")
        else:
            print(f"[WARN] contrastive_mode={contrastive_mode} but no properties in cache. Disabling.")
            contrastive_mode = "none"

    if contrastive_mode != "none":
        c_prop_name = prop_names[contrastive_property_idx] if prop_names else f"prop_{contrastive_property_idx}"
        print(f"[contrastive] mode={contrastive_mode}, property={c_prop_name} (idx={contrastive_property_idx})")
        print(f"[contrastive] temperature={contrastive_temperature}, sigma={contrastive_sigma}, lambda={lambda_contrastive}")
        print(f"[contrastive] warmup_epochs={contrastive_warmup_epochs}")

    # ── z-space statistics (for normalization) ──
    z_mean = z_train.mean(dim=0, keepdim=True)
    z_std = z_train.std(dim=0, keepdim=True).clamp(min=1e-6)
    # Normalize z to zero-mean unit-variance per dimension
    z_train_n = (z_train - z_mean) / z_std
    z_val_n = (z_val - z_mean) / z_std
    print(f"[z stats] mean={z_train.mean():.3f}, std={z_train.std():.3f}")
    print(f"[z stats] normalized mean={z_train_n.mean():.3f}, std={z_train_n.std():.3f}")

    # ── Multi-scale config ──
    multi_scale_dims = cfg.get("multi_scale_dims", None)  # e.g. [8, 16, 32]
    if multi_scale_dims is not None:
        print(f"[multi-scale] dims={multi_scale_dims}")

    # ── Model ──
    model = LatentMAE(
        latent_dim=latent_dim,
        phi_dim=cfg.get("phi_dim", 128),
        hidden_dim=cfg.get("hidden_dim", 256),
        num_encoder_layers=cfg.get("num_encoder_layers", 4),
        num_decoder_layers=cfg.get("num_decoder_layers", 2),
        mask_ratio=cfg.get("mask_ratio", 0.5),
        num_properties=num_properties,
        dropout=cfg.get("dropout", 0.1),
        normalize_output=cfg.get("normalize_output", True),
        prop_grad_through_encoder=prop_grad_through_encoder,
        contrastive_mode=contrastive_mode,
        contrastive_property_idx=contrastive_property_idx,
        contrastive_temperature=contrastive_temperature,
        contrastive_sigma=contrastive_sigma,
        multi_scale_dims=multi_scale_dims,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] LatentMAE params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(cfg.get("epochs", 100))
    batch_size = int(cfg.get("batch_size", 1024))
    best_val = float("inf")

    # ── LR Scheduler ──
    warmup_epochs = int(cfg.get("warmup_epochs", 5))
    lr_schedule = str(cfg.get("lr_schedule", "cosine"))
    min_lr_ratio = float(cfg.get("min_lr_ratio", 0.01))
    scheduler = build_lr_scheduler(
        optimizer, total_epochs=epochs,
        warmup_epochs=warmup_epochs, schedule=lr_schedule,
        min_lr_ratio=min_lr_ratio,
    )
    print(f"[lr] schedule={lr_schedule}, warmup={warmup_epochs}, epochs={epochs}")

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        # Contrastive warmup: linearly ramp lambda_contrastive
        if contrastive_warmup_epochs > 0 and epoch <= contrastive_warmup_epochs:
            contra_weight = lambda_contrastive * (epoch / contrastive_warmup_epochs)
        else:
            contra_weight = lambda_contrastive

        model.train()
        perm = torch.randperm(z_train_n.shape[0], device=device)
        running_loss = 0.0
        running_recon = 0.0
        running_prop = 0.0
        running_contra = 0.0
        n_batches = 0

        for i in range(0, z_train_n.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            z_batch = z_train_n[idx]
            prop_batch = props_train[idx] if props_train is not None else None

            out = model(z_batch, properties=prop_batch)
            recon_loss = out["recon_loss"]
            if lambda_recon > 0 and not torch.isnan(recon_loss):
                loss = lambda_recon * recon_loss
            else:
                loss = torch.tensor(0.0, device=device)

            # Add property prediction loss
            if "prop_loss" in out and lambda_prop > 0:
                loss = loss + lambda_prop * out["prop_loss"]

            # Contrastive + VICReg: multi-scale or single-scale
            if "phis" in out and (contra_weight > 0 or lambda_vicreg_var > 0 or lambda_vicreg_cov > 0):
                # Per-scale losses for each multi-scale head
                for phi_s in out["phis"]:
                    if contra_weight > 0 and prop_batch is not None and multi_prop_contrastive:
                        loss_mc = multi_prop_supcon_loss(
                            phi_s, prop_batch,
                            temperature=contrastive_temperature,
                            sigma=multi_prop_sigma,
                        )
                        loss = loss + contra_weight * loss_mc
                    if lambda_vicreg_var > 0:
                        loss = loss + lambda_vicreg_var * vicreg_variance_loss(phi_s, gamma=vicreg_gamma)
                    if lambda_vicreg_cov > 0:
                        loss = loss + lambda_vicreg_cov * vicreg_covariance_loss(phi_s)
            else:
                # Single-scale path (backward compat)
                if "contrastive_loss" in out and contra_weight > 0:
                    if multi_prop_contrastive and prop_batch is not None:
                        loss_mc = multi_prop_supcon_loss(
                            out["phi"], prop_batch,
                            temperature=contrastive_temperature,
                            sigma=multi_prop_sigma,
                        )
                        loss = loss + contra_weight * loss_mc
                    else:
                        loss = loss + contra_weight * out["contrastive_loss"]
                phi_for_vicreg = out["phi"]
                if lambda_vicreg_var > 0:
                    loss = loss + lambda_vicreg_var * vicreg_variance_loss(phi_for_vicreg, gamma=vicreg_gamma)
                if lambda_vicreg_cov > 0:
                    loss = loss + lambda_vicreg_cov * vicreg_covariance_loss(phi_for_vicreg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_recon += recon_loss.item() if not torch.isnan(recon_loss) else 0.0
            if "prop_loss" in out:
                running_prop += out["prop_loss"].item()
            if "contrastive_loss" in out:
                running_contra += out["contrastive_loss"].item()
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        train_recon = running_recon / max(1, n_batches)
        train_prop = running_prop / max(1, n_batches)
        train_contra = running_contra / max(1, n_batches)

        # Validation
        model.eval()
        with torch.no_grad():
            # Process val in batches to avoid OOM
            val_losses = []
            val_recon_losses = []
            val_prop_losses = []
            val_contra_losses = []
            for i in range(0, z_val_n.shape[0], batch_size):
                z_batch = z_val_n[i:i + batch_size]
                prop_batch = props_val[i:i + batch_size] if props_val is not None else None
                out = model(z_batch, properties=prop_batch)
                val_recon = out["recon_loss"].item()
                if not (val_recon != val_recon):  # NaN check
                    val_recon_losses.append(val_recon)
                    total = lambda_recon * val_recon
                else:
                    val_recon_losses.append(0.0)
                    total = 0.0
                if "prop_loss" in out and lambda_prop > 0:
                    val_p = out["prop_loss"].item()
                    val_prop_losses.append(val_p)
                    total += lambda_prop * val_p
                if "contrastive_loss" in out and lambda_contrastive > 0:
                    val_c = out["contrastive_loss"].item()
                    val_contra_losses.append(val_c)
                    total += lambda_contrastive * val_c
                val_losses.append(total)
            val_loss = sum(val_losses) / len(val_losses)
            val_recon_avg = sum(val_recon_losses) / len(val_recon_losses)
            val_prop_avg = sum(val_prop_losses) / len(val_prop_losses) if val_prop_losses else 0.0
            val_contra_avg = sum(val_contra_losses) / len(val_contra_losses) if val_contra_losses else 0.0

        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            improved = "  ★ best"
            # Save checkpoint
            save_data = {
                "model_state": model.state_dict(),
                "cfg": cfg,
                "latent_dim": latent_dim,
                "z_mean": z_mean.cpu(),
                "z_std": z_std.cpu(),
                "best_val_loss": best_val,
                "contrastive_mode": contrastive_mode,
            }
            # Save property normalization stats for downstream use
            if props_train is not None:
                save_data["prop_mean"] = prop_mean.cpu()
                save_data["prop_std"] = prop_std.cpu()
                save_data["prop_names"] = prop_names
                save_data["num_properties"] = num_properties
            torch.save(save_data, out_dir / "best_latent_mae.pt")

        if epoch % 10 == 0 or epoch <= 5 or improved:
            log = f"Epoch {epoch:3d}/{epochs}  train={train_loss:.4f} (recon={train_recon:.4f}"
            if props_train is not None and lambda_prop > 0:
                log += f" prop={train_prop:.4f}"
            if contrastive_mode != "none":
                log += f" contra={train_contra:.4f}"
            log += f")  val={val_loss:.4f} (recon={val_recon_avg:.4f}"
            if val_prop_losses:
                log += f" prop={val_prop_avg:.4f}"
            if val_contra_losses:
                log += f" contra={val_contra_avg:.4f}"
            log += f"){improved}"
            print(log)

        scheduler.step()

    # ── Final: check feature quality ──
    model.eval()
    with torch.no_grad():
        import numpy as np
        from sklearn.decomposition import PCA
        # Compute φ for some samples and check diversity
        phi_train = model.extract_features(z_train_n[:2000])
        cos_sim = phi_train @ phi_train.T
        cos_sim.fill_diagonal_(0)
        print(f"\n[φ quality] mean cosine sim: {cos_sim.mean():.4f} (lower = more discriminative)")
        print(f"[φ quality] max cosine sim: {cos_sim.max():.4f}")
        print(f"[φ quality] φ dim={phi_train.shape[-1]}, norms mean={phi_train.norm(dim=-1).mean():.3f}")

        # PCA participation ratio
        phi_np = phi_train.cpu().numpy()
        pca = PCA(n_components=min(phi_np.shape[1], phi_np.shape[0]))
        pca.fit(phi_np)
        ev = pca.explained_variance_ratio_
        pr = 1.0 / (ev ** 2).sum()
        cum = np.cumsum(ev)
        d90 = np.searchsorted(cum, 0.9) + 1
        d95 = np.searchsorted(cum, 0.95) + 1
        print(f"[φ quality] PR={pr:.1f}, d90={d90}, d95={d95}, top1={ev[0]*100:.1f}%")

        # Linear probe R² for properties
        if props_train is not None:
            from sklearn.linear_model import Ridge
            phi_all = model.extract_features(z_train_n[:5000]).cpu().numpy()
            p_all = props_train[:5000].cpu().numpy()
            for i in range(p_all.shape[1]):
                ridge = Ridge(alpha=1.0)
                ridge.fit(phi_all[:4000], p_all[:4000, i])
                r2 = ridge.score(phi_all[4000:], p_all[4000:, i])
                pn = prop_names[i] if i < len(prop_names) else f"prop_{i}"
                print(f"  R²({pn}) = {r2:.3f}")

        # If contrastive mode was used, check pairwise ordering quality
        if contrastive_mode != "none" and props_train is not None:
            from scipy.stats import spearmanr
            n_check = min(2000, props_train.shape[0])
            phi_check = model.extract_features(z_train_n[:n_check])
            prop_check = props_train[:n_check, contrastive_property_idx].cpu().numpy()

            # Pairwise distances
            phi_dists = torch.cdist(phi_check, phi_check).cpu().numpy()
            import numpy as np
            prop_dists = np.abs(prop_check[:, None] - prop_check[None, :])

            # Extract upper triangle (exclude diagonal)
            triu_idx = np.triu_indices(n_check, k=1)
            rho, pval = spearmanr(phi_dists[triu_idx], prop_dists[triu_idx])
            c_prop_name = prop_names[contrastive_property_idx] if prop_names else f"prop_{contrastive_property_idx}"
            print(f"[φ metric] Spearman(φ_dist, {c_prop_name}_dist) = {rho:.4f} (p={pval:.2e})")
            print(f"[φ metric] (higher ρ = φ-space better reflects property ordering)")

        # Multi-scale quality: per-scale PR and Spearman
        if multi_scale_dims is not None:
            z_check = z_train_n[:2000]
            phis_ms = model.extract_features_multiscale(z_check)
            for si, (phi_s, d_s) in enumerate(zip(phis_ms, multi_scale_dims)):
                phi_s_np = phi_s.cpu().numpy()
                pca_s = PCA(n_components=min(d_s, phi_s_np.shape[0]))
                pca_s.fit(phi_s_np)
                ev_s = pca_s.explained_variance_ratio_
                pr_s = 1.0 / (ev_s ** 2).sum()
                print(f"[scale {d_s}D] PR={pr_s:.1f}, top1={ev_s[0]*100:.1f}%", end="")
                if props_train is not None:
                    from scipy.stats import spearmanr as _spearmanr
                    phi_s_d = torch.cdist(phi_s, phi_s).cpu().numpy()
                    prop_d = np.abs(props_train[:2000, 0].cpu().numpy()[:, None] - props_train[:2000, 0].cpu().numpy()[None, :])
                    _tri = np.triu_indices(phi_s.shape[0], k=1)
                    _rho, _ = _spearmanr(phi_s_d[_tri], prop_d[_tri])
                    print(f", Spearman={_rho:.3f}", end="")
                print()

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Checkpoint saved to {out_dir / 'best_latent_mae.pt'}")


if __name__ == "__main__":
    main()
