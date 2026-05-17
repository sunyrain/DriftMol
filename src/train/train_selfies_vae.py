"""
Training script for the one-shot SELFIES VAE.

Usage:
    python -m src.train.train_selfies_vae --config configs/selfies_vae.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.selfies_vae import (
    SelfiesVAE,
    SelfiesVAEConfig,
    init_vocab,
    set_vocab,
    smiles_to_token_ids,
    token_ids_to_smiles,
    batch_token_ids_to_smiles,
    PAD_IDX,
)
from src.models.selfies_vae_spatial import (
    SelfiesSpatialVAE,
    SelfiesSpatialVAEConfig,
)


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def compute_properties(smi: str) -> list[float] | None:
    """Compute [QED, SA, LogP, MolWt] for a SMILES string."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED
    try:
        from rdkit.Chem import RDConfig
        import sys
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer
    except Exception:
        sascorer = None

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        qed = QED.qed(mol)
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        sa = sascorer.calculateScore(mol) if sascorer else 0.0
        return [qed, sa, logp, mw]
    except Exception:
        return None


@torch.no_grad()
def evaluate(model: SelfiesVAE, data: torch.Tensor, train_smiles_set: set[str],
             device: torch.device, num_samples: int = 2000) -> dict:
    """Evaluate reconstruction + prior sampling quality."""
    from rdkit import Chem
    model.eval()

    # --- Reconstruction ---
    subset = data[:min(2000, len(data))].to(device)
    mu, _ = model.encode(subset)
    logits = model.decode(mu)
    preds = logits.argmax(dim=-1).cpu()
    mask = subset.cpu() != PAD_IDX
    token_acc = (preds[mask] == subset.cpu()[mask]).float().mean().item()

    exact_match = 0
    for i in range(len(subset)):
        pred_smi = token_ids_to_smiles(preds[i])
        orig_smi = token_ids_to_smiles(subset[i].cpu())
        try:
            pm = Chem.MolFromSmiles(pred_smi)
            om = Chem.MolFromSmiles(orig_smi)
            if pm and om and Chem.MolToSmiles(pm) == Chem.MolToSmiles(om):
                exact_match += 1
        except Exception:
            pass
    exact_rate = exact_match / len(subset)

    # --- Prior sampling ---
    z_rand = torch.randn(num_samples, model.cfg.latent_dim, device=device)
    logits_r = model.decode(z_rand)
    pred_ids = logits_r.argmax(dim=-1).cpu()
    smiles_list = batch_token_ids_to_smiles(pred_ids)

    valid_count = 0
    unique_set: set[str] = set()
    for smi in smiles_list:
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_count += 1
            unique_set.add(Chem.MolToSmiles(mol))

    novel_count = sum(1 for s in unique_set if s not in train_smiles_set)

    v = valid_count / num_samples
    u = len(unique_set) / max(valid_count, 1)
    n = novel_count / max(len(unique_set), 1)

    return {
        "token_acc": token_acc,
        "exact_recon": exact_rate,
        "prior_validity": v,
        "prior_uniqueness": u,
        "prior_novelty": n,
        "prior_vun": v * u * n,
        "prior_num_unique": len(unique_set),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_cfg = cfg["experiment"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    out_dir = Path(exp_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    with open(out_dir / "resolved_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    device = torch.device("cuda" if train_cfg.get("use_cuda", True)
                          and torch.cuda.is_available() else "cpu")
    torch.manual_seed(exp_cfg.get("seed", 42))

    # ── Load data ─────────────────────────────────────────────────
    print("[data] Loading QM9 graph cache...")
    cache = torch.load(data_cfg["cache_path"], map_location="cpu")
    all_smiles: list[str] = cache["smiles"]
    print(f"  Total SMILES: {len(all_smiles)}")

    # Build vocabulary
    print("[data] Building SELFIES vocabulary...")
    vocab, tok2idx, idx2tok = init_vocab(all_smiles)
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Tokens: {vocab}")

    max_len = model_cfg.get("max_len", 20)

    # Tokenize all data
    print("[data] Tokenizing...")
    token_data = []
    selfies_failures = 0
    for smi in all_smiles:
        ids = smiles_to_token_ids(smi, max_len)
        if ids is None:
            selfies_failures += 1
            continue
        token_data.append(ids)
    data_tensor = torch.stack(token_data)
    print(f"  Tokenized: {len(token_data)} / {len(all_smiles)} "
          f"(failures: {selfies_failures})")
    print(f"  Shape: {data_tensor.shape}")

    # Train/val split
    N = len(data_tensor)
    n_train = int(N * data_cfg.get("train_ratio", 0.9))
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(42))
    train_data = data_tensor[perm[:n_train]]
    val_data = data_tensor[perm[n_train:]]
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # Build train SMILES set (for novelty computation)
    from rdkit import Chem
    train_smiles_set: set[str] = set()
    for smi in all_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            train_smiles_set.add(Chem.MolToSmiles(mol))

    # ── Build model ───────────────────────────────────────────────
    model_type = model_cfg.get("type", "flat")

    if model_type == "spatial":
        vae_cfg = SelfiesSpatialVAEConfig(
            max_len=max_len,
            vocab_size=len(vocab),
            latent_channels=int(model_cfg.get("latent_channels", 4)),
            latent_height=int(model_cfg.get("latent_height", 8)),
            latent_width=int(model_cfg.get("latent_width", 8)),
            hidden_dim=model_cfg.get("hidden_dim", 512),
            num_layers=model_cfg.get("num_layers", 6),
            num_heads=model_cfg.get("num_heads", 8),
            ff_mult=model_cfg.get("ff_mult", 4),
            dropout=model_cfg.get("dropout", 0.1),
            spatial_mid_channels=int(model_cfg.get("spatial_mid_channels", 64)),
            num_spatial_blocks=int(model_cfg.get("num_spatial_blocks", 4)),
            dec_num_layers=int(model_cfg.get("dec_num_layers", 4)),
            beta=train_cfg.get("beta", 0.005),
            num_properties=model_cfg.get("num_properties", 0),
            decoder_noise_std=float(model_cfg.get("decoder_noise_std", 0.0)),
            free_bits=float(model_cfg.get("free_bits", 0.25)),
            pos_dropout=float(model_cfg.get("pos_dropout", 0.15)),
            pad_loss_weight=float(model_cfg.get("pad_loss_weight", 0.5)),
        )
        model = SelfiesSpatialVAE(vae_cfg).to(device)
        print(f"[model] SelfiesSpatialVAE: latent={vae_cfg.latent_channels}×"
              f"{vae_cfg.latent_height}×{vae_cfg.latent_width}="
              f"{vae_cfg.latent_dim}d")
    else:
        vae_cfg = SelfiesVAEConfig(
            max_len=max_len,
            vocab_size=len(vocab),
            latent_dim=model_cfg.get("latent_dim", 128),
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 4),
            num_heads=model_cfg.get("num_heads", 4),
            ff_mult=model_cfg.get("ff_mult", 4),
            dropout=model_cfg.get("dropout", 0.1),
            beta=train_cfg.get("beta", 0.01),
            num_properties=model_cfg.get("num_properties", 0),
            decoder_noise_std=float(model_cfg.get("decoder_noise_std", 0.0)),
            free_bits=float(model_cfg.get("free_bits", 0.0)),
            pos_dropout=float(model_cfg.get("pos_dropout", 0.0)),
            dec_num_layers=int(model_cfg.get("dec_num_layers", 0)),
            pad_loss_weight=float(model_cfg.get("pad_loss_weight", 0.5)),
        )
        model = SelfiesVAE(vae_cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {type(model).__name__}: {n_params / 1e6:.2f}M params")
    print(f"  latent_dim={vae_cfg.latent_dim}, hidden={vae_cfg.hidden_dim}, "
          f"layers={vae_cfg.num_layers}, heads={vae_cfg.num_heads}")

    # ── Multi-GPU DataParallel ────────────────────────────────────
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"[multi-gpu] Using DataParallel on {n_gpus} GPUs")
        model = nn.DataParallel(model)
    # Helper to access underlying model (DataParallel wraps in .module)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    # ── Optimizer ─────────────────────────────────────────────────
    lr = float(train_cfg.get("lr", 3e-4))
    wd = float(train_cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    epochs = int(train_cfg.get("epochs", 100))
    bs = int(train_cfg.get("batch_size", 512))
    if n_gpus > 1:
        # Don't scale batch size — DataParallel splits it across GPUs automatically
        # This gives ~N× throughput at same memory footprint per GPU
        print(f"  DataParallel: bs={bs} split across {n_gpus} GPUs ({bs // n_gpus}/GPU)")
    eval_every = int(train_cfg.get("eval_every", 5))
    patience = int(train_cfg.get("patience", 20))
    beta = float(train_cfg.get("beta", 0.01))
    beta_warmup = int(train_cfg.get("beta_warmup_epochs", 0))  # linear warmup from 0→beta
    kl_cycles = int(train_cfg.get("kl_cycles", 0))  # cyclical annealing (0=disabled)
    selection_metric = train_cfg.get("selection_metric", "vun")  # vun or val_loss
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    # ── Resume ────────────────────────────────────────────────────
    start_epoch = 0
    best_score = -1.0 if selection_metric == "vun" else float("inf")
    no_improve = 0

    if args.resume and (out_dir / "last.pt").exists():
        ckpt = torch.load(out_dir / "last.pt", map_location=device)
        raw_model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        # Reset best_score if selection metric changed
        old_metric = ckpt.get("selection_metric", "vun")
        if old_metric == selection_metric:
            best_score = ckpt.get("best_score", best_score)
            no_improve = ckpt.get("no_improve", 0)
        else:
            print(f"  [warn] selection metric changed ({old_metric} → {selection_metric}), resetting best_score")
            no_improve = 0
        print(f"[resume] from epoch {start_epoch}, best_score={best_score:.4f}")

    # ── Training loop ─────────────────────────────────────────────
    print(f"\n[train] {epochs} epochs, bs={bs}, lr={lr}, β={beta}, "
          f"eval_every={eval_every}, patience={patience}")
    if beta_warmup > 0:
        print(f"  KL warmup: 0 \u2192 {beta} over {beta_warmup} epochs")
    if kl_cycles > 0:
        print(f"  KL cycles: {kl_cycles} cycles over {epochs} epochs")
    if vae_cfg.free_bits > 0:
        print(f"  Free bits: {vae_cfg.free_bits} per dim (min total KL = {vae_cfg.free_bits * vae_cfg.latent_dim:.1f})")
    if vae_cfg.pos_dropout > 0:
        print(f"  Pos dropout: {vae_cfg.pos_dropout}")
    if vae_cfg.dec_num_layers > 0:
        print(f"  Decoder layers: {vae_cfg.dec_num_layers} (encoder: {vae_cfg.num_layers})")
    history = []

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        perm_e = torch.randperm(len(train_data))
        total_loss, total_recon, total_kl, total_acc = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        # Compute effective beta for this epoch (warmup / cycling)
        if kl_cycles > 0:
            # Cyclical annealing: linearly increase beta within each cycle
            cycle_len = epochs / kl_cycles
            cycle_pos = (epoch % cycle_len) / cycle_len  # 0→1 within cycle
            eff_beta = beta * min(cycle_pos * 2, 1.0)  # ramp up in first half
        elif beta_warmup > 0 and epoch < beta_warmup:
            eff_beta = beta * (epoch / beta_warmup)
        else:
            eff_beta = beta

        for i in range(0, len(train_data), bs):
            batch = train_data[perm_e[i:i + bs]].to(device)
            out = model(batch, beta=eff_beta)

            # DataParallel gathers per-GPU scalars into a vector; mean them
            loss = out["loss"].mean()
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Token accuracy (use raw_model on device 0 to avoid DataParallel overhead)
            with torch.no_grad():
                sub = batch[:min(len(batch), 1024)]  # small subset for speed
                mu, _ = raw_model.encode(sub)
                pred = raw_model.decode(mu).argmax(-1)
                mask = sub != PAD_IDX
                acc = (pred[mask] == sub[mask]).float().mean().item()

            total_loss += out["loss"].mean().item()
            total_recon += out["recon_loss"].mean().item()
            total_kl += out["kl_loss"].mean().item()
            total_acc += acc
            n_batches += 1

        scheduler.step()
        dt = time.time() - t0
        avg = lambda x: x / n_batches
        beta_str = f"\u03b2={eff_beta:.4f}" if (beta_warmup > 0 or kl_cycles > 0) else f"\u03b2={beta}"
        line = (f"[epoch {epoch:3d}] lr={optimizer.param_groups[0]['lr']:.2e} "
                f"loss={avg(total_loss):.4f} recon={avg(total_recon):.4f} "
                f"kl={avg(total_kl):.1f} {beta_str} tok_acc={avg(total_acc):.4f} "
                f"({dt:.0f}s)")

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_bs = min(len(val_data), 4096)
            vb = val_data[:val_bs].to(device)
            vout = raw_model(vb, beta=eff_beta)
            val_loss = vout["loss"].item()
            line += f" val_loss={val_loss:.4f}"
            # Log active latent dims (std > 0.1)
            mu_v, _ = raw_model.encode(vb)
            mu_std = mu_v.std(0)
            active_dims = (mu_std > 0.1).sum().item()
            line += f" adim={active_dims}/{vae_cfg.latent_dim}"

        # Eval
        epoch_metrics = {
            "epoch": epoch,
            "loss": avg(total_loss),
            "recon": avg(total_recon),
            "kl": avg(total_kl),
            "tok_acc": avg(total_acc),
            "val_loss": val_loss,
        }

        if epoch % eval_every == eval_every - 1 or epoch == 0:
            metrics = evaluate(raw_model, val_data, train_smiles_set, device,
                               num_samples=2000)
            line += (f" exact={metrics['exact_recon']*100:.1f}%"
                     f" | prior: V={metrics['prior_validity']*100:.0f}%"
                     f" U={metrics['prior_uniqueness']*100:.1f}%"
                     f" N={metrics['prior_novelty']*100:.1f}%"
                     f" VUN={metrics['prior_vun']:.3f}")
            epoch_metrics.update(metrics)

            if selection_metric == "val_loss":
                score = val_loss
                improved = score < best_score
            else:
                score = metrics["prior_vun"]
                improved = score > best_score
            if improved:
                best_score = score
                no_improve = 0
                # Save best
                torch.save({
                    "model_state": raw_model.state_dict(),
                    "cfg": cfg,
                    "vae_cfg": {k: v for k, v in vae_cfg.__dict__.items()
                                if not k.startswith('_')},
                    "model_type": model_type,
                    "vocab": vocab,
                    "epoch": epoch,
                    "best_score": best_score,
                }, out_dir / "best.pt")
                line += " ★"
            else:
                no_improve += eval_every

        print(line, flush=True)
        history.append(epoch_metrics)

        # Save last checkpoint
        torch.save({
            "model_state": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "no_improve": no_improve,
            "selection_metric": selection_metric,
            "cfg": cfg,
            "vae_cfg": {k: v for k, v in vae_cfg.__dict__.items()
                        if not k.startswith('_')},
            "model_type": model_type,
            "vocab": vocab,
        }, out_dir / "last.pt")

        if no_improve >= patience:
            print(f"[early stop] no improvement for {patience} epochs")
            break

    # ── Final eval ────────────────────────────────────────────────
    print("\n[final] Loading best checkpoint and evaluating...")
    best_ckpt = torch.load(out_dir / "best.pt", map_location=device)
    raw_model.load_state_dict(best_ckpt["model_state"])
    final_metrics = evaluate(raw_model, val_data, train_smiles_set, device,
                              num_samples=5000)
    print(f"  Exact recon: {final_metrics['exact_recon']*100:.1f}%")
    print(f"  Prior: V={final_metrics['prior_validity']*100:.1f}% "
          f"U={final_metrics['prior_uniqueness']*100:.1f}% "
          f"N={final_metrics['prior_novelty']*100:.1f}%"
          f" VUN={final_metrics['prior_vun']:.4f}")

    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    with open(out_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("[done]")


if __name__ == "__main__":
    main()
