#!/usr/bin/env python3
"""Train same-backbone conditional generative baselines in SELFIES latent space.

These baselines answer how DriftingMol compares with representative
*generative* families under the same ZINC250K latent cache, SELFIES decoder,
QED target-bin protocol, and metric code.

Implemented model families:

* cvae: conditional latent VAE prior, p(z | y)
* gan: conditional latent WGAN-GP
* diffusion: conditional latent DDPM in normalized latent space
* flow_matching: conditional latent flow matching with Euler sampling

The models generate normalized VAE latents, which are de-normalized before
decoding with the frozen SELFIES VAE.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_matched_baselines import (  # noqa: E402
    build_qed_bins,
    canonicalize,
    fmt,
    load_vae,
    molecular_metrics,
    pct,
)

RDLogger.DisableLog("rdApp.warning")

RESULT_DIR = ROOT / "results" / "generative_baselines"
SUMMARY_JSON = ROOT / "results" / "generative_baselines_qed.json"
SUMMARY_TEX = ROOT / "results" / "tables" / "tab_generative_baselines_qed.tex"


METHOD_LABELS = {
    "cvae": "Conditional latent VAE",
    "gan": "Conditional latent WGAN-GP",
    "diffusion": "Conditional latent DDPM",
    "flow_matching": "Conditional latent Flow Matching",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mlp(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for _ in range(max(1, num_layers)):
        layers.extend([nn.Linear(last, hidden_dim), nn.SiLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last = hidden_dim
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding for scalar t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device, dtype=t.dtype)
    )
    args = t.view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ConditionalVAE(nn.Module):
    def __init__(self, z_dim: int, cond_dim: int, noise_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.noise_dim = noise_dim
        self.encoder = mlp(z_dim + cond_dim, 2 * noise_dim, hidden_dim, num_layers)
        self.decoder = mlp(noise_dim + cond_dim, z_dim, hidden_dim, num_layers)

    def encode(self, z: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([z, cond], dim=1))
        return h.chunk(2, dim=1)

    def decode(self, eps: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([eps, cond], dim=1))

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(z, cond)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        rec = self.decode(mu + eps * std, cond)
        return rec, mu, logvar


class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim: int, cond_dim: int, noise_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.noise_dim = noise_dim
        self.net = mlp(noise_dim + cond_dim, z_dim, hidden_dim, num_layers)

    def forward(self, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([noise, cond], dim=1))


class ConditionalCritic(nn.Module):
    def __init__(self, z_dim: int, cond_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.net = mlp(z_dim + cond_dim, 1, hidden_dim, num_layers)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, cond], dim=1)).view(-1)


class ConditionalTimeNet(nn.Module):
    def __init__(
        self,
        z_dim: int,
        cond_dim: int,
        hidden_dim: int,
        num_layers: int,
        time_dim: int = 64,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.net = mlp(z_dim + cond_dim + time_dim, z_dim, hidden_dim, num_layers)

    def forward(self, z: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, cond, time_embedding(t, self.time_dim)], dim=1))


@dataclass
class LatentData:
    train_z: torch.Tensor
    train_cond: torch.Tensor
    train_qed_raw: torch.Tensor
    train_smiles: list[str]
    train_canon: set[str]
    z_mean: torch.Tensor
    z_std: torch.Tensor
    prop_mean: torch.Tensor
    prop_std: torch.Tensor
    bins: list[torch.Tensor]
    centers_raw: list[float]


def load_latent_data(cache_path: Path, device: torch.device, n_bins: int) -> LatentData:
    cache = torch.load(cache_path, map_location="cpu")
    train_z_raw = cache["train"].float()
    train_qed_raw = cache["train_props"][:, 0].float()
    z_mean = train_z_raw.mean(dim=0, keepdim=True)
    z_std = train_z_raw.std(dim=0, keepdim=True).clamp(min=1e-6)
    prop_mean = train_qed_raw.mean().view(1, 1)
    prop_std = train_qed_raw.std().clamp(min=1e-6).view(1, 1)
    train_z = (train_z_raw - z_mean) / z_std
    train_cond = ((train_qed_raw.view(-1, 1) - prop_mean) / prop_std).float()
    train_smiles = [str(s).strip() for s in cache["train_smiles"]]
    train_canon = {can for smi in train_smiles if (can := canonicalize(smi)) is not None}
    bins, centers_raw = build_qed_bins(train_qed_raw, n_bins)
    return LatentData(
        train_z=train_z.to(device),
        train_cond=train_cond.to(device),
        train_qed_raw=train_qed_raw,
        train_smiles=train_smiles,
        train_canon=train_canon,
        z_mean=z_mean.to(device),
        z_std=z_std.to(device),
        prop_mean=prop_mean.to(device),
        prop_std=prop_std.to(device),
        bins=bins,
        centers_raw=centers_raw,
    )


def sample_batch(data: LatentData, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, data.train_z.shape[0], (batch_size,), device=data.train_z.device)
    return data.train_z[idx], data.train_cond[idx]


def train_cvae(args: argparse.Namespace, data: LatentData, z_dim: int, device: torch.device) -> nn.Module:
    model = ConditionalVAE(z_dim, 1, args.noise_dim, args.hidden_dim, args.num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = recon_total = kl_total = 0.0
        for _ in range(args.steps_per_epoch):
            z, y = sample_batch(data, args.batch_size)
            rec, mu, logvar = model(z, y)
            recon = F.mse_loss(rec, z)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + args.kl_beta * kl
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total += float(loss.item())
            recon_total += float(recon.item())
            kl_total += float(kl.item())
        item = {
            "epoch": epoch,
            "loss": total / args.steps_per_epoch,
            "recon": recon_total / args.steps_per_epoch,
            "kl": kl_total / args.steps_per_epoch,
        }
        history.append(item)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"[cvae] epoch={epoch:03d} loss={item['loss']:.4f} recon={item['recon']:.4f} kl={item['kl']:.4f}", flush=True)
    save_training_history(args, history)
    return model


def gradient_penalty(
    critic: ConditionalCritic,
    real: torch.Tensor,
    fake: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    eps = torch.rand(real.shape[0], 1, device=real.device)
    mixed = eps * real + (1.0 - eps) * fake
    mixed.requires_grad_(True)
    score = critic(mixed, cond)
    grad = torch.autograd.grad(
        outputs=score.sum(),
        inputs=mixed,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()


def train_gan(args: argparse.Namespace, data: LatentData, z_dim: int, device: torch.device) -> nn.Module:
    gen = ConditionalGenerator(z_dim, 1, args.noise_dim, args.hidden_dim, args.num_layers).to(device)
    critic = ConditionalCritic(z_dim, 1, args.hidden_dim, args.num_layers).to(device)
    opt_g = torch.optim.AdamW(gen.parameters(), lr=args.lr, betas=(0.0, 0.9), weight_decay=args.weight_decay)
    opt_d = torch.optim.AdamW(critic.parameters(), lr=args.lr, betas=(0.0, 0.9), weight_decay=args.weight_decay)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        gen.train()
        critic.train()
        g_total = d_total = gp_total = 0.0
        for _ in range(args.steps_per_epoch):
            for _ in range(args.n_critic):
                real, y = sample_batch(data, args.batch_size)
                noise = torch.randn(args.batch_size, args.noise_dim, device=device)
                fake = gen(noise, y).detach()
                gp = gradient_penalty(critic, real, fake, y)
                d_loss = critic(fake, y).mean() - critic(real, y).mean() + args.gp_lambda * gp
                opt_d.zero_grad()
                d_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
                opt_d.step()
                d_total += float(d_loss.item())
                gp_total += float(gp.item())
            _, y = sample_batch(data, args.batch_size)
            noise = torch.randn(args.batch_size, args.noise_dim, device=device)
            fake = gen(noise, y)
            g_loss = -critic(fake, y).mean()
            opt_g.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(gen.parameters(), args.grad_clip)
            opt_g.step()
            g_total += float(g_loss.item())
        denom_d = args.steps_per_epoch * args.n_critic
        item = {
            "epoch": epoch,
            "g_loss": g_total / args.steps_per_epoch,
            "d_loss": d_total / denom_d,
            "gp": gp_total / denom_d,
        }
        history.append(item)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"[gan] epoch={epoch:03d} g={item['g_loss']:.4f} d={item['d_loss']:.4f} gp={item['gp']:.4f}", flush=True)
    save_training_history(args, history)
    return gen


def diffusion_schedule(steps: int, device: torch.device) -> dict[str, torch.Tensor]:
    betas = torch.linspace(1e-4, 2e-2, steps, device=device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "abar": abar,
        "sqrt_abar": torch.sqrt(abar),
        "sqrt_one_minus_abar": torch.sqrt(1.0 - abar),
    }


def train_diffusion(args: argparse.Namespace, data: LatentData, z_dim: int, device: torch.device) -> nn.Module:
    model = ConditionalTimeNet(z_dim, 1, args.hidden_dim, args.num_layers, args.time_dim).to(device)
    sched = diffusion_schedule(args.diffusion_steps, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(args.steps_per_epoch):
            z0, y = sample_batch(data, args.batch_size)
            tidx = torch.randint(0, args.diffusion_steps, (args.batch_size,), device=device)
            eps = torch.randn_like(z0)
            zt = sched["sqrt_abar"][tidx].view(-1, 1) * z0 + sched["sqrt_one_minus_abar"][tidx].view(-1, 1) * eps
            t = tidx.float() / max(args.diffusion_steps - 1, 1)
            pred = model(zt, y, t)
            loss = F.mse_loss(pred, eps)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total += float(loss.item())
        item = {"epoch": epoch, "loss": total / args.steps_per_epoch}
        history.append(item)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"[diffusion] epoch={epoch:03d} loss={item['loss']:.4f}", flush=True)
    save_training_history(args, history)
    return model


def train_flow_matching(args: argparse.Namespace, data: LatentData, z_dim: int, device: torch.device) -> nn.Module:
    model = ConditionalTimeNet(z_dim, 1, args.hidden_dim, args.num_layers, args.time_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(args.steps_per_epoch):
            z1, y = sample_batch(data, args.batch_size)
            z0 = torch.randn_like(z1)
            t = torch.rand(z1.shape[0], device=device)
            zt = (1.0 - t.view(-1, 1)) * z0 + t.view(-1, 1) * z1
            target_v = z1 - z0
            pred_v = model(zt, y, t)
            loss = F.mse_loss(pred_v, target_v)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total += float(loss.item())
        item = {"epoch": epoch, "loss": total / args.steps_per_epoch}
        history.append(item)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"[flow_matching] epoch={epoch:03d} loss={item['loss']:.4f}", flush=True)
    save_training_history(args, history)
    return model


def save_training_history(args: argparse.Namespace, history: list[dict[str, float]]) -> None:
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_history.json").write_text(json.dumps(history, indent=2) + "\n")


@torch.no_grad()
def sample_normalized_latents(
    method: str,
    model: nn.Module,
    cond: torch.Tensor,
    n: int,
    args: argparse.Namespace,
    device: torch.device,
    z_dim: int,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    if method == "diffusion":
        sched = diffusion_schedule(args.diffusion_steps, device)
    for start in range(0, n, args.eval_batch_size):
        batch = min(args.eval_batch_size, n - start)
        y = cond.expand(batch, -1)
        if method == "cvae":
            eps = torch.randn(batch, args.noise_dim, device=device)
            z = model.decode(eps, y)
        elif method == "gan":
            eps = torch.randn(batch, args.noise_dim, device=device)
            z = model(eps, y)
        elif method == "flow_matching":
            z = torch.randn(batch, z_dim, device=device)
            dt = 1.0 / args.sample_steps
            for step in range(args.sample_steps):
                t = torch.full((batch,), (step + 0.5) * dt, device=device)
                z = z + model(z, y, t) * dt
        elif method == "diffusion":
            z = torch.randn(batch, z_dim, device=device)
            for step in reversed(range(args.diffusion_steps)):
                tidx = torch.full((batch,), step, device=device, dtype=torch.long)
                t = tidx.float() / max(args.diffusion_steps - 1, 1)
                eps_pred = model(z, y, t)
                beta = sched["betas"][step]
                alpha = sched["alphas"][step]
                abar = sched["abar"][step]
                mean = (z - beta / torch.sqrt(1.0 - abar) * eps_pred) / torch.sqrt(alpha)
                if step > 0:
                    z = mean + torch.sqrt(beta) * torch.randn_like(z)
                else:
                    z = mean
        else:
            raise ValueError(f"Unknown method: {method}")
        pieces.append(z.detach().cpu())
    return torch.cat(pieces, dim=0)


@torch.no_grad()
def evaluate(
    method: str,
    model: nn.Module,
    data: LatentData,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    vae = load_vae(device)
    model.eval()
    z_dim = data.train_z.shape[1]
    per_bin = args.num_samples // args.n_bins
    all_smiles: list[str] = []
    all_targets: list[float] = []
    bin_summaries: list[dict[str, Any]] = []
    for bin_idx, target_raw in enumerate(data.centers_raw):
        cond_raw = torch.tensor([[target_raw]], dtype=torch.float32, device=device)
        cond_norm = (cond_raw - data.prop_mean) / data.prop_std
        z_norm_cpu = sample_normalized_latents(method, model, cond_norm, per_bin, args, device, z_dim)
        z_norm = z_norm_cpu.to(device)
        z_raw = z_norm * data.z_std + data.z_mean
        smiles: list[str] = []
        for start in range(0, per_bin, args.eval_batch_size):
            smiles.extend(vae.sample_smiles(z_raw[start:start + args.eval_batch_size], temperature=0.0))
        all_smiles.extend(smiles[:per_bin])
        all_targets.extend([target_raw] * per_bin)
        bin_summaries.append({"bin": bin_idx, "target": float(target_raw), "selected": per_bin})
        print(f"[eval] bin={bin_idx:02d} target={target_raw:.3f} selected={per_bin}", flush=True)
    metrics = molecular_metrics(all_smiles, all_targets, data.train_canon)
    nfe = 1
    if method == "diffusion":
        nfe = args.diffusion_steps
    elif method == "flow_matching":
        nfe = args.sample_steps
    return {
        "protocol": {
            "baseline_type": "same-backbone conditional generative baseline",
            "method": method,
            "label": METHOD_LABELS[method],
            "target_property": "qed",
            "num_samples": args.num_samples,
            "n_bins": args.n_bins,
            "samples_per_bin": per_bin,
            "seed": args.seed,
            "nfe": nfe,
            "normalized_latent_space": True,
            "selfies_decoder": "outputs/foundation/zinc_selfies_vae_v2/best.pt",
        },
        "target_centers": data.centers_raw,
        "bin_summaries": bin_summaries,
        "metrics": metrics,
    }


def result_path(method: str, seed: int) -> Path:
    return RESULT_DIR / f"{method}_qed_s{seed}.json"


def write_result(args: argparse.Namespace, payload: dict[str, Any], model: nn.Module) -> None:
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": args.method,
            "model_state": model.state_dict(),
            "args": vars(args),
        },
        out_dir / "model.pt",
    )
    (out_dir / "final_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_path(args.method, args.seed).write_text(json.dumps(payload, indent=2) + "\n")


def load_all_results() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(RESULT_DIR.glob("*_qed_s*.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return rows


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def fmt_mean_std(mean: float | None, std: float | None, digits: int = 3) -> str:
    if mean is None:
        return "---"
    if std is None or std == 0:
        return fmt(mean, digits)
    return f"{mean:.{digits}f}$\\pm${std:.{digits}f}"


def pct_mean_std(mean: float | None, std: float | None, digits: int = 1) -> str:
    if mean is None:
        return "---"
    mean_pct = 100.0 * mean
    std_pct = 0.0 if std is None else 100.0 * std
    if std_pct == 0:
        return f"{mean_pct:.{digits}f}"
    return f"{mean_pct:.{digits}f}$\\pm${std_pct:.{digits}f}"


def aggregate_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        method = str(row.get("protocol", {}).get("method", ""))
        if method:
            grouped.setdefault(method, []).append(row)
    out: list[dict[str, Any]] = []
    order = {"cvae": 0, "gan": 1, "diffusion": 2, "flow_matching": 3}
    for method in sorted(grouped, key=lambda key: order.get(key, 99)):
        items = sorted(grouped[method], key=lambda item: item.get("protocol", {}).get("seed", 999))
        protocol0 = items[0].get("protocol", {})

        def vals(key: str) -> list[float]:
            values = []
            for item in items:
                value = item.get("metrics", {}).get(key)
                if value is not None:
                    values.append(float(value))
            return values

        metrics: dict[str, dict[str, float | None]] = {}
        for key in ["spearman_rho", "mae", "slope", "uniqueness", "novelty", "int_div", "success_0p10"]:
            m, s = mean_std(vals(key))
            metrics[key] = {"mean": m, "std": s}
        out.append(
            {
                "method": method,
                "label": protocol0.get("label", METHOD_LABELS.get(method, method)),
                "nfe": protocol0.get("nfe"),
                "n": len(items),
                "seeds": [item.get("protocol", {}).get("seed") for item in items],
                "metrics": metrics,
            }
        )
    return out


def write_summary() -> None:
    rows = load_all_results()
    aggregates = aggregate_results(rows)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps({"aggregates": aggregates, "results": rows}, indent=2) + "\n")
    SUMMARY_TEX.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated by scripts/train_latent_generative_baseline.py",
        "\\begin{tabular}{l c c c c c c c c c}",
        "\\toprule",
        "Generative baseline & Seeds & NFE & $\\rho$ & MAE & Slope & U (\\%) & N (\\%) & IntDiv & Succ@0.10 \\\\",
        "\\midrule",
    ]
    for row in aggregates:
        metrics = row["metrics"]
        lines.append(
            f"{row['label']} & {row['n']} & {row.get('nfe', '---')} & "
            f"{fmt_mean_std(metrics['spearman_rho']['mean'], metrics['spearman_rho']['std'])} & "
            f"{fmt_mean_std(metrics['mae']['mean'], metrics['mae']['std'])} & "
            f"{fmt_mean_std(metrics['slope']['mean'], metrics['slope']['std'])} & "
            f"{pct_mean_std(metrics['uniqueness']['mean'], metrics['uniqueness']['std'])} & "
            f"{pct_mean_std(metrics['novelty']['mean'], metrics['novelty']['std'])} & "
            f"{fmt_mean_std(metrics['int_div']['mean'], metrics['int_div']['std'])} & "
            f"{pct_mean_std(metrics['success_0p10']['mean'], metrics['success_0p10']['std'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    SUMMARY_TEX.write_text("\n".join(lines) + "\n")


def train_model(args: argparse.Namespace, data: LatentData, device: torch.device) -> nn.Module:
    z_dim = data.train_z.shape[1]
    if args.method == "cvae":
        return train_cvae(args, data, z_dim, device)
    if args.method == "gan":
        return train_gan(args, data, z_dim, device)
    if args.method == "diffusion":
        return train_diffusion(args, data, z_dim, device)
    if args.method == "flow_matching":
        return train_flow_matching(args, data, z_dim, device)
    raise ValueError(f"Unknown method: {args.method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHOD_LABELS), required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache", default="data/cache/zinc_latent_cache_v2.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--steps-per-epoch", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--noise-dim", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--n-critic", type=int, default=3)
    parser.add_argument("--gp-lambda", type=float, default=10.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f"outputs/publication_ext/generative_baselines/{args.method}_qed_s{args.seed}"
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    data = load_latent_data(ROOT / args.cache, device, args.n_bins)
    print(
        f"[setup] method={args.method} z_dim={data.train_z.shape[1]} "
        f"N={data.train_z.shape[0]} device={device}",
        flush=True,
    )
    t0 = time.time()
    model = train_model(args, data, device)
    payload = evaluate(args.method, model, data, args, device)
    payload["runtime"] = {"total_seconds": time.time() - t0}
    write_result(args, payload, model)
    write_summary()
    metrics = payload["metrics"]
    print(f"Wrote {ROOT / args.output_dir / 'final_metrics.json'}")
    print(f"Wrote {result_path(args.method, args.seed)}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_TEX}")
    print(
        {
            key: metrics.get(key)
            for key in ["spearman_rho", "mae", "slope", "uniqueness", "novelty", "int_div", "success_0p10"]
        }
    )


if __name__ == "__main__":
    main()
