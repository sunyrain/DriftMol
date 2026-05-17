#!/usr/bin/env python3
"""Benchmark DriftingMol single-pass inference throughput.

The benchmark records:
  - generator_mol_per_s: one generator forward pass per molecule
  - decoder_logits_mol_per_s: generator + VAE decoder logits/argmax
  - end_to_end_mol_per_s: generator + VAE `sample_smiles` conversion

Use a quiet GPU for final paper numbers; running this while publication
training jobs are active will understate throughput.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.train_selfies_cfg import load_selfies_vae
from src.utils import build_latent_generator


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def batches(total: int, batch_size: int):
    done = 0
    while done < total:
        n = min(batch_size, total - done)
        done += n
        yield n


@torch.no_grad()
def time_loop(fn, total: int, batch_size: int, device: torch.device) -> float:
    sync(device)
    t0 = time.perf_counter()
    for n in batches(total, batch_size):
        fn(n)
    sync(device)
    return time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/publication/pub_F_qed_s43.yaml")
    parser.add_argument("--checkpoint", default="outputs/publication/seeds/pub_F_qed_s43/best.pt")
    parser.add_argument("--output", default="results/inference_benchmark.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--condition", type=float, default=0.0, help="Normalized condition value.")
    args = parser.parse_args()

    device = torch.device(args.device)
    config_path = ROOT / args.config
    ckpt_path = ROOT / args.checkpoint
    output_path = ROOT / args.output

    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    vae = load_selfies_vae(str(ROOT / cfg["vae"]["checkpoint"]), device)
    generator = build_latent_generator(cfg, vae.latent_dim).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    generator.load_state_dict(ckpt["model_state"])

    noise_dim = int(cfg.get("generator", {}).get("noise_dim", generator.noise_dim))
    cond_dim = int(cfg.get("generator", {}).get("cond_dim", 1))

    def make_inputs(n: int):
        noise = torch.randn(n, noise_dim, device=device)
        cond = torch.full((n, cond_dim), float(args.condition), device=device)
        return noise, cond

    for _ in range(args.warmup_batches):
        noise, cond = make_inputs(args.batch_size)
        z = generator(noise, cond=cond, alpha=args.alpha)
        _ = vae.decode(z).argmax(dim=-1)
    sync(device)

    def generator_only(n: int):
        noise, cond = make_inputs(n)
        return generator(noise, cond=cond, alpha=args.alpha)

    def decoder_logits(n: int):
        noise, cond = make_inputs(n)
        z = generator(noise, cond=cond, alpha=args.alpha)
        return vae.decode(z).argmax(dim=-1)

    def end_to_end(n: int):
        noise, cond = make_inputs(n)
        z = generator(noise, cond=cond, alpha=args.alpha)
        return vae.sample_smiles(z, temperature=0.0)

    gen_s = time_loop(generator_only, args.num_samples, args.batch_size, device)
    dec_s = time_loop(decoder_logits, args.num_samples, args.batch_size, device)
    e2e_s = time_loop(end_to_end, args.num_samples, args.batch_size, device)

    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "alpha": args.alpha,
        "condition": args.condition,
        "nfe": 1,
        "two_pass_cfg": False,
        "generator_seconds": gen_s,
        "decoder_logits_seconds": dec_s,
        "end_to_end_seconds": e2e_s,
        "generator_mol_per_s": args.num_samples / gen_s,
        "decoder_logits_mol_per_s": args.num_samples / dec_s,
        "end_to_end_mol_per_s": args.num_samples / e2e_s,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
