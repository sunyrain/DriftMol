"""
Shared utilities for the Drifting Molecule Generation pipeline (SELFIES line).

Provides:
  - LatentGenerator (MLP), LatentDiTGenerator, LatentDiTGeneratorCFG model classes
  - build_latent_generator: factory function from config
  - build_lr_scheduler: warmup + cosine / linear / constant LR schedule
  - set_seed, load_config, save_json: common utilities

Note: Graph VAE utilities (load_vae, discretize_logits) are exposed below as
      archive-backed compatibility helpers for the graph stress package.
"""
from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required.") from exc


_GRAPH_ARCHIVE_ROOT = Path(__file__).resolve().parents[1] / "archive" / "graph_vae_line"


@lru_cache(maxsize=None)
def _load_graph_module(module_name: str, rel_path: str):
    module_path = _GRAPH_ARCHIVE_ROOT / rel_path
    if not module_path.exists():
        raise FileNotFoundError(f"Missing archived graph module: {module_path}")

    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ── Generator Models ─────────────────────────────────────────────────

class LatentGenerator(nn.Module):
    """Baseline MLP generator: noise → z ∈ R^latent_dim."""

    def __init__(self, noise_dim: int = 32, hidden_dim: int = 256,
                 latent_dim: int = 32, num_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim

        layers: list[nn.Module] = [nn.Linear(noise_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ])
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LatentDiTGenerator(nn.Module):
    """Transformer/DiT-style latent generator: noise → token sequence → pooled latent z."""

    def __init__(
        self,
        noise_dim: int = 64,
        hidden_dim: int = 384,
        latent_dim: int = 64,
        num_layers: int = 8,
        num_heads: int = 8,
        num_tokens: int = 16,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        token_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.num_tokens = num_tokens
        self.token_noise_std = token_noise_std

        self.input_proj = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.token_proj = nn.Linear(hidden_dim, num_tokens * hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)

        ff_dim = hidden_dim * mlp_ratio
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        bsz = z.shape[0]
        h = self.input_proj(z)
        tokens = self.token_proj(h).view(bsz, self.num_tokens, -1)
        tokens = tokens + self.pos_embed
        if self.training and self.token_noise_std > 0:
            tokens = tokens + self.token_noise_std * torch.randn_like(tokens)
        tokens = self.encoder(tokens)
        query = self.pool_query.expand(bsz, -1, -1)
        pooled, _ = self.pool_attn(query, tokens, tokens, need_weights=False)
        return self.out(pooled.squeeze(1))


class LatentDiTGeneratorCFG(nn.Module):
    """
    Classifier-Free Guidance (CFG) enabled LatentDiT Generator.

    Extends LatentDiTGenerator with:
      - Condition embedding: property vector → FiLM modulation at each Transformer layer
      - Null condition token for unconditional path
      - Single-pass CFG inference via an embedded guidance scale α

    The condition is a vector of molecular properties (e.g., QED, SA, LogP, MolWt).
    During training, conditions are randomly dropped with probability p_uncond
    to learn both conditional and unconditional generation.  At inference time,
    sample_cfg calls one forward pass with (condition, α) rather than two-pass
    interpolation.
    """

    def __init__(
        self,
        noise_dim: int = 64,
        hidden_dim: int = 384,
        latent_dim: int = 64,
        num_layers: int = 8,
        num_heads: int = 8,
        num_tokens: int = 16,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        token_noise_std: float = 0.0,
        cond_dim: int = 4,              # number of conditioning properties
        p_uncond: float = 0.1,          # probability of dropping condition (CFG training)
        num_classes: int = 0,           # >0: discrete class conditioning via nn.Embedding
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.num_tokens = num_tokens
        self.token_noise_std = token_noise_std
        self.cond_dim = cond_dim
        self.p_uncond = p_uncond
        self.num_classes = num_classes

        # Noise → hidden
        self.input_proj = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Token projection
        self.token_proj = nn.Linear(hidden_dim, num_tokens * hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)

        # Condition embedding
        if num_classes > 0:
            # Discrete class conditioning (like ImageNet).
            # Index num_classes = null/unconditional class.
            self.class_embed = nn.Embedding(num_classes + 1, hidden_dim)
            self.null_class_id = num_classes
            # Small MLP to add non-linearity after embedding
            self.cond_post = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
        else:
            # Continuous property conditioning: properties → hidden_dim
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            # Learnable null condition token (used when condition is dropped)
            self.null_cond = nn.Parameter(torch.zeros(cond_dim))
            nn.init.normal_(self.null_cond, std=0.02)

        # Per-layer FiLM: hidden_dim → 2*hidden_dim (scale, shift)
        self.cond_film = nn.ModuleList([
            nn.Linear(hidden_dim, 2 * hidden_dim) for _ in range(num_layers)
        ])

        # α (guidance scale) embedding — lets the generator know the CFG strength
        self.alpha_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer layers (manual, not nn.TransformerEncoder, for FiLM injection)
        ff_dim = hidden_dim * mlp_ratio
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )
            # Post-FiLM LayerNorm
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Attention pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output head
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def _prepare_cond(
        self,
        cond: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Prepare condition embedding.

        For num_classes>0: cond is integer class IDs [B] or [B,1].
            Returns [B, hidden_dim] from class_embed + cond_post.
        For num_classes==0: cond is continuous [B, cond_dim].
            Returns [B, cond_dim] (run through cond_proj later in forward).
        """
        if self.num_classes > 0:
            # ── Discrete class conditioning ──
            if cond is None:
                class_ids = torch.full((batch_size,), self.null_class_id,
                                       device=device, dtype=torch.long)
            else:
                class_ids = cond.long().view(-1)   # [B]
                if self.training and self.p_uncond > 0:
                    drop = torch.rand(batch_size, device=device) < self.p_uncond
                    class_ids = torch.where(drop,
                                            torch.full_like(class_ids, self.null_class_id),
                                            class_ids)
            return self.cond_post(self.class_embed(class_ids))  # [B, hidden_dim]
        else:
            # ── Continuous conditioning ──
            if cond is None:
                return self.null_cond.unsqueeze(0).expand(batch_size, -1)
            if self.training and self.p_uncond > 0:
                drop_mask = torch.rand(batch_size, 1, device=device) < self.p_uncond
                null = self.null_cond.unsqueeze(0).expand(batch_size, -1)
                cond = torch.where(drop_mask, null, cond)
            return cond

    def forward(
        self,
        z: torch.Tensor,
        cond: torch.Tensor | None = None,
        alpha: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional condition and guidance scale α.

        Args:
            z: [B, noise_dim] noise vector
            cond: [B, cond_dim] property condition (or None for unconditional)
            alpha: scalar or [B] or [B,1] CFG guidance scale (or None)

        Returns: [B, latent_dim] generated latent vector
        """
        bsz = z.shape[0]

        # Prepare condition embedding
        c = self._prepare_cond(cond, bsz, z.device)
        if self.num_classes > 0:
            c_embed = c                   # already [B, hidden_dim] from class_embed
        else:
            c_embed = self.cond_proj(c)    # [B, cond_dim] → [B, hidden_dim]

        # Add α (guidance scale) embedding if provided
        if alpha is not None:
            if isinstance(alpha, (int, float)):
                alpha_t = torch.full((bsz, 1), float(alpha), device=z.device, dtype=z.dtype)
            else:
                alpha_t = alpha.view(-1, 1) if alpha.dim() == 1 else alpha
            c_embed = c_embed + self.alpha_proj(alpha_t)

        # Project noise to tokens
        h = self.input_proj(z)
        tokens = self.token_proj(h).view(bsz, self.num_tokens, -1)
        tokens = tokens + self.pos_embed

        if self.training and self.token_noise_std > 0:
            tokens = tokens + self.token_noise_std * torch.randn_like(tokens)

        # Transformer layers with FiLM conditioning
        for i, (layer, ln, film) in enumerate(
            zip(self.layers, self.layer_norms, self.cond_film)
        ):
            tokens = layer(tokens)
            # FiLM modulation: scale * x + shift
            film_params = film(c_embed)  # [B, 2*hidden_dim]
            scale, shift = film_params.chunk(2, dim=-1)  # each [B, hidden_dim]
            scale = scale.unsqueeze(1)   # [B, 1, hidden_dim]
            shift = shift.unsqueeze(1)   # [B, 1, hidden_dim]
            tokens = ln(tokens) * (1.0 + scale) + shift

        # Attention pooling
        query = self.pool_query.expand(bsz, -1, -1)
        pooled, _ = self.pool_attn(query, tokens, tokens, need_weights=False)
        return self.out(pooled.squeeze(1))

    @torch.no_grad()
    def sample_cfg(
        self,
        z: torch.Tensor,
        cond: torch.Tensor,
        alpha: float = 2.0,
    ) -> torch.Tensor:
        """
        CFG inference — single forward pass with (cond, α).

        In the new A.7 approach, the generator directly learns to modulate
        its output based on α during training (via drift field weighting).
        No two-pass interpolation needed.

        Args:
            z: [B, noise_dim] noise
            cond: [B, cond_dim] target properties
            alpha: CFG guidance scale (1.0 = unconditional, >1 = stronger guidance)

        Returns: [B, latent_dim]
        """
        self.eval()
        return self.forward(z, cond=cond, alpha=alpha)


# ── Factory ───────────────────────────────────────────────────────────

def build_latent_generator(cfg: dict, latent_dim: int) -> nn.Module:
    """Build a latent generator from config dict."""
    gc = cfg.get("generator", {})
    gen_type = str(gc.get("type", "mlp")).lower()
    noise_dim = int(gc.get("noise_dim", latent_dim))

    if gen_type in {"latent_dit_cfg", "dit_cfg"}:
        return LatentDiTGeneratorCFG(
            noise_dim=noise_dim,
            hidden_dim=int(gc.get("hidden_dim", 384)),
            latent_dim=latent_dim,
            num_layers=int(gc.get("num_layers", 8)),
            num_heads=int(gc.get("num_heads", 8)),
            num_tokens=int(gc.get("num_tokens", 16)),
            mlp_ratio=int(gc.get("mlp_ratio", 4)),
            dropout=float(gc.get("dropout", 0.1)),
            token_noise_std=float(gc.get("token_noise_std", 0.0)),
            cond_dim=int(gc.get("cond_dim", 4)),
            p_uncond=float(gc.get("p_uncond", 0.1)),
            num_classes=int(gc.get("num_classes", 0)),
        )

    if gen_type in {"latent_dit", "dit", "transformer"}:
        return LatentDiTGenerator(
            noise_dim=noise_dim,
            hidden_dim=int(gc.get("hidden_dim", 384)),
            latent_dim=latent_dim,
            num_layers=int(gc.get("num_layers", 8)),
            num_heads=int(gc.get("num_heads", 8)),
            num_tokens=int(gc.get("num_tokens", 16)),
            mlp_ratio=int(gc.get("mlp_ratio", 4)),
            dropout=float(gc.get("dropout", 0.1)),
            token_noise_std=float(gc.get("token_noise_std", 0.0)),
        )

    return LatentGenerator(
        noise_dim=noise_dim,
        hidden_dim=int(gc.get("hidden_dim", 256)),
        latent_dim=latent_dim,
        num_layers=int(gc.get("num_layers", 4)),
        dropout=float(gc.get("dropout", 0.0)),
    )


# ── LR Scheduling ── (Graph VAE load_vae / discretize_logits archived) ──

# ── LR Scheduling ────────────────────────────────────────────────────

def _warmup_cosine_lr(
    epoch: int,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.01,
) -> float:
    """Return LR multiplier ∈ [min_lr_ratio, 1.0] for given epoch (1-indexed).

    Phase 1: linear warmup from min_lr_ratio to 1.0 over warmup_epochs.
    Phase 2: cosine decay from 1.0 to min_lr_ratio over remaining epochs.
    """
    if epoch <= warmup_epochs:
        return min_lr_ratio + (1.0 - min_lr_ratio) * (epoch / max(warmup_epochs, 1))
    # Cosine decay
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    progress = min(progress, 1.0)
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))


def _warmup_linear_lr(
    epoch: int,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.01,
) -> float:
    """Warmup then linear decay to min_lr_ratio."""
    if epoch <= warmup_epochs:
        return min_lr_ratio + (1.0 - min_lr_ratio) * (epoch / max(warmup_epochs, 1))
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    progress = min(progress, 1.0)
    return 1.0 - (1.0 - min_lr_ratio) * progress


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 5,
    schedule: str = "cosine",
    min_lr_ratio: float = 0.01,
    last_epoch: int = 0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a warmup + decay LR scheduler.

    Args:
        optimizer:     The optimizer whose LR groups to schedule.
        total_epochs:  Total training epochs.
        warmup_epochs: Linear warmup phase length.
        schedule:      "cosine" | "linear" | "constant".
        min_lr_ratio:  Floor LR = base_lr × min_lr_ratio.
        last_epoch:    Epoch to resume from (0 = fresh start).

    Returns:
        LambdaLR scheduler.  Call scheduler.step() once per epoch
        (after optimizer.step()), WITHOUT passing an epoch argument.
    """
    if schedule == "cosine":
        fn = lambda e: _warmup_cosine_lr(e, warmup_epochs, total_epochs, min_lr_ratio)
    elif schedule == "linear":
        fn = lambda e: _warmup_linear_lr(e, warmup_epochs, total_epochs, min_lr_ratio)
    elif schedule == "constant":
        fn = lambda e: min_lr_ratio + (1.0 - min_lr_ratio) * min(e / max(warmup_epochs, 1), 1.0)
    else:
        raise ValueError(f"Unknown LR schedule: {schedule!r}. Choose from cosine/linear/constant.")

    # PyTorch LambdaLR uses last_epoch=-1 for fresh start.
    # When last_epoch > 0 (resume), we need initial_lr set in param_groups.
    pytorch_last_epoch = last_epoch - 1 if last_epoch > 0 else -1
    if last_epoch > 0:
        for pg in optimizer.param_groups:
            if "initial_lr" not in pg:
                pg["initial_lr"] = pg["lr"]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn, last_epoch=pytorch_last_epoch)


# ── Common Utilities ─────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) if p.suffix.lower() in {".yml", ".yaml"} else json.load(f)


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if any(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def _infer_num_properties(state_dict: dict[str, torch.Tensor]) -> int:
    candidates: list[int] = []
    for key, value in state_dict.items():
        if key.startswith("property_head") and hasattr(value, "shape") and getattr(value, "ndim", 0) == 2:
            candidates.append(int(value.shape[0]))
    return min(candidates) if candidates else 0


def discretize_logits(
    node_logits: torch.Tensor,
    edge_logits: torch.Tensor,
) -> list[dict[str, torch.Tensor]]:
    node_type = torch.argmax(node_logits, dim=-1)
    edge_type = torch.argmax(edge_logits, dim=-1)
    edge_type = torch.triu(edge_type, diagonal=1)
    edge_type = edge_type + edge_type.transpose(1, 2)
    graphs: list[dict[str, torch.Tensor]] = []
    for i in range(node_type.shape[0]):
        nt = node_type[i].detach().cpu()
        et = edge_type[i].detach().cpu()
        graphs.append(
            {
                "node_type": nt,
                "edge_type": et,
                "node_mask": nt != 0,
            }
        )
    return graphs


def load_vae(ckpt_path: str, device: torch.device | str) -> nn.Module:
    """Load an archived graph VAE checkpoint on the requested device."""
    graph_ae = _load_graph_module("driftingmol_graph_transformer_ae", "src_models/graph_transformer_ae.py")
    sys.modules.setdefault("src.models.graph_transformer_ae", graph_ae)
    graph_ae_v2 = _load_graph_module("driftingmol_graph_transformer_ae_v2", "src_models/graph_transformer_ae_v2.py")
    sys.modules.setdefault("src.models.graph_transformer_ae_v2", graph_ae_v2)

    GraphTransformerAE = graph_ae.GraphTransformerAE
    GraphTransformerAEConfig = graph_ae.GraphTransformerAEConfig
    GraphTransformerAE_V2 = graph_ae_v2.GraphTransformerAE_V2

    device = torch.device(device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    cfg = ckpt.get("cfg") or ckpt.get("config") or {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}

    state_dict = (
        ckpt.get("model_state")
        or ckpt.get("model_state_dict")
        or ckpt.get("state_dict")
        or {}
    )
    if not isinstance(state_dict, dict):
        state_dict = {}
    state_dict = _strip_module_prefix(state_dict)

    atom_types = tuple(data_cfg.get("atom_types", [1, 6, 7, 8, 9]))
    num_atom_classes = len(atom_types) + 1
    num_bond_classes = int(model_cfg.get("num_bond_classes", 5))
    max_nodes = int(data_cfg.get("max_nodes", 29))
    latent_dim = int(model_cfg.get("latent_dim", 64))
    num_properties = int(model_cfg.get("num_properties", _infer_num_properties(state_dict)))
    model_type = str(model_cfg.get("type", "graph_transformer_ae_v2")).lower()

    gt_cfg = GraphTransformerAEConfig(
        max_nodes=max_nodes,
        num_atom_classes=num_atom_classes,
        num_bond_classes=num_bond_classes,
        latent_dim=latent_dim,
        enc_hidden_dim=int(model_cfg.get("enc_hidden_dim", 128)),
        enc_num_layers=int(model_cfg.get("enc_num_layers", 3)),
        dec_hidden_dim=int(model_cfg.get("dec_hidden_dim", 128)),
        dec_edge_dim=int(model_cfg.get("dec_edge_dim", 64)),
        dec_num_layers=int(model_cfg.get("dec_num_layers", 4)),
        dec_n_head=int(model_cfg.get("dec_n_head", 4)),
        dec_ff_mult=int(model_cfg.get("dec_ff_mult", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )

    if model_type == "graph_transformer_ae_v2":
        model: nn.Module = GraphTransformerAE_V2(gt_cfg, num_properties=num_properties)
    elif model_type == "graph_transformer_ae":
        model = GraphTransformerAE(gt_cfg)
    else:
        raise ValueError(f"Unsupported graph VAE type: {model_type!r}")

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[load_vae] warning: missing={missing}, unexpected={unexpected}")

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    print(
        f"[load_vae] loaded {model_type} from {ckpt_path} "
        f"(latent_dim={latent_dim}, num_properties={num_properties})"
    )
    return model
