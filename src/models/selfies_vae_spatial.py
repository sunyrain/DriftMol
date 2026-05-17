"""
Spatial SELFIES VAE — 2D spatial latent for drift-friendly gradients.

Key idea: latent z ∈ R^{C×H×W} is arranged as a 2D grid.
The decoder processes z through Conv2D ResBlocks before the Transformer,
enforcing spatial locality in the gradient ∂loss/∂z.

When H×W = max_len (e.g., 8×8 = 64), each spatial position maps 1:1
to a sequence token.  Drift at spatial position (i,j) primarily affects
nearby tokens → local modifications don't destroy the whole molecule.

External API identical to SelfiesVAE (encode/decode/forward all use flat z).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.selfies_vae import (
    PAD_IDX,
    batch_token_ids_to_smiles,
)


# ── Config ────────────────────────────────────────────────────────────

@dataclass
class SelfiesSpatialVAEConfig:
    # Sequence / vocabulary
    max_len: int = 64
    vocab_size: int = 108

    # Spatial latent: total dims = C × H × W
    latent_channels: int = 4       # C
    latent_height: int = 8         # H  (ideally H × W = max_len)
    latent_width: int = 8          # W

    # Transformer encoder
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1

    # Spatial conv decoder
    spatial_mid_channels: int = 64     # intermediate channels
    num_spatial_blocks: int = 4        # ResNet blocks

    # Transformer decoder
    dec_num_layers: int = 4

    # Training
    beta: float = 0.005
    num_properties: int = 0
    decoder_noise_std: float = 0.0

    # Anti-posterior-collapse
    free_bits: float = 0.25
    pos_dropout: float = 0.15
    pad_loss_weight: float = 0.5

    @property
    def latent_dim(self) -> int:
        """Total flat latent dimension (API-compatible with SelfiesVAEConfig)."""
        return self.latent_channels * self.latent_height * self.latent_width


# ── Building blocks ───────────────────────────────────────────────────

class SpatialResBlock(nn.Module):
    """Conv2D residual block: GN → GELU → Conv → GN → GELU → Conv + skip."""
    def __init__(self, channels: int):
        super().__init__()
        ng = min(8, channels)
        self.block = nn.Sequential(
            nn.GroupNorm(ng, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(ng, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


# ── Model ─────────────────────────────────────────────────────────────

class SelfiesSpatialVAE(nn.Module):
    """
    SELFIES VAE with 2D spatial latent.

    Encoder: tokens → Transformer → attn pool → flat μ/logvar.
    Decoder: flat z → reshape [C,H,W] → Conv2D ResBlocks → per-position
             project to Transformer hidden → Transformer → logits.
    """

    def __init__(self, cfg: SelfiesSpatialVAEConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_dim

        self.lat_c = cfg.latent_channels
        self.lat_h = cfg.latent_height
        self.lat_w = cfg.latent_width
        self._latent_dim_flat = cfg.latent_dim            # C * H * W

        # ── Transformer Encoder (same as SelfiesVAE) ──
        self.enc_embed = nn.Embedding(cfg.vocab_size, H, padding_idx=PAD_IDX)
        self.enc_pos = nn.Parameter(torch.randn(1, cfg.max_len, H) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            H, cfg.num_heads, H * cfg.ff_mult, cfg.dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, cfg.num_layers)
        self.enc_pool = nn.Linear(H, 1)

        # Encoder → flat latent (then reshaped to spatial in decoder)
        self.to_mu = nn.Linear(H, self._latent_dim_flat)
        self.to_logvar = nn.Linear(H, self._latent_dim_flat)

        # ── Spatial Conv Decoder ──
        mid_c = cfg.spatial_mid_channels

        self.dec_conv_in = nn.Sequential(
            nn.Conv2d(self.lat_c, mid_c, 3, padding=1),
            nn.GELU(),
        )
        self.dec_conv_blocks = nn.Sequential(
            *[SpatialResBlock(mid_c) for _ in range(cfg.num_spatial_blocks)]
        )
        self.dec_conv_out = nn.Sequential(
            nn.GroupNorm(min(8, mid_c), mid_c),
            nn.GELU(),
            nn.Conv2d(mid_c, mid_c, 3, padding=1),
        )

        # Spatial → sequence mapping
        n_spatial = self.lat_h * self.lat_w
        if n_spatial == cfg.max_len:
            # 1:1: each spatial position → one sequence position
            # [B, HW, mid_c] → Linear → [B, L, H]
            self.spatial_to_seq = nn.Linear(mid_c, H)
            self._spatial_mode = "direct"
        else:
            # General: flatten all spatial features → project to L × H
            self.spatial_to_seq = nn.Linear(n_spatial * mid_c, cfg.max_len * H)
            self._spatial_mode = "project"

        # ── Transformer Decoder ──
        self.dec_pos = nn.Parameter(torch.randn(1, cfg.max_len, H) * 0.02)
        self.pos_dropout = cfg.pos_dropout
        dec_layer = nn.TransformerEncoderLayer(
            H, cfg.num_heads, H * cfg.ff_mult, cfg.dropout,
            batch_first=True, norm_first=True,
        )
        self.dec_transformer = nn.TransformerEncoder(dec_layer, cfg.dec_num_layers)
        self.out_head = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, cfg.vocab_size),
        )

        # ── Optional property head ──
        self.property_head: nn.Module | None = None
        if cfg.num_properties > 0:
            self.property_head = nn.Sequential(
                nn.Linear(self._latent_dim_flat, H),
                nn.SiLU(),
                nn.LayerNorm(H),
                nn.Linear(H, H),
                nn.SiLU(),
                nn.LayerNorm(H),
                nn.Linear(H, cfg.num_properties),
            )

        self._init_weights()

    # ── Init ──────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim_flat

    # ── Encoder ───────────────────────────────────────────────────

    def encode(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            token_ids: [B, L] int
        Returns:
            mu: [B, C*H*W] flat, logvar: [B, C*H*W] flat
        """
        pad_mask = (token_ids == PAD_IDX)
        h = self.enc_embed(token_ids) + self.enc_pos
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        # Attention pooling
        pool_w = self.enc_pool(h).squeeze(-1)
        pool_w = pool_w.masked_fill(pad_mask, float("-inf"))
        pool_w = pool_w.softmax(dim=1).unsqueeze(-1)
        pooled = (h * pool_w).sum(dim=1)

        return self.to_mu(pooled), self.to_logvar(pooled)

    # ── Decoder ───────────────────────────────────────────────────

    def decode(self, z: Tensor) -> Tensor:
        """
        Args:
            z: [B, C*H*W] flat
        Returns:
            logits: [B, L, vocab_size]
        """
        B = z.shape[0]
        Hdim = self.cfg.hidden_dim

        # ── Spatial Conv processing ──
        z_spatial = z.view(B, self.lat_c, self.lat_h, self.lat_w)
        h = self.dec_conv_in(z_spatial)          # [B, mid_c, Hlat, Wlat]
        h = self.dec_conv_blocks(h)              # [B, mid_c, Hlat, Wlat]
        h = self.dec_conv_out(h)                 # [B, mid_c, Hlat, Wlat]

        # ── Spatial → sequence ──
        if self._spatial_mode == "direct":
            # [B, mid_c, Hlat, Wlat] → [B, HW, mid_c]
            h_seq = h.flatten(2).permute(0, 2, 1)
            h_seq = self.spatial_to_seq(h_seq)   # [B, L, H]
        else:
            h_flat = h.reshape(B, -1)
            h_seq = self.spatial_to_seq(h_flat).view(B, self.cfg.max_len, Hdim)

        # ── Transformer decoder ──
        pos = self.dec_pos
        if self.training and self.pos_dropout > 0:
            mask = torch.rand(1, self.cfg.max_len, 1, device=z.device) > self.pos_dropout
            pos = pos * mask
        h_seq = h_seq + pos

        h_seq = self.dec_transformer(h_seq)
        return self.out_head(h_seq)

    # ── Reparameterize ────────────────────────────────────────────

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    # ── Forward ───────────────────────────────────────────────────

    def forward(
        self, token_ids: Tensor, beta: float | None = None
    ) -> dict[str, Tensor]:
        if beta is None:
            beta = self.cfg.beta

        mu, logvar = self.encode(token_ids)
        z = self.reparameterize(mu, logvar)

        if self.training and self.cfg.decoder_noise_std > 0:
            z = z + torch.randn_like(z) * self.cfg.decoder_noise_std

        logits = self.decode(z)

        # Reconstruction loss (content tokens)
        recon_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            token_ids.reshape(-1),
            ignore_index=PAD_IDX,
        )

        # PAD prediction loss
        pad_mask = (token_ids == PAD_IDX)
        if pad_mask.any():
            pad_loss = F.cross_entropy(logits[pad_mask], token_ids[pad_mask])
        else:
            pad_loss = torch.tensor(0.0, device=logits.device)

        total_recon = recon_loss + self.cfg.pad_loss_weight * pad_loss

        # KL divergence with free bits
        kl_per_dim = -0.5 * (1 + logvar - mu ** 2 - logvar.exp())
        if self.cfg.free_bits > 0:
            kl_per_dim = kl_per_dim.clamp(min=self.cfg.free_bits)
        kl_loss = kl_per_dim.sum(-1).mean()

        result = {
            "loss": total_recon + beta * kl_loss,
            "recon_loss": recon_loss,
            "pad_loss": pad_loss,
            "kl_loss": kl_loss,
            "logits": logits,
            "mu": mu,
            "logvar": logvar,
        }

        if self.property_head is not None:
            result["pred_props"] = self.property_head(mu)

        return result

    # ── Sampling ──────────────────────────────────────────────────

    @torch.no_grad()
    def sample_smiles(self, z: Tensor, temperature: float = 0.0) -> list[str]:
        logits = self.decode(z)
        if temperature > 0:
            token_ids = torch.distributions.Categorical(
                logits=logits / temperature
            ).sample()
        else:
            token_ids = logits.argmax(dim=-1)
        return batch_token_ids_to_smiles(token_ids.cpu())

    @torch.no_grad()
    def sample_from_prior(
        self, num_samples: int, device: torch.device, temperature: float = 0.0
    ) -> list[str]:
        z = torch.randn(num_samples, self._latent_dim_flat, device=device)
        return self.sample_smiles(z, temperature)
