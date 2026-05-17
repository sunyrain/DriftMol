"""
One-shot SELFIES VAE for molecular generation.

Encoder: SELFIES token sequence → Transformer → μ, logvar → z
Decoder: z → Transformer (non-causal, one-shot) → token logits → argmax → SELFIES → always valid

Key advantage: ANY token sequence decoded from the output is a valid
SELFIES string, so validity is 100% by construction.  No repair or
constrained decoding is needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import selfies as sf
import torch
import torch.nn as nn
from torch import Tensor


# ── Vocabulary ────────────────────────────────────────────────────────

# Build QM9 vocabulary lazily (cached at module level)
_VOCAB: list[str] | None = None
_TOK2IDX: dict[str, int] | None = None
_IDX2TOK: dict[int, str] | None = None

PAD_TOKEN = "[PAD]"
PAD_IDX = 0


def build_vocab_from_smiles(smiles_list: list[str]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Build SELFIES vocabulary from a list of SMILES strings."""
    all_tokens: set[str] = set()
    for smi in smiles_list:
        sel = sf.encoder(smi)
        if sel is None:
            continue
        all_tokens.update(sf.split_selfies(sel))
    vocab = [PAD_TOKEN] + sorted(all_tokens)
    tok2idx = {t: i for i, t in enumerate(vocab)}
    idx2tok = {i: t for t, i in tok2idx.items()}
    return vocab, tok2idx, idx2tok


def get_vocab() -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Return cached vocabulary. Must call init_vocab() first."""
    global _VOCAB, _TOK2IDX, _IDX2TOK
    if _VOCAB is None:
        raise RuntimeError("Vocabulary not initialized. Call init_vocab() first.")
    return _VOCAB, _TOK2IDX, _IDX2TOK


def init_vocab(smiles_list: list[str]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Initialize and cache the SELFIES vocabulary."""
    global _VOCAB, _TOK2IDX, _IDX2TOK
    _VOCAB, _TOK2IDX, _IDX2TOK = build_vocab_from_smiles(smiles_list)
    return _VOCAB, _TOK2IDX, _IDX2TOK


def set_vocab(vocab: list[str]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Set vocabulary from a pre-built list (e.g. loaded from checkpoint)."""
    global _VOCAB, _TOK2IDX, _IDX2TOK
    _VOCAB = vocab
    _TOK2IDX = {t: i for i, t in enumerate(vocab)}
    _IDX2TOK = {i: t for t, i in _TOK2IDX.items()}
    return _VOCAB, _TOK2IDX, _IDX2TOK


# ── Tokenization ─────────────────────────────────────────────────────

def smiles_to_token_ids(smi: str, max_len: int) -> Tensor | None:
    """Convert SMILES → SELFIES → padded token ID tensor. Returns None on failure."""
    _, tok2idx, _ = get_vocab()
    sel = sf.encoder(smi)
    if sel is None:
        return None
    tokens = list(sf.split_selfies(sel))
    ids = [tok2idx.get(t, PAD_IDX) for t in tokens[:max_len]]
    ids += [PAD_IDX] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def token_ids_to_smiles(ids: Tensor) -> str:
    """Convert token ID tensor → SELFIES → SMILES. Always returns valid SMILES or empty string."""
    _, _, idx2tok = get_vocab()
    tokens = []
    for idx in ids.tolist():
        tok = idx2tok.get(idx, PAD_TOKEN)
        if tok == PAD_TOKEN:
            break
        tokens.append(tok)
    selfies_str = "".join(tokens)
    if not selfies_str:
        return ""
    try:
        return sf.decoder(selfies_str) or ""
    except Exception:
        return ""


def batch_token_ids_to_smiles(ids: Tensor) -> list[str]:
    """Convert batch of token ID tensors → list of SMILES."""
    return [token_ids_to_smiles(ids[i]) for i in range(ids.shape[0])]


# ── Config ────────────────────────────────────────────────────────────

@dataclass
class SelfiesVAEConfig:
    max_len: int = 20          # max SELFIES token length (QM9: 99%=16)
    vocab_size: int = 25       # will be set from data
    latent_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    # Training
    beta: float = 0.01         # KL weight (fixed, NOT warmup)
    num_properties: int = 0    # property prediction head
    decoder_noise_std: float = 0.0  # Gaussian noise added to z before decode (MoFlow-inspired)
    # Anti-posterior-collapse
    free_bits: float = 0.0     # Per-dim KL floor (Kingma 2016). 0 = disabled. Recommended: 0.25
    pos_dropout: float = 0.0   # Decoder position embedding dropout. Forces z-reliance. Recommended: 0.15
    dec_num_layers: int = 0    # Decoder layers (0 = same as num_layers). Use fewer for weaker decoder.
    pad_loss_weight: float = 0.5  # Weight for PAD prediction loss (teaches decoder when to stop)


# ── Model ─────────────────────────────────────────────────────────────

class SelfiesVAE(nn.Module):
    """One-shot SELFIES VAE with Transformer encoder/decoder."""

    def __init__(self, cfg: SelfiesVAEConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_dim

        # ── Encoder ──
        self.enc_embed = nn.Embedding(cfg.vocab_size, H, padding_idx=PAD_IDX)
        self.enc_pos = nn.Parameter(torch.randn(1, cfg.max_len, H) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            H, cfg.num_heads, H * cfg.ff_mult, cfg.dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, cfg.num_layers)
        # Attention pooling over sequence
        self.enc_pool = nn.Linear(H, 1)
        self.to_mu = nn.Linear(H, cfg.latent_dim)
        self.to_logvar = nn.Linear(H, cfg.latent_dim)

        # ── Decoder (one-shot, NON-causal) ──
        self.dec_z_proj = nn.Linear(cfg.latent_dim, H)
        self.dec_pos = nn.Parameter(torch.randn(1, cfg.max_len, H) * 0.02)
        self.pos_dropout = cfg.pos_dropout
        dec_n_layers = cfg.dec_num_layers if cfg.dec_num_layers > 0 else cfg.num_layers
        dec_layer = nn.TransformerEncoderLayer(
            H, cfg.num_heads, H * cfg.ff_mult, cfg.dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, dec_n_layers)
        self.out_head = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, cfg.vocab_size),
        )

        # ── Optional property head ──
        self.property_head: nn.Module | None = None
        if cfg.num_properties > 0:
            self.property_head = nn.Sequential(
                nn.Linear(cfg.latent_dim, H),
                nn.SiLU(),
                nn.LayerNorm(H),
                nn.Linear(H, H),
                nn.SiLU(),
                nn.LayerNorm(H),
                nn.Linear(H, cfg.num_properties),
            )

        self._init_weights()

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

    @property
    def latent_dim(self) -> int:
        return self.cfg.latent_dim

    # ── Encoder ──

    def encode(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Encode token sequences to latent distribution parameters.
        
        Args:
            token_ids: [B, L] int — padded SELFIES token indices (0 = PAD)
        Returns:
            mu: [B, latent_dim], logvar: [B, latent_dim]
        """
        pad_mask = (token_ids == PAD_IDX)  # [B, L], True = ignore
        h = self.enc_embed(token_ids) + self.enc_pos  # [B, L, H]
        h = self.encoder(h, src_key_padding_mask=pad_mask)  # [B, L, H]

        # Attention pooling (mask padding positions)
        pool_w = self.enc_pool(h).squeeze(-1)  # [B, L]
        pool_w = pool_w.masked_fill(pad_mask, float("-inf"))
        pool_w = pool_w.softmax(dim=1).unsqueeze(-1)  # [B, L, 1]
        pooled = (h * pool_w).sum(dim=1)  # [B, H]

        return self.to_mu(pooled), self.to_logvar(pooled)

    # ── Decoder ──

    def decode(self, z: Tensor) -> Tensor:
        """One-shot decode z to token logits.
        
        Args:
            z: [B, latent_dim]
        Returns:
            logits: [B, L, vocab_size]
        """
        z_proj = self.dec_z_proj(z).unsqueeze(1)  # [B, 1, H]
        pos = self.dec_pos  # [1, L, H]
        # Position dropout: randomly zero out some positions to force decoder to rely on z
        if self.training and self.pos_dropout > 0:
            mask = torch.rand(1, self.cfg.max_len, 1, device=z.device) > self.pos_dropout
            pos = pos * mask
        h = z_proj.expand(-1, self.cfg.max_len, -1) + pos  # [B, L, H]
        h = self.decoder(h)  # [B, L, H]
        return self.out_head(h)  # [B, L, V]

    # ── Reparameterize ──

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    # ── Forward ──

    def forward(self, token_ids: Tensor, beta: float | None = None
                ) -> dict[str, Tensor]:
        """Full forward pass: encode → reparameterize → decode.
        
        Args:
            token_ids: [B, L] padded SELFIES token indices
            beta: KL weight override (default: self.cfg.beta)
        Returns:
            dict with keys: loss, recon_loss, kl_loss, logits
        """
        if beta is None:
            beta = self.cfg.beta

        mu, logvar = self.encode(token_ids)
        z = self.reparameterize(mu, logvar)

        # Decoder noise injection (MoFlow-inspired: forces decoder to generalize)
        if self.training and self.cfg.decoder_noise_std > 0:
            z = z + torch.randn_like(z) * self.cfg.decoder_noise_std

        logits = self.decode(z)  # [B, L, V]

        # Reconstruction loss (ignore PAD positions for content tokens)
        recon_loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            token_ids.reshape(-1),
            ignore_index=PAD_IDX,
        )

        # PAD prediction loss: teach decoder to output PAD where input is PAD
        # This prevents the "tail generation" problem (predicting [C] in PAD region)
        pad_mask = (token_ids == PAD_IDX)  # [B, L]
        if pad_mask.any():
            pad_logits = logits[pad_mask]  # [N_pad, V]
            pad_targets = token_ids[pad_mask]  # [N_pad] all zeros (PAD_IDX)
            pad_loss = nn.functional.cross_entropy(pad_logits, pad_targets)
        else:
            pad_loss = torch.tensor(0.0, device=logits.device)

        # Combined: content + 0.5 * pad (lower weight since PAD positions are numerous)
        pad_weight = getattr(self.cfg, 'pad_loss_weight', 0.5)
        total_recon = recon_loss + pad_weight * pad_loss

        # KL divergence (with optional free bits)
        kl_per_dim = -0.5 * (1 + logvar - mu ** 2 - logvar.exp())  # [B, D]
        if self.cfg.free_bits > 0:
            # Per-dim KL floor: prevent individual dims from collapsing
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

        # Property prediction (from mu, not z — more stable)
        if self.property_head is not None:
            result["pred_props"] = self.property_head(mu)

        return result

    # ── Sampling utilities ──

    @torch.no_grad()
    def sample_smiles(self, z: Tensor, temperature: float = 0.0) -> list[str]:
        """Decode z vectors to SMILES strings.
        
        Args:
            z: [B, latent_dim]
            temperature: 0 = argmax, >0 = categorical sampling
        Returns:
            list of SMILES strings (always valid by SELFIES construction)
        """
        logits = self.decode(z)  # [B, L, V]
        if temperature > 0:
            token_ids = torch.distributions.Categorical(
                logits=logits / temperature
            ).sample()
        else:
            token_ids = logits.argmax(dim=-1)  # [B, L]
        return batch_token_ids_to_smiles(token_ids.cpu())

    @torch.no_grad()
    def sample_from_prior(self, num_samples: int, device: torch.device,
                          temperature: float = 0.0) -> list[str]:
        """Sample molecules from N(0, I) prior."""
        z = torch.randn(num_samples, self.cfg.latent_dim, device=device)
        return self.sample_smiles(z, temperature)
