"""
φ-space Drifting Loss with V+ (attraction) and V- (repulsion).

Implements the core Drifting Models objective in φ-space:
  V+ = attraction toward real data distribution
  V- = repulsion from other generated samples (prevents mode collapse)
  V  = V+ - V-

Loss = ||φ(x) - stopgrad(φ(x) + V(φ(x)))||²

Gradient flows: loss → φ(x) → φ (frozen) → x (generator output) → generator θ

All operations are in φ-space (unit sphere if φ has normalize_output=True).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Decoder-output feature extraction ─────────────────────────────────

def extract_decoder_features(
    vae: nn.Module,
    z: torch.Tensor,                                    # [B, latent_dim]
    triu_idx: tuple[torch.Tensor, torch.Tensor],        # pre-computed upper-tri indices
) -> torch.Tensor:
    """
    Extract soft decoder features: softmax(logits) → flat vector [B, 2204].

    Feature = concat(
        softmax(node_logits).flatten,      # [B, 29*6]  = [B, 174]
        softmax(edge_upper_tri_logits).flatten  # [B, 406*5] = [B, 2030]
    )
    Total dim = 174 + 2030 = 2204.

    Gradient flows: loss → feat → softmax → logits → Decoder → z → Generator θ.
    The VAE decoder is frozen (parameters don't update) but gradients pass through
    its forward pass to affect z, which is the generator's output.

    Args:
        vae:      Frozen VAE with .decode(z) → (node_logits, edge_logits)
        z:        [B, latent_dim] generator output (requires_grad for training)
        triu_idx: (row_idx, col_idx) from torch.triu_indices(max_nodes, max_nodes, offset=1)

    Returns:
        feat: [B, 2204] soft decoder features
    """
    node_logits, edge_logits = vae.decode(z)  # [B,29,6], [B,29,29,5]
    node_probs = torch.softmax(node_logits, dim=-1)           # [B, 29, 6]
    edge_probs = torch.softmax(
        edge_logits[:, triu_idx[0], triu_idx[1], :], dim=-1   # [B, 406, 5]
    )
    B = z.shape[0]
    feat = torch.cat([
        node_probs.reshape(B, -1),   # [B, 174]
        edge_probs.reshape(B, -1),   # [B, 2030]
    ], dim=-1)                       # [B, 2204]
    return feat


def _rbf_kernel_weights(
    query: torch.Tensor,     # [G, D]
    keys: torch.Tensor,      # [P, D]
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute RBF kernel weights with adaptive bandwidth (median heuristic).

    Returns:
        w: [G, P] kernel weights, normalized via softmax per query.
    """
    # Pairwise squared distances [G, P]
    sq_dist = torch.cdist(query, keys, p=2).pow(2)

    # Adaptive bandwidth: median squared distance
    sigma2 = sq_dist.median().clamp(min=1e-6) * temperature

    # Softmax normalization for numerical stability
    log_w = -sq_dist / (2.0 * sigma2)
    w = torch.softmax(log_w, dim=1)  # [G, P], sums to 1 per row
    return w


@torch.no_grad()
def compute_drift_field(
    phi_gen: torch.Tensor,        # [G, D] φ(x_gen), generated embeddings
    phi_data: torch.Tensor,       # [P, D] φ(y+), real data embeddings
    temperature: float = 1.0,
    repulsion_weight: float = 1.0,  # λ_rep: strength of inter-gen repulsion
) -> torch.Tensor:
    """
    Compute the full drift field V = V+ - V-.

    V+(φ(x_i)) = Σ_j w+_ij · (φ(y+_j) - φ(x_i))    [attraction to data]
    V-(φ(x_i)) = Σ_{k≠i} w-_ik · (φ(x_k) - φ(x_i))  [repulsion from others]

    Args:
        phi_gen:          [G, D] generated sample embeddings (detached for target computation)
        phi_data:         [P, D] real data embeddings
        temperature:      kernel bandwidth multiplier
        repulsion_weight: relative strength of V- vs V+

    Returns:
        V: [G, D] drift vectors
    """
    G, D = phi_gen.shape

    # ── V+: attraction to data ──
    w_pos = _rbf_kernel_weights(phi_gen, phi_data, temperature)  # [G, P]
    target_pos = w_pos @ phi_data  # [G, D] weighted mean of data
    V_pos = target_pos - phi_gen   # [G, D] direction toward data

    # ── V-: repulsion from other generated samples ──
    if repulsion_weight > 0 and G > 1:
        # Self-distances [G, G]
        sq_dist_gen = torch.cdist(phi_gen, phi_gen, p=2).pow(2)
        sigma2_gen = sq_dist_gen.median().clamp(min=1e-6) * temperature

        # Mask diagonal (don't repel from self)
        log_w_neg = -sq_dist_gen / (2.0 * sigma2_gen)
        log_w_neg.fill_diagonal_(float("-inf"))  # exclude self
        w_neg = torch.softmax(log_w_neg, dim=1)  # [G, G]

        target_neg = w_neg @ phi_gen  # [G, D] weighted mean of neighbors
        V_neg = target_neg - phi_gen  # [G, D] direction toward neighbors

        V = V_pos - repulsion_weight * V_neg
    else:
        V = V_pos

    return V


def phi_drift_loss_v4(
    phi_gen: torch.Tensor,        # [G, D] φ(x_gen), WITH gradient
    phi_data: torch.Tensor,       # [P, D] φ(y+), no gradient needed
    temperature: float = 1.0,
    repulsion_weight: float = 1.0,
) -> torch.Tensor:
    """
    Full φ-space drifting loss with attraction + repulsion.

    L = E[||φ(x) - stopgrad(φ(x) + V(φ(x)))||²]

    Gradient flows through φ(x) = φ(f_θ(ε)), updating generator θ.

    Args:
        phi_gen:          [G, D] generated embeddings (requires grad)
        phi_data:         [P, D] data embeddings (detached internally)
        temperature:      kernel bandwidth
        repulsion_weight: λ_rep

    Returns:
        scalar loss (mean of squared L2 distances over batch)
    """
    phi_data_sg = phi_data.detach()

    # Compute drift target with stopgrad
    with torch.no_grad():
        V = compute_drift_field(
            phi_gen.detach(), phi_data_sg,
            temperature=temperature,
            repulsion_weight=repulsion_weight,
        )
        target = phi_gen.detach() + V  # [G, D] stopgrad target

    # MSE loss: mean over both D and G
    diff = phi_gen - target
    loss = (diff * diff).mean()
    return loss


# ── V4.1: Batch-Normalized Kernel Drift (ported from v1) ─────────────

@torch.no_grad()
def compute_drift_field_bn(
    phi_gen: torch.Tensor,        # [G, D] φ(x_gen)
    phi_data: torch.Tensor,       # [P, D] φ(y+)
    temperature: float = 5.0,
) -> torch.Tensor:
    """
    Drift field using v1-style batch-normalized kernel.

    Concatenates gen and data into a single target set, then computes:
      - V+ = pull toward data (weighted by batch-normalized kernel)
      - V- = push from other generated (weighted by batch-normalized kernel)

    The batch-normalized kernel automatically balances V+/V- based on
    relative densities, eliminating the need for a repulsion_weight parameter.

    Args:
        phi_gen:     [G, D] generated sample embeddings
        phi_data:    [P, D] real data embeddings
        temperature: RBF kernel bandwidth (use L1 distance, not squared)

    Returns:
        V: [G, D] drift vectors
    """
    G = phi_gen.shape[0]
    targets = torch.cat([phi_gen, phi_data], dim=0)  # [G+P, D]

    # L2 distances (NOT squared, matching v1)
    dist = torch.cdist(phi_gen, targets, p=2)  # [G, G+P]
    dist[:, :G].fill_diagonal_(1e6)  # mask self

    # RBF kernel
    kernel = (-dist / temperature).exp()  # [G, G+P]

    # Batch-normalised kernel (doubly stochastic approximation)
    norm_row = kernel.sum(dim=-1, keepdim=True).clamp_min(1e-12)     # [G, 1]
    norm_col = kernel.sum(dim=-2, keepdim=True).clamp_min(1e-12)     # [1, G+P]
    normalizer = (norm_row * norm_col).sqrt()
    nk = kernel / normalizer  # [G, G+P]

    # V+: pull toward data  (columns G: onward)
    pos_coeff = nk[:, G:] * nk[:, :G].sum(dim=-1, keepdim=True)     # [G, P]
    pos_V = pos_coeff @ targets[G:]                                   # [G, D]

    # V-: push from other generated  (columns :G)
    neg_coeff = nk[:, :G] * nk[:, G:].sum(dim=-1, keepdim=True)     # [G, G]
    neg_V = neg_coeff @ targets[:G]                                   # [G, D]

    return pos_V - neg_V


def phi_drift_loss_v4_bn(
    phi_gen: torch.Tensor,        # [G, D] φ(x_gen), WITH gradient
    phi_data: torch.Tensor,       # [P, D] φ(y+), no gradient needed
    temperature: float = 5.0,
) -> torch.Tensor:
    """
    φ-space drifting loss using batch-normalized kernel (v1-style).

    More robust than separate-kernel approach:
      - V+/V- balance is automatic via kernel normalization
      - No repulsion_weight parameter needed
      - Temperature controls kernel bandwidth directly

    L = ||φ(x) - stopgrad(φ(x) + V_bn(φ(x)))||²

    Args:
        phi_gen:     [G, D] generated embeddings (requires grad)
        phi_data:    [P, D] data embeddings
        temperature: kernel bandwidth

    Returns:
        scalar loss
    """
    phi_data_sg = phi_data.detach()

    with torch.no_grad():
        V = compute_drift_field_bn(
            phi_gen.detach(), phi_data_sg,
            temperature=temperature,
        )
        target = phi_gen.detach() + V

    diff = phi_gen - target
    loss = (diff * diff).mean()
    return loss


# ── Z-space Diversity Loss ───────────────────────────────────────────

def z_space_repulsion_loss(
    z_gen: torch.Tensor,          # [G, D] generated latent vectors
    margin: float = 1.0,
    top_k: int = 5,
) -> torch.Tensor:
    """
    Explicit z-space pairwise repulsion loss.

    Penalizes generated z-vectors that are too close to each other.
    Uses top-K nearest neighbors to focus on the most problematic cases.

    L = mean(ReLU(margin - dist_to_kNN))

    This directly prevents z-space clustering, which is the root cause
    of low uniqueness (nearby z's decode to the same SMILES via VAE).

    Args:
        z_gen:  [G, D] generated latent vectors
        margin: minimum desired pairwise distance
        top_k:  number of nearest neighbors to consider

    Returns:
        scalar repulsion loss
    """
    G = z_gen.shape[0]
    if G < 2:
        return torch.zeros((), device=z_gen.device)

    # Pairwise L2 distances
    dists = torch.cdist(z_gen, z_gen, p=2)  # [G, G]

    # Mask diagonal with large value (no in-place op for gradient flow)
    mask = torch.eye(G, dtype=torch.bool, device=z_gen.device)
    dists = dists.masked_fill(mask, float("inf"))

    # Top-K nearest neighbor distances
    k = min(top_k, G - 1)
    topk_dists = dists.topk(k, dim=1, largest=False).values  # [G, K]

    # Hinge loss: penalize if distance < margin
    loss = F.relu(margin - topk_dists).mean()
    return loss


def phi_space_repulsion_loss(
    phi_gen: torch.Tensor,          # [G, D] φ-embedded generated vectors
    margin: float = 5.0,
    top_k: int = 5,
) -> torch.Tensor:
    """
    φ-space pairwise repulsion loss.

    Same logic as z_space_repulsion_loss but operates in φ-space.
    Critical because drift loss operates in φ-space: z-space diversity
    does NOT prevent φ-space collapse (φ is non-linear).

    L = mean(ReLU(margin - dist_to_kNN_in_phi_space))

    Args:
        phi_gen:  [G, D] φ-embedded generated vectors
        margin: minimum desired pairwise distance in φ-space
        top_k:  number of nearest neighbors to consider

    Returns:
        scalar repulsion loss
    """
    G = phi_gen.shape[0]
    if G < 2:
        return torch.zeros((), device=phi_gen.device)

    dists = torch.cdist(phi_gen, phi_gen, p=2)  # [G, G]
    mask = torch.eye(G, dtype=torch.bool, device=phi_gen.device)
    dists = dists.masked_fill(mask, float("inf"))

    k = min(top_k, G - 1)
    topk_dists = dists.topk(k, dim=1, largest=False).values  # [G, K]

    loss = F.relu(margin - topk_dists).mean()
    return loss


# ── Multi-Temperature Drift (Paper Section A.6) ─────────────────────

@torch.no_grad()
def compute_multi_temp_drift(
    phi_gen: torch.Tensor,
    phi_pos: torch.Tensor,
    temperatures: list[float] | None = None,
    phi_unc: torch.Tensor | None = None,
    cfg_w: float = 0.0,
    normalize_drift: bool = True,
    normalize_distances: bool = True,
    fixed_lambdas: dict[float, float] | None = None,
    norm_mode: str = "xy",
    knn_restrict_k: int | None = None,
    attraction_scale: float = 1.0,
    repulsion_scale: float = 1.0,
) -> torch.Tensor:
    """Aggregate drift vectors across multiple temperatures.

    Args:
        fixed_lambdas: If provided, maps τ -> λ_τ (pre-computed from data).
            This avoids the per-batch λ_τ that kills the loss signal.
            When normalize_drift=True and fixed_lambdas is given, use the fixed values.
    """
    if temperatures is None:
        temperatures = [0.02, 0.05, 0.2]

    D = phi_gen.shape[-1]
    V_total = torch.zeros_like(phi_gen)

    for tau in temperatures:
        V_tau = compute_drift_field_paper(
            phi_gen, phi_pos, temperature=tau,
            phi_unc=phi_unc, cfg_w=cfg_w,
            normalize_distances=normalize_distances,
            norm_mode=norm_mode,
            knn_restrict_k=knn_restrict_k,
            attraction_scale=attraction_scale,
            repulsion_scale=repulsion_scale,
        )
        if normalize_drift:
            if fixed_lambdas is not None and tau in fixed_lambdas:
                lambda_tau = max(fixed_lambdas[tau], 1e-8)
            else:
                V_norm_sq_per_D = V_tau.pow(2).sum(dim=-1).mean() / D
                lambda_tau = V_norm_sq_per_D.sqrt().clamp(min=1e-8)
            V_tilde = V_tau / lambda_tau
        else:
            V_tilde = V_tau
        V_total = V_total + V_tilde

    return V_total

def multi_temp_drift_loss(
    phi_gen: torch.Tensor,        # [N, D] WITH gradient
    phi_pos: torch.Tensor,        # [Npos, D] positive data
    temperatures: list[float] | None = None,
    phi_unc: torch.Tensor | None = None,
    cfg_w: float = 0.0,
    normalize_drift: bool = True,
    normalize_distances: bool = True,
    fixed_lambdas: dict[float, float] | None = None,
    norm_mode: str = "xy",
    knn_restrict_k: int | None = None,
    attraction_scale: float = 1.0,
    repulsion_scale: float = 1.0,
) -> torch.Tensor:
    """
    Multi-temperature drifting loss (Section A.6).

    L = ||φ(x) - stopgrad(φ(x) + V_agg(φ(x)))||²

    Args:
        fixed_lambdas: If provided, maps τ -> λ_τ (pre-computed from data).
            This keeps the normalization scale fixed so that loss properly
            decreases as q → p (like the paper's Figure 4).
    """
    phi_pos_sg = phi_pos.detach()
    phi_unc_sg = phi_unc.detach() if phi_unc is not None else None

    with torch.no_grad():
        V = compute_multi_temp_drift(
            phi_gen.detach(), phi_pos_sg,
            temperatures=temperatures,
            phi_unc=phi_unc_sg, cfg_w=cfg_w,
            normalize_drift=normalize_drift,
            normalize_distances=normalize_distances,
            fixed_lambdas=fixed_lambdas,
            norm_mode=norm_mode,
            knn_restrict_k=knn_restrict_k,
            attraction_scale=attraction_scale,
            repulsion_scale=repulsion_scale,
        )
        target = phi_gen.detach() + V

    diff = phi_gen - target
    loss = (diff * diff).mean()
    return loss


def sample_cfg_alpha(
    power: float = 3.0,
    alpha_min: float = 1.0,
    alpha_max: float = 4.0,
) -> float:
    """
    Sample α from p(α) ∝ α^{-power} on [alpha_min, alpha_max].

    Paper Table 8: ablation default uses power=3, final uses power=5.
    Inverse CDF sampling.
    """
    u = torch.rand(1).item()
    k = power
    a_min_k = alpha_min ** (1 - k)
    a_max_k = alpha_max ** (1 - k)
    alpha = (a_min_k + u * (a_max_k - a_min_k)) ** (1.0 / (1.0 - k))
    return float(alpha)


def decoupled_phi_drift_loss(
    z_gen: torch.Tensor,        # [N, D_z] generated z WITH gradient
    z_pos: torch.Tensor,        # [Npos, D_z] positive real data z (detached)
    phi_gen: torch.Tensor,      # [N, D_phi] generated φ features (will be detached)
    phi_pos: torch.Tensor,      # [Npos, D_phi] positive φ features (will be detached)
    temperatures: list[float] | None = None,
    normalize_distances: bool = True,
    norm_mode: str = "y_nocross",
    knn_restrict_k: int | None = None,
) -> torch.Tensor:
    """
    Decoupled drift loss: φ oracle weights + z-space gradient.

    φ determines "which real samples are similar" via soft attention weights,
    but the actual displacement and gradient are computed purely in z-space.
    This avoids Jacobian distortion from backpropagating through φ.

    For each temperature τ:
      1. Compute attention A from φ distances (stop-gradient)
      2. V_pos = Σ_j A⁺_ij (z_j - z_i)   ← pull toward φ-similar real data
      3. V_neg = Σ_k A⁻_ik (z_k - z_i)   ← push from generated neighbors
      4. V = V_pos - V_neg
    """
    if temperatures is None:
        temperatures = [0.02, 0.05, 0.2]

    N, D_z = z_gen.shape
    Npos = z_pos.shape[0]
    z_pos_d = z_pos.detach()

    # All φ computations are stop-gradient
    phi_g = phi_gen.detach()
    phi_p = phi_pos.detach()

    V_total = torch.zeros_like(z_gen)  # [N, D_z]

    for tau in temperatures:
        # Compute φ-space distances for attention weights
        dist_pos_phi = torch.cdist(phi_g, phi_p, p=2)  # [N, Npos]
        dist_neg_phi = torch.cdist(phi_g, phi_g, p=2)  # [N, N]

        # Global distance normalization in φ-space
        if normalize_distances:
            eye_N = torch.eye(N, dtype=torch.bool, device=z_gen.device)
            d_global = torch.cat([
                dist_pos_phi.flatten(),
                dist_neg_phi[~eye_N],
            ]).mean().clamp(min=1e-6)
            dist_pos_phi = dist_pos_phi / d_global
            dist_neg_phi = dist_neg_phi / d_global

        dist_neg_phi_masked = dist_neg_phi.clone()
        dist_neg_phi_masked.fill_diagonal_(1e6)

        logit_pos = -dist_pos_phi / tau  # [N, Npos]

        # kNN restriction in φ-space: only attend to k-nearest positives
        if knn_restrict_k is not None and knn_restrict_k < Npos:
            _, topk_idx = dist_pos_phi.topk(knn_restrict_k, dim=1, largest=False)
            knn_mask = torch.ones_like(logit_pos, dtype=torch.bool)
            knn_mask.scatter_(1, topk_idx, False)
            logit_pos = logit_pos.masked_fill(knn_mask, float('-inf'))

        logit_neg = -dist_neg_phi_masked / tau  # [N, N]

        # Joint softmax for attention weights (stop-gradient)
        logit_all = torch.cat([logit_pos, logit_neg], dim=1)  # [N, Npos+N]

        if norm_mode in ("y_nocross", "y"):
            A = logit_all.softmax(dim=1)  # row-wise softmax
        else:
            A_row = logit_all.softmax(dim=1)
            A_col = logit_all.softmax(dim=0)
            A = (A_row * A_col).sqrt()

        A_pos = A[:, :Npos]              # [N, Npos]
        A_neg = A[:, Npos:]              # [N, N]

        # Z-space displacements (gradient flows here)
        diff_pos = z_pos_d.unsqueeze(0) - z_gen.unsqueeze(1)  # [N, Npos, D_z]
        diff_neg = z_gen.detach().unsqueeze(0) - z_gen.unsqueeze(1)  # [N, N, D_z]

        V_pos = (A_pos.unsqueeze(-1) * diff_pos).sum(dim=1)  # [N, D_z]
        V_neg = (A_neg.unsqueeze(-1) * diff_neg).sum(dim=1)  # [N, D_z]

        V_tau = V_pos - V_neg
        V_total = V_total + V_tau

    # Regression loss: push z_gen toward z_gen + V
    target = (z_gen + V_total).detach()
    loss = (z_gen - target).pow(2).mean()
    return loss


def multiscale_decoupled_drift_loss(
    z_gen: torch.Tensor,                    # [N, D_z] generated z WITH gradient
    z_pos: torch.Tensor,                    # [Npos, D_z] positive real data z
    phi_gen_scales: list[torch.Tensor],     # list of [N, D_s] per-scale φ(z_gen)
    phi_pos_scales: list[torch.Tensor],     # list of [Npos, D_s] per-scale φ(z_pos)
    scale_temperatures: list[float],        # one τ per scale, matched to dimensionality
    normalize_distances: bool = True,
    norm_mode: str = "y_nocross",
    knn_restrict_k: int | None = None,
) -> torch.Tensor:
    """
    Multi-scale decoupled drift: each φ scale uses its matched temperature.

    Coarse scale (8D) + large τ → captures global distribution shape
    Medium scale (16D) + medium τ → captures regional structure
    Fine scale (32D) + small τ → captures local neighborhood precision

    All scales produce z-space displacement; sum into unified V.
    """
    assert len(phi_gen_scales) == len(phi_pos_scales) == len(scale_temperatures)

    N, D_z = z_gen.shape
    Npos = z_pos.shape[0]
    z_pos_d = z_pos.detach()

    V_total = torch.zeros_like(z_gen)  # [N, D_z]

    for phi_g_s, phi_p_s, tau in zip(phi_gen_scales, phi_pos_scales, scale_temperatures):
        phi_g = phi_g_s.detach()
        phi_p = phi_p_s.detach()

        dist_pos = torch.cdist(phi_g, phi_p, p=2)  # [N, Npos]
        dist_neg = torch.cdist(phi_g, phi_g, p=2)  # [N, N]

        if normalize_distances:
            eye_N = torch.eye(N, dtype=torch.bool, device=z_gen.device)
            d_global = torch.cat([
                dist_pos.flatten(),
                dist_neg[~eye_N],
            ]).mean().clamp(min=1e-6)
            dist_pos = dist_pos / d_global
            dist_neg = dist_neg / d_global

        dist_neg_masked = dist_neg.clone()
        dist_neg_masked.fill_diagonal_(1e6)

        logit_pos = -dist_pos / tau

        # kNN restriction in φ-space: only attend to k-nearest positives
        if knn_restrict_k is not None and knn_restrict_k < Npos:
            _, topk_idx = dist_pos.topk(knn_restrict_k, dim=1, largest=False)
            knn_mask = torch.ones_like(logit_pos, dtype=torch.bool)
            knn_mask.scatter_(1, topk_idx, False)
            logit_pos = logit_pos.masked_fill(knn_mask, float('-inf'))

        logit_neg = -dist_neg_masked / tau
        logit_all = torch.cat([logit_pos, logit_neg], dim=1)

        if norm_mode in ("y_nocross", "y"):
            A = logit_all.softmax(dim=1)
        else:
            A_row = logit_all.softmax(dim=1)
            A_col = logit_all.softmax(dim=0)
            A = (A_row * A_col).sqrt()

        A_pos = A[:, :Npos]
        A_neg = A[:, Npos:]

        diff_pos = z_pos_d.unsqueeze(0) - z_gen.unsqueeze(1)
        diff_neg = z_gen.detach().unsqueeze(0) - z_gen.unsqueeze(1)

        V_pos = (A_pos.unsqueeze(-1) * diff_pos).sum(dim=1)
        V_neg = (A_neg.unsqueeze(-1) * diff_neg).sum(dim=1)

        V_total = V_total + (V_pos - V_neg)

    target = (z_gen + V_total).detach()
    loss = (z_gen - target).pow(2).mean()
    return loss


# ── CFG-aware Drift Field (Paper Algorithm 2 + Section A.7) ──────────

@torch.no_grad()
def compute_drift_field_paper(
    phi_gen: torch.Tensor,        # [N, D] generated embeddings (= y_neg)
    phi_pos: torch.Tensor,        # [Npos, D] condition-matched positive data
    temperature: float = 0.05,
    phi_unc: torch.Tensor | None = None,  # [Nunc, D] unconditional negatives (CFG)
    cfg_w: float = 0.0,           # CFG weight for unconditional negatives
    normalize_distances: bool = True,  # global distance normalization (Paper A.6)
    pos_weights: torch.Tensor | None = None,  # [Npos] importance weights for positives
    return_diagnostics: bool = False,  # return kernel diagnostics dict
    norm_mode: str = "xy",        # normalization variant for ablation
    knn_restrict_k: int | None = None,  # if set, restrict V⁺ to k-nearest positives per sample
    attraction_scale: float = 1.0, # destructive ablation scale for V+
    repulsion_scale: float = 1.0,  # destructive ablation scale for V-
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """
    Compute drift field V following Paper Algorithm 2 faithfully.

    Key differences from our old implementation:
      1. Joint softmax over concatenated (pos, neg) logits
      2. Bi-dimensional softmax (over both x-axis and y-axis), then sqrt(row * col)
      3. Cross-multiplication: W_pos *= A_neg.sum(), W_neg *= A_pos.sum()
      4. Kernel uses L2 distance (not squared): k = exp(-||x-y|| / τ)
      5. Generated samples themselves serve as negatives (y_neg = x)
      6. Global distance normalization: distances / d_global before kernel (Paper A.6)
      7. Optional pos_weights: π_i importance weights added as log(π_i) to positive logits
         This implements weighted V⁺ for continuous condition p(x|s₀).

    For CFG (A.7): unconditional negatives are added with weight w to kernel logits.
    The w also participates in global distance normalization.

    Args:
        phi_gen:              [N, D] generated embeddings (also serve as negatives)
        phi_pos:              [Npos, D] condition-matched positive data
        temperature:          kernel temperature τ
        phi_unc:              [Nunc, D] unconditional negatives for CFG (optional)
        cfg_w:                weight for unconditional negatives
                              (= max(0, (α-1)(N-1)/Nunc)); if zero, the
                              unconditional branch is omitted instead of
                              evaluating log(0)
        normalize_distances:  if True, normalize all distances by global mean (Paper A.6)
        pos_weights:          [Npos] importance weights π_i for soft-conditioning
        return_diagnostics:   if True, also return dict with kernel entropy etc.
        norm_mode:            Normalization variant (Paper Table 5 ablation):
                              "xy"    — default: A = sqrt(softmax_row * softmax_col), with cross-mult
                              "y"     — y-only:  A = softmax_row, with cross-mult
                              "none"  — no norm: A = exp(logit), with cross-mult
                              "xy_nocross" — bidimensional but NO cross-multiplication
                              "y_nocross"  — y-only softmax, NO cross-multiplication
        attraction_scale:     scale for V+; set to 0 for repulsion-only destructive ablation
        repulsion_scale:      scale for V-; set to 0 for attraction-only destructive ablation

    Returns:
        V: [N, D] drift vectors (or tuple (V, diag_dict) if return_diagnostics)
    """
    N, D = phi_gen.shape
    Npos = phi_pos.shape[0]

    # ── Pairwise L2 distances (NOT squared, per paper Eq. 12) ──
    dist_pos = torch.cdist(phi_gen, phi_pos, p=2)   # [N, Npos]
    dist_neg = torch.cdist(phi_gen, phi_gen, p=2)    # [N, N]

    # Compute unconditional distances early (needed for normalization)
    dist_unc = None
    Nunc = 0
    if phi_unc is not None and cfg_w > 0:
        Nunc = phi_unc.shape[0]
        dist_unc = torch.cdist(phi_gen, phi_unc, p=2)  # [N, Nunc]

    # ── Global distance normalization (Paper A.6) ──
    # Normalize all pairwise distances by the global mean before applying kernel.
    # This ensures τ operates on a standardized scale regardless of feature geometry.
    # CFG unconditional distances participate in the normalization.
    if normalize_distances:
        eye_N = torch.eye(N, dtype=torch.bool, device=phi_gen.device)
        dists_for_norm = [dist_pos.flatten(), dist_neg[~eye_N]]
        if dist_unc is not None:
            dists_for_norm.append(dist_unc.flatten())
        d_global = torch.cat(dists_for_norm).mean().clamp(min=1e-6)
        dist_pos = dist_pos / d_global
        dist_neg = dist_neg / d_global
        if dist_unc is not None:
            dist_unc = dist_unc / d_global

    # Mask self-distance in negatives (after normalization to keep clean distances)
    dist_neg.fill_diagonal_(1e6)

    # ── Compute logits ──
    logit_pos = -dist_pos / temperature              # [N, Npos]

    # ── kNN restriction: mask non-kNN positive logits to -inf ──
    # This makes V⁺ concentrate on the local manifold neighborhood per generated sample,
    # equivalent to the kNN barycentric effect but inside the drift field natively.
    if knn_restrict_k is not None and knn_restrict_k < Npos:
        _, topk_idx = dist_pos.topk(knn_restrict_k, dim=1, largest=False)  # [N, k]
        knn_mask = torch.ones_like(logit_pos, dtype=torch.bool)  # True = mask out
        knn_mask.scatter_(1, topk_idx, False)  # keep top-k
        logit_pos = logit_pos.masked_fill(knn_mask, float('-inf'))

    # Apply importance weights to positive logits: log(π_i) added to each positive's logit
    # This makes highly-weighted positives contribute more to V⁺ through the softmax
    if pos_weights is not None:
        # pos_weights: [Npos], normalize to sum=1 then scale by Npos to keep magnitudes
        pw = pos_weights.clamp(min=1e-8)
        pw = pw / pw.sum() * Npos  # normalized, mean=1
        logit_pos = logit_pos + pw.unsqueeze(0).log()  # [N, Npos] + [1, Npos]

    logit_neg = -dist_neg / temperature              # [N, N]

    # CFG: add unconditional negatives with weight w. For α=1, cfg_w=0 and
    # this branch is omitted, avoiding log(0).
    if dist_unc is not None and cfg_w > 0:
        logit_unc = -dist_unc / temperature + torch.log(
            torch.tensor(cfg_w, device=phi_gen.device).clamp(min=1e-8)
        )   # [N, Nunc]  — weight w applied as log(w) to logits
        logit_neg_all = torch.cat([logit_neg, logit_unc], dim=1)  # [N, N+Nunc]
    else:
        logit_neg_all = logit_neg

    # ── Concatenate for joint normalization ──
    logit = torch.cat([logit_pos, logit_neg_all], dim=1)  # [N, Npos+N+Nunc]

    # ── Attention matrix A: depends on norm_mode (Paper Table 5 ablation) ──
    use_cross = norm_mode in ("xy", "y", "none")  # cross-mult variants

    if norm_mode in ("xy", "xy_nocross"):
        # Bi-dimensional softmax (paper default Algorithm 2)
        A_row = logit.softmax(dim=-1)    # normalize over y (columns)
        A_col = logit.softmax(dim=-2)    # normalize over x (rows)
        # When kNN masking is active, some positive columns may be all -inf
        # (no sample selected that positive), producing NaN in column softmax.
        # Replace NaN with 0 — these positives contribute nothing to the drift.
        A_col = torch.nan_to_num(A_col, nan=0.0)
        A = (A_row * A_col).sqrt()       # geometric mean
    elif norm_mode in ("y", "y_nocross"):
        # Y-only softmax (paper Table 5 ablation)
        A = logit.softmax(dim=-1)
    elif norm_mode == "none":
        # No normalization: raw exp kernel (paper Table 5 ablation)
        A = (logit - logit.max(dim=-1, keepdim=True).values).exp()
    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")

    # ── Split back into pos and neg ──
    A_pos = A[:, :Npos]                         # [N, Npos]
    A_neg = A[:, Npos:]                         # [N, N+Nunc]

    # ── Weight computation ──
    if use_cross:
        # Cross-multiplication (paper Eq.11 / Algorithm 2 key step)
        W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)   # [N, Npos]
        W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)    # [N, N+Nunc]
    else:
        # No cross-multiplication: use A directly as weights
        W_pos = A_pos
        W_neg = A_neg

    # ── Compute drift ──
    drift_pos = W_pos @ phi_pos                        # [N, D]
    # Negative targets = gen samples (+ unconditional if CFG)
    if Nunc > 0:
        neg_targets = torch.cat([phi_gen, phi_unc], dim=0)  # [N+Nunc, D]
    else:
        neg_targets = phi_gen                               # [N, D]
    drift_neg = W_neg @ neg_targets                    # [N, D]

    V = attraction_scale * drift_pos - repulsion_scale * drift_neg

    if return_diagnostics:
        diag = {}
        # Kernel entropy over positives: how "peaked" is the attention?
        # Low entropy = few positives dominate (good). High = uniform (flat kernel).
        W_pos_normed = W_pos / (W_pos.sum(dim=1, keepdim=True) + 1e-8)
        ent_pos = -(W_pos_normed * (W_pos_normed + 1e-8).log()).sum(dim=1)  # [N]
        diag["kernel_entropy_pos"] = ent_pos.mean().item()
        diag["kernel_entropy_pos_max"] = math.log(max(Npos, 1))
        # Effective number of neighbors: exp(entropy)
        diag["eff_neighbors_pos"] = ent_pos.exp().mean().item()
        # V magnitude
        diag["V_norm"] = V.norm(dim=-1).mean().item()
        diag["attraction_scale"] = float(attraction_scale)
        diag["repulsion_scale"] = float(repulsion_scale)
        # Distance to positives
        diag["dist_pos_median"] = dist_pos.median().item()
        return V, diag

    return V


def phi_drift_loss_paper(
    phi_gen: torch.Tensor,        # [N, D] WITH gradient
    phi_pos: torch.Tensor,        # [Npos, D] condition-matched positives
    temperature: float = 0.05,
    phi_unc: torch.Tensor | None = None,
    cfg_w: float = 0.0,
    pos_weights: torch.Tensor | None = None,
    return_diagnostics: bool = False,
    attraction_scale: float = 1.0,
    repulsion_scale: float = 1.0,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """
    Drifting loss using paper-faithful Algorithm 2.

    L = ||φ(x) - stopgrad(φ(x) + V(φ(x)))||²

    Gradient flows: loss → φ(x) → x → generator θ.

    Args:
        pos_weights:       [Npos] importance weights to inject into V⁺ attention
        return_diagnostics: if True, return (loss, diag_dict)
    """
    phi_pos_sg = phi_pos.detach()
    phi_unc_sg = phi_unc.detach() if phi_unc is not None else None
    pw_sg = pos_weights.detach() if pos_weights is not None else None

    with torch.no_grad():
        result = compute_drift_field_paper(
            phi_gen.detach(), phi_pos_sg,
            temperature=temperature,
            phi_unc=phi_unc_sg,
            cfg_w=cfg_w,
            pos_weights=pw_sg,
            return_diagnostics=return_diagnostics,
            attraction_scale=attraction_scale,
            repulsion_scale=repulsion_scale,
        )
        if return_diagnostics:
            V, diag = result
        else:
            V = result
            diag = None
        target = phi_gen.detach() + V

    diff = phi_gen - target
    loss = (diff * diff).mean()

    if return_diagnostics:
        return loss, diag
    return loss


# ── Soft-token novelty repulsion loss ─────────────────────────────────

def soft_token_novelty_loss(
    vae_decode_fn,
    z_gen: torch.Tensor,             # [B, latent_dim] generator output (WITH gradient)
    soft_train: torch.Tensor,        # [N, L*V] pre-computed soft-token features (detached)
    data_batch_size: int = 2048,     # subsample training set per step
    tau_soft: float = 0.1,           # temperature for soft-token sharpening
    margin: float = 0.0,             # only penalize if closer than margin
) -> torch.Tensor:
    """
    Soft-token novelty repulsion: push z_gen away from training molecules
    in the decoder's soft-token space.

    The gradient chain is fully differentiable:
        loss → softmax(logits/τ) → logits → frozen_decoder(z_gen) → z_gen → generator θ

    At inference, this does NOT affect the 1-NFE pipeline:
        noise → generator → z → decoder → argmax → SMILES

    The loss pushes the generator to produce z's whose decoder output
    distributions are far from any training molecule's distributions.

    Args:
        vae_decode_fn: callable z → logits [B, L, V] (frozen VAE decoder)
        z_gen:         [B, latent_dim] generated latent vectors (requires_grad)
        soft_train:    [N, L*V] pre-computed soft-token features for training set
        data_batch_size: number of training samples to compare against per step
        tau_soft:      sharpening temperature (lower = closer to argmax, default 0.1)
        margin:        if > 0, only penalize when min_dist < margin

    Returns:
        loss: scalar, minimizing this pushes z_gen away from training molecules
    """
    B = z_gen.shape[0]
    N = soft_train.shape[0]

    # Compute sharpened soft tokens for generated z (gradient flows through!)
    logits_gen = vae_decode_fn(z_gen)         # [B, L, V]
    soft_gen = F.softmax(logits_gen / tau_soft, dim=-1)   # [B, L, V]
    soft_gen_flat = soft_gen.reshape(B, -1)   # [B, L*V]

    # Subsample training soft tokens for efficiency
    if data_batch_size < N:
        idx = torch.randint(0, N, (data_batch_size,), device=soft_train.device)
        soft_data = soft_train[idx]           # [P, L*V]
    else:
        soft_data = soft_train

    # Compute L2 distance from each z_gen to nearest training soft token
    # Use cdist for batched computation
    dists = torch.cdist(soft_gen_flat, soft_data.detach(), p=2)  # [B, P]
    min_dists = dists.min(dim=1).values       # [B]

    # Loss: penalize z_gen that are close to training molecules
    if margin > 0:
        loss = F.relu(margin - min_dists).mean()
    else:
        # Simply maximize distance to nearest training molecule
        loss = -min_dists.mean()

    return loss


def precompute_soft_train_features(
    vae_decode_fn,
    z_train: torch.Tensor,          # [N, latent_dim]
    tau_soft: float = 0.1,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Pre-compute sharpened soft-token features for training set (once, before training).

    Returns:
        soft_train: [N, L*V] on same device as z_train
    """
    device = z_train.device
    chunks = []
    with torch.no_grad():
        for i in range(0, z_train.shape[0], batch_size):
            logits = vae_decode_fn(z_train[i:i + batch_size])  # [B, L, V]
            soft = F.softmax(logits / tau_soft, dim=-1)        # [B, L, V]
            chunks.append(soft.reshape(soft.shape[0], -1))     # [B, L*V]
    return torch.cat(chunks, dim=0)  # [N, L*V]


# ── Legacy implementations below (kept for backward compatibility) ───


@torch.no_grad()
def compute_drift_field_cfg(
    phi_gen: torch.Tensor,        # [G, D] generated embeddings
    phi_pos: torch.Tensor,        # [P, D] condition-matched positive (y+)
    phi_unc: torch.Tensor,        # [U, D] unconditional real samples (y_unc)
    alpha: float,                 # CFG guidance scale (≥1)
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    CFG-aware drift field  V = V⁺ − V⁻  (Paper Appendix A.7).

    Three forces act on each generated sample:
      V⁺      : attraction toward condition-matched positive samples y⁺
      V⁻_gen  : repulsion from other generated samples  (normal kernel weights)
      V⁻_unc  : repulsion from unconditional real samples (kernel weights × w)

    CFG weight:
        w = (α − 1)(N_gen − 1) / N_unc
        - α=1 → w=0 → standard unconditional drift field
        - α>1 → pushes generator away from marginal → conditional concentration

    Args:
        phi_gen:      [G, D] generated embeddings
        phi_pos:      [P, D] condition-matched positive data embeddings
        phi_unc:      [U, D] unconditional real data embeddings
        alpha:        CFG guidance scale (≥1.0)
        temperature:  kernel bandwidth multiplier

    Returns:
        V: [G, D] drift vectors
    """
    G, D = phi_gen.shape
    U = phi_unc.shape[0]

    # CFG weight for unconditional repulsion
    w_cfg = (alpha - 1.0) * max(G - 1, 1) / max(U, 1)

    # ── V⁺: attraction to condition-matched positive samples ──
    w_pos = _rbf_kernel_weights(phi_gen, phi_pos, temperature)  # [G, P]
    V_pos = w_pos @ phi_pos - phi_gen  # [G, D]

    # ── V⁻_gen: repulsion from other generated samples ──
    V_neg_gen = torch.zeros_like(phi_gen)
    if G > 1:
        sq_dist_gg = torch.cdist(phi_gen, phi_gen, p=2).pow(2)  # [G, G]
        sigma2_gg = sq_dist_gg.median().clamp(min=1e-6) * temperature
        log_w_gg = -sq_dist_gg / (2.0 * sigma2_gg)
        log_w_gg.fill_diagonal_(float("-inf"))
        w_gg = torch.softmax(log_w_gg, dim=1)  # [G, G]
        V_neg_gen = w_gg @ phi_gen - phi_gen  # [G, D]

    # ── V⁻_unc: repulsion from unconditional samples (weighted by w) ──
    V_neg_unc = torch.zeros_like(phi_gen)
    if U > 0 and w_cfg > 0:
        w_unc = _rbf_kernel_weights(phi_gen, phi_unc, temperature)  # [G, U]
        V_neg_unc = w_unc @ phi_unc - phi_gen  # [G, D]

    # V = V⁺ − (V⁻_gen + w · V⁻_unc)
    V = V_pos - (V_neg_gen + w_cfg * V_neg_unc)
    return V


def phi_drift_loss_cfg(
    phi_gen: torch.Tensor,        # [G, D] WITH gradient
    phi_pos: torch.Tensor,        # [P, D] condition-matched positive
    phi_unc: torch.Tensor,        # [U, D] unconditional real samples
    alpha: float,                 # CFG guidance scale
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    CFG-aware φ-space drifting loss (Paper Appendix A.7).

    L = ‖φ(x) − stopgrad(φ(x) + V_cfg(φ(x)))‖²

    Gradient flows:  loss → φ(x) → x → generator θ.
    The drift field V_cfg is computed under stopgrad.
    """
    phi_pos_sg = phi_pos.detach()
    phi_unc_sg = phi_unc.detach()

    with torch.no_grad():
        V = compute_drift_field_cfg(
            phi_gen.detach(), phi_pos_sg, phi_unc_sg,
            alpha=alpha,
            temperature=temperature,
        )
        target = phi_gen.detach() + V

    diff = phi_gen - target
    loss = (diff * diff).mean()
    return loss


# ══════════════════════════════════════════════════════════════════════
# Set-Level Alignment Losses (replacing zmatch)
# ══════════════════════════════════════════════════════════════════════

def _sinkhorn_log(
    cost: torch.Tensor,   # [G, P] pairwise cost matrix
    epsilon: float,
    n_iter: int = 20,
) -> torch.Tensor:
    """Log-domain Sinkhorn iteration for entropic OT.

    Returns log transport plan T (in log space for numerical stability).
    """
    G, P = cost.shape
    # Uniform marginals
    log_mu = -math.log(G) * torch.ones(G, device=cost.device)
    log_nu = -math.log(P) * torch.ones(P, device=cost.device)

    log_K = -cost / epsilon  # [G, P]

    # Sinkhorn iterations in log space
    log_u = torch.zeros(G, device=cost.device)
    log_v = torch.zeros(P, device=cost.device)
    for _ in range(n_iter):
        log_u = log_mu - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_nu - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)

    log_T = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)  # [G, P]
    return log_T


def sinkhorn_alignment_loss(
    z_gen: torch.Tensor,     # [G, D] generated latents (requires_grad)
    z_pos: torch.Tensor,     # [P, D] condition-matched real latents
    epsilon: float = 1.0,    # entropic regularization strength
    n_iter: int = 20,        # Sinkhorn iterations
) -> torch.Tensor:
    """Sinkhorn (entropic OT) set-level alignment loss.

    Replaces zmatch's single-point centroid/nn_soft that collapsed uniqueness.
    Uses soft optimal transport to match the generated set to the real set
    while preserving diversity.

    L = Σ_{i,j} T_{ij} · ||z_gen_i - z_pos_j||²

    where T is the entropic OT plan (doubly-stochastic via Sinkhorn).
    Gradient flows through z_gen; z_pos and T are detached.
    """
    with torch.no_grad():
        cost = torch.cdist(z_gen.detach(), z_pos.detach(), p=2).pow(2)  # [G, P]
        log_T = _sinkhorn_log(cost, epsilon=epsilon, n_iter=n_iter)
        T = log_T.exp()  # [G, P] transport plan

    # Weighted targets: each z_gen_i targets a weighted barycenter of z_pos
    # T[i,:] sums to ~1/G, so normalize per-row for target computation
    T_row = T / T.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [G, P]
    z_target = T_row @ z_pos.detach()  # [G, D]

    diff = z_gen - z_target
    loss = (diff * diff).mean()
    return loss


def knn_barycentric_alignment_loss(
    z_gen: torch.Tensor,     # [G, D] generated latents (requires_grad)
    z_pos: torch.Tensor,     # [P, D] condition-matched real latents
    k: int = 8,              # number of nearest neighbors
    temperature: float = 1.0, # softmax temperature for weighting
) -> torch.Tensor:
    """kNN barycentric alignment loss with entropy regularization.

    Each generated sample is pulled toward a soft barycenter of its k-nearest
    neighbors in the real set, with temperature-controlled softmax weights.

    Unlike zmatch-centroid (ALL z_gen → single mean) or zmatch-nn_soft
    (soft-nearest but still mode-seeking), this provides:
    - Per-sample diverse targets (different z_gen can target different regions)
    - Soft interpolation prevents collapse to single point
    - Temperature controls exploration vs exploitation

    L = mean_i ||z_gen_i - barycenter_i||²
    barycenter_i = Σ_j softmax(-d(z_gen_i, z_pos_j)/τ) · z_pos_j  (over top-k j)
    """
    with torch.no_grad():
        dists = torch.cdist(z_gen.detach(), z_pos.detach(), p=2)  # [G, P]
        # Select top-k nearest neighbors
        topk_dists, topk_idx = dists.topk(k, dim=1, largest=False)  # [G, k]
        # Softmax weights over kNN distances
        weights = F.softmax(-topk_dists / temperature, dim=1)  # [G, k]
        # Gather kNN z_pos vectors
        z_knn = z_pos[topk_idx]  # [G, k, D]
        # Compute weighted barycenter
        z_target = (weights.unsqueeze(-1) * z_knn).sum(dim=1)  # [G, D]

    diff = z_gen - z_target
    loss = (diff * diff).mean()
    return loss


# ══════════════════════════════════════════════════════════════════════
# Correlation Structure Loss (for MW recovery)
# ══════════════════════════════════════════════════════════════════════

def correlation_structure_loss(
    z_gen: torch.Tensor,        # [G, D] generated latents
    z_data: torch.Tensor,       # [P, D] real data latents
    subspace_dims: int | None = None,  # if set, only match top-k PCA dims
) -> torch.Tensor:
    """Match the cross-dimensional correlation structure between generated and real latents.

    Key insight: MW is encoded in high-order inter-dimensional structure.
    Shuffling z dims independently drops MW from 331→272 (same marginals,
    destroyed correlations). This loss explicitly matches the correlation
    matrix of generated z to the real data's correlation matrix.

    L = ||corr(z_gen) - corr(z_data)||_F² / D²

    If subspace_dims is set, projects both sets into the top PCA subspace
    of z_data first (focusing on the most informative dimensions).
    """
    # Optionally project into PCA subspace
    if subspace_dims is not None and subspace_dims < z_gen.shape[1]:
        with torch.no_grad():
            # PCA of real data
            z_data_c = z_data - z_data.mean(0, keepdim=True)
            _, _, Vh = torch.linalg.svd(z_data_c, full_matrices=False)
            proj = Vh[:subspace_dims]  # [k, D]
        z_gen_proj = (z_gen - z_data.mean(0, keepdim=True).detach()) @ proj.T  # [G, k]
        z_data_proj = z_data_c @ proj.T  # [P, k]
    else:
        z_gen_proj = z_gen
        z_data_proj = z_data

    D = z_gen_proj.shape[1]

    # Compute correlation matrices
    def _corr(z: torch.Tensor) -> torch.Tensor:
        z_c = z - z.mean(0, keepdim=True)
        std = z_c.std(0, keepdim=True).clamp(min=1e-6)
        z_n = z_c / std
        return z_n.T @ z_n / max(z.shape[0] - 1, 1)  # [D, D]

    corr_gen = _corr(z_gen_proj)
    with torch.no_grad():
        corr_data = _corr(z_data_proj)

    diff = corr_gen - corr_data
    loss = (diff * diff).sum() / (D * D)
    return loss


def covariance_matching_loss(
    z_gen: torch.Tensor,     # [G, D]
    z_data: torch.Tensor,    # [P, D]
) -> torch.Tensor:
    """Match the covariance matrix of generated latents to real data.

    Simpler than correlation_structure_loss — matches raw covariance
    without normalization. Better for preserving scale information.

    L = ||cov(z_gen) - cov(z_data)||_F² / D²
    """
    D = z_gen.shape[1]

    z_gen_c = z_gen - z_gen.mean(0, keepdim=True)
    cov_gen = z_gen_c.T @ z_gen_c / max(z_gen.shape[0] - 1, 1)

    with torch.no_grad():
        z_data_c = z_data - z_data.mean(0, keepdim=True)
        cov_data = z_data_c.T @ z_data_c / max(z_data.shape[0] - 1, 1)

    diff = cov_gen - cov_data
    loss = (diff * diff).sum() / (D * D)
    return loss
