"""Quality Gate: strict multi-metric evaluation gate for molecular generation.

Prevents experiments with:
- Collapsed uniqueness (zmatch-style memorization)
- Per-bin coverage collapse (any condition bin failing)
- Training set memorization (high NN similarity)
- Poor controllability (low ρ)
- Mode collapse (low internal diversity)

Usage:
    from src.eval.quality_gate import QualityGate
    gate = QualityGate()  # or QualityGate.from_config(cfg)
    result = gate.evaluate(metrics_dict)
    # result.passed, result.gated_score, result.reasons
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GateResult:
    """Result of quality gate evaluation."""
    passed: bool
    gated_score: float
    raw_score: float
    checks: dict[str, bool]
    reasons: list[str]

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        parts = [f"Gate={status} score={self.gated_score:.4f}"]
        if not self.passed:
            parts.append(f"failures: {', '.join(self.reasons)}")
        return " | ".join(parts)


@dataclass
class QualityGate:
    """Multi-metric quality gate with configurable thresholds.

    All minimum thresholds must be met simultaneously. Any single failure
    results in gated_score=0 and passed=False.
    """

    # ── V/U/N thresholds ──
    min_validity: float = 0.75
    min_uniqueness: float = 0.60       # zmatch collapsed to 0.02–0.25
    min_novelty: float = 0.40

    # ── Diversity thresholds ──
    min_int_div: float = 0.65
    min_scaffold_diversity: float = 0.10  # unique scaffolds / valid mols

    # ── Conditional control ──
    min_spearman_rho: float = 0.10
    min_per_bin_uniqueness: float = 0.40  # any bin below → collapse

    # ── Distribution quality ──
    max_fcd: float = 35.0               # generous; data→decode FCD ≈ 1.54

    # ── Memorization detection ──
    max_nn_sim: float = 0.90            # avg NN Tanimoto to train set

    # ── Mode ──
    is_conditional: bool = True

    @classmethod
    def from_config(cls, gate_cfg: dict) -> QualityGate:
        """Create gate from a config dict (only known fields are used)."""
        known = {k: v for k, v in gate_cfg.items()
                 if k in cls.__dataclass_fields__}
        return cls(**known)

    def evaluate(self, metrics: dict) -> GateResult:
        """Run all checks and return a GateResult."""
        checks: dict[str, bool] = {}
        reasons: list[str] = []

        # ── V / U / N ──
        v = metrics.get("validity", 0)
        u = metrics.get("uniqueness", 0)
        n = metrics.get("novelty", 0)

        checks["validity"] = v >= self.min_validity
        if not checks["validity"]:
            reasons.append(f"V={v:.3f}<{self.min_validity}")

        checks["uniqueness"] = u >= self.min_uniqueness
        if not checks["uniqueness"]:
            reasons.append(f"U={u:.3f}<{self.min_uniqueness}")

        checks["novelty"] = n >= self.min_novelty
        if not checks["novelty"]:
            reasons.append(f"N={n:.3f}<{self.min_novelty}")

        # ── Internal diversity ──
        int_div = metrics.get("int_div", 0)
        checks["int_div"] = int_div >= self.min_int_div
        if not checks["int_div"]:
            reasons.append(f"IntDiv={int_div:.3f}<{self.min_int_div}")

        # ── Scaffold diversity (skip if not provided) ──
        scaf_div = metrics.get("scaffold_diversity", -1)
        if scaf_div >= 0:
            checks["scaffold_diversity"] = scaf_div >= self.min_scaffold_diversity
            if not checks["scaffold_diversity"]:
                reasons.append(
                    f"ScafDiv={scaf_div:.3f}<{self.min_scaffold_diversity}")

        # ── Conditional-specific gates ──
        if self.is_conditional:
            rho = metrics.get("spearman_rho", 0)
            checks["spearman_rho"] = rho >= self.min_spearman_rho
            if not checks["spearman_rho"]:
                reasons.append(f"ρ={rho:.3f}<{self.min_spearman_rho}")

            # Per-bin uniqueness collapse detection
            per_bin_u = metrics.get("per_bin_uniqueness", {})
            if per_bin_u:
                min_bin_u = min(per_bin_u.values())
                checks["per_bin_u"] = min_bin_u >= self.min_per_bin_uniqueness
                if not checks["per_bin_u"]:
                    bad = {k: f"{bv:.2f}" for k, bv in per_bin_u.items()
                           if bv < self.min_per_bin_uniqueness}
                    reasons.append(f"per_bin_U collapse: {bad}")

        # ── FCD (lower is better; skip if not computed) ──
        fcd = metrics.get("fcd", -1)
        if fcd >= 0:
            checks["fcd"] = fcd <= self.max_fcd
            if not checks["fcd"]:
                reasons.append(f"FCD={fcd:.2f}>{self.max_fcd}")

        # ── NN similarity / memorization (skip if not computed) ──
        nn_sim = metrics.get("nn_sim_mean", -1)
        if nn_sim >= 0:
            checks["nn_sim"] = nn_sim <= self.max_nn_sim
            if not checks["nn_sim"]:
                reasons.append(
                    f"NN_sim={nn_sim:.3f}>{self.max_nn_sim} (memorization)")

        # ── Verdict ──
        passed = all(checks.values()) if checks else False

        # ── Composite score ──
        vun = v * u * n
        if self.is_conditional:
            rho = metrics.get("spearman_rho", 0)
            raw_score = vun * (1 + max(rho, 0))
        else:
            raw_score = vun

        gated_score = raw_score if passed else 0.0

        return GateResult(
            passed=passed,
            gated_score=gated_score,
            raw_score=raw_score,
            checks=checks,
            reasons=reasons,
        )
