import json
import os
import tempfile
import unittest
from pathlib import Path

import scripts.audit_reviewer_experiment_readiness as audit


class ReviewerReadinessAuditTest(unittest.TestCase):
    def _write_json(self, root: Path, rel: str, payload: dict):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    def _touch(self, root: Path, rel: str):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n")

    def _prepare_root(self, root: Path, *, complete: bool):
        for rel in [
            "docs/PUBLICATION_PLAN.md",
            "docs/REVIEWER_EXPERIMENT_MATRIX.md",
            "docs/REVIEWER_GOAL_COMPLETION_AUDIT.md",
            "docs/DRIFTING_FAITHFULNESS_PLAN.md",
            "docs/REVIEWER_PROMPT_TO_ARTIFACT_CHECKLIST.md",
            "docs/DRIFTING_ALGORITHM_AUDIT.md",
            "docs/GRAPH_NAMESPACE_ADAPTER_PLAN.md",
            "results/faithful_drifting.csv",
            "results/tables/tab_faithful_drifting_core.tex",
            "results/tables/tab_faithful_drifting_allocation.tex",
            "scripts/render_faithful_supplement.py",
            "scripts/collect_trained_baselines.py",
            "scripts/collect_vae_drift_results.py",
            "scripts/collect_generalization_results.py",
            "scripts/collect_reviewer_extra_results.py",
            "scripts/watch_reviewer_extra_postprocess.py",
            "scripts/audit_graph_archive_launchability.py",
            "scripts/generate_next_wave_configs.py",
            "scripts/collect_next_wave_results.py",
            "docs/SUPPLEMENT_FAITHFUL_DRIFTING.tex",
            "docs/SUPPLEMENT_FAITHFUL_DRIFTING_INLINED.tex",
            "docs/SUPPLEMENT_FAITHFUL_DRIFTING_AAAI.tex",
            "DriftingMol_AAAI_FaithfulSupplement.pdf",
            "docs/NEXT_WAVE_EXPERIMENT_PLAN.md",
            "results/tables/tab_trained_baseline_qed.tex",
            "results/tables/tab_vae_drift_downstream.tex",
            "results/tables/tab_generalization.tex",
            "results/tables/tab_reviewer_extra.tex",
            "results/tables/tab_next_wave.tex",
        ]:
            self._touch(root, rel)

        entries = []
        for idx in range(4):
            config = f"configs/reviewer_faithful/core/run_{idx}.yaml"
            self._touch(root, config)
            entries.append({"group": "faithful_core", "name": f"core_{idx}", "config": config})
        for idx in range(6):
            config = f"configs/reviewer_faithful/allocation/run_{idx}.yaml"
            self._touch(root, config)
            entries.append({"group": "faithful_allocation", "name": f"alloc_{idx}", "config": config})
        self._write_json(root, "configs/reviewer_faithful/manifest.json", {"entries": entries})
        baseline_entries = []
        for seed in [42, 43, 44]:
            config = f"configs/publication_ext/baselines/ext_B_LINEAR_PROP_QED_s{seed}.yaml"
            self._touch(root, config)
            baseline_entries.append({
                "group": "trained_baseline",
                "name": f"ext_B_LINEAR_PROP_QED_s{seed}",
                "config": config,
            })
        self._write_json(root, "configs/publication_ext/baseline_manifest.json", {"entries": baseline_entries})
        vae_drift_entries = []
        for label in ["lowbeta", "highbeta", "latent128", "dec6"]:
            config = f"configs/publication_ext/vae_drift/ext_vae_{label}_drift_qed_s42.yaml"
            self._touch(root, config)
            vae_drift_entries.append({
                "group": "vae_drift_downstream",
                "name": f"ext_vae_{label}_drift_qed_s42",
                "config": config,
            })
        self._write_json(root, "configs/publication_ext/vae_drift_manifest.json", {"entries": vae_drift_entries})
        generalization_entries = []
        for seed in [43, 44]:
            config = f"configs/publication_ext/generalization/ext_G4_multi4_v2_s{seed}.yaml"
            self._touch(root, config)
            generalization_entries.append({
                "group": "multi4_seed_stability",
                "name": f"ext_G4_multi4_v2_s{seed}",
                "config": config,
            })
        for target in ["logp", "sa"]:
            config = f"configs/publication_ext/generalization/ext_G4_{target}_qed_s42.yaml"
            self._touch(root, config)
            generalization_entries.append({
                "group": "single_property_generalization",
                "name": f"ext_G4_{target}_qed_s42",
                "config": config,
            })
        self._write_json(root, "configs/publication_ext/generalization_manifest.json", {"entries": generalization_entries})
        reviewer_extra_specs = [
            ("continuous_conditioning", "ext_G4_qed_continuous_s42"),
            ("single_property_seed_extension", "ext_G4_logp_qed_s43"),
            ("single_property_seed_extension", "ext_G4_sa_qed_s43"),
            ("vae_drift_seed_extension", "ext_vae_lowbeta_drift_qed_s43"),
        ]
        reviewer_extra_entries = []
        for group, name in reviewer_extra_specs:
            config = f"configs/publication_ext/reviewer_extra/{name}.yaml"
            self._touch(root, config)
            reviewer_extra_entries.append({"group": group, "name": name, "config": config})
        self._write_json(root, "configs/publication_ext/reviewer_extra_manifest.json", {"entries": reviewer_extra_entries})
        next_wave_specs = [
            ("property_guidance_baseline", "ext_NW_LINEAR_PROP_LOGP_s42"),
            ("property_guidance_baseline", "ext_NW_LINEAR_PROP_SA_s42"),
            ("property_guidance_baseline", "ext_NW_LINEAR_PROP_MULTI4_s42"),
            ("conditioning_seed_stability", "ext_NW_QED_CONTINUOUS_s43"),
        ]
        next_wave_entries = []
        for group, name in next_wave_specs:
            config = f"configs/publication_ext/next_wave/{name}.yaml"
            self._touch(root, config)
            next_wave_entries.append({"group": group, "name": name, "config": config})
        self._write_json(root, "configs/publication_ext/next_wave_manifest.json", {"entries": next_wave_entries})
        self._write_json(
            root,
            "configs/publication_ext/graph_stress_manifest.json",
            {
                "resource_policy": {"launch_now": False, "minimum_recommended_disk_free_gb": 20},
                "preconditions": [
                    "Isolate archive/graph_vae_line",
                    "Recover graph VAE",
                    "Rebuild latent cache",
                    "Recover latent MAE",
                ],
                "archived_diagnostics": [
                    {"name": "e36_dec_drift_cfg", "metrics": "archive/graph_vae_line/outputs/e36_dec_drift_cfg/final_metrics.json"},
                    {"name": "e40_logp_bins_queue", "metrics": "archive/graph_vae_line/outputs/e40_logp_bins_queue/final_metrics.json"},
                ],
                "entries": [
                    {"group": "graph_recovery", "name": "graph_recover_vae_v3_valence", "config": "archive/graph_vae_line/configs/vae_v3_valence.yaml"},
                    {"group": "graph_recovery", "name": "graph_rebuild_latent_cache_v3", "config": "archive/graph_vae_line/configs/vae_v3_valence.yaml"},
                    {"group": "graph_recovery", "name": "graph_recover_latent_mae_v3", "config": "archive/graph_vae_line/configs/latent_mae_v3.yaml"},
                    {"group": "graph_control", "name": "graph_fresh_qed_e36", "config": "archive/graph_vae_line/configs/e36_dec_drift_cfg.yaml"},
                    {"group": "graph_control", "name": "graph_fresh_logp_e40", "config": "archive/graph_vae_line/configs/e40_logp_bins_queue.yaml"},
                    {"group": "graph_destructive", "name": "graph_qed_destructive_no_drift", "config": "to_create_from_e36_with_drift_disabled"},
                    {"group": "graph_decode_diagnostic", "name": "graph_raw_vs_repaired_decode", "config": "archive graph decode repair scripts"},
                ],
            },
        )
        self._write_json(
            root,
            "results/drifting_faithfulness_status.json",
            {
                "algorithm2_equivalence": {"status": "PASS", "max_abs_diff": 0.0},
                "strict_protocol_configs": {"status": "PASS", "checked": 10, "failures": []},
            },
        )
        self._write_json(
            root,
            "results/destructive_ablation_status.json",
            {"minimum_completed_runs_reached": True, "complete": 4, "num_experiments": 7},
        )

        faithful_complete = 4 if complete else 0
        allocation_complete = 6 if complete else 0
        self._write_json(
            root,
            "results/faithful_drifting_status.json",
            {
                "faithful_core_complete": complete,
                "groups": {
                    "faithful_core": {
                        "complete": faithful_complete,
                        "total": 4,
                        "pending": 4 - faithful_complete,
                    },
                    "faithful_allocation": {
                        "complete": allocation_complete,
                        "total": 6,
                        "pending": 6 - allocation_complete,
                    },
                },
            },
        )
        self._write_json(
            root,
            "results/vae_sensitivity_status.json",
            {"complete": 1 if complete else 0, "num_experiments": 4},
        )
        self._write_json(
            root,
            "results/graph_archive_launchability_status.json",
            {
                "complete": False,
                "missing_required_artifacts": [{"path": "missing"}],
                "namespace": {"blockers": ["namespace"]},
            },
        )
        self._touch(root, "results/graph_archive_launchability_audit.md")
        self._write_json(root, "results/extension_completion_status.json", {"complete": complete})
        self._write_json(
            root,
            "results/trained_baseline_status.json",
            {"complete": 0, "num_experiments": 3, "pending_or_incomplete": 3},
        )
        self._write_json(
            root,
            "results/vae_drift_downstream_status.json",
            {"complete": 1 if complete else 0, "num_experiments": 4, "pending_or_incomplete": 3 if complete else 4},
        )
        self._write_json(
            root,
            "results/generalization_status.json",
            {"complete": 1 if complete else 0, "num_experiments": 4, "pending_or_incomplete": 3 if complete else 4},
        )
        self._write_json(
            root,
            "results/reviewer_extra_status.json",
            {"complete": 1 if complete else 0, "num_experiments": 4, "pending_or_incomplete": 3 if complete else 4},
        )
        self._write_json(
            root,
            "results/next_wave_status.json",
            {"complete": 0, "num_experiments": 4, "pending_or_incomplete": 4},
        )

        pid_file = root / "outputs/reviewer_faithful/deferred_faithful_core_launcher.pid"
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()))
        for gpu in range(4):
            vae_pid = root / f"outputs/publication_ext/vae_drift_launcher_gpu{gpu}.pid"
            vae_pid.parent.mkdir(parents=True, exist_ok=True)
            vae_pid.write_text(str(os.getpid()))
            gen_pid = root / f"outputs/publication_ext/generalization_launcher_gpu{gpu}.pid"
            gen_pid.parent.mkdir(parents=True, exist_ok=True)
            gen_pid.write_text(str(os.getpid()))
            extra_pid = root / f"outputs/publication_ext/reviewer_extra_launcher_gpu{gpu}.pid"
            extra_pid.parent.mkdir(parents=True, exist_ok=True)
            extra_pid.write_text(str(os.getpid()))
        vae_post = root / "outputs/publication_ext/vae_drift_postprocess.pid"
        vae_post.parent.mkdir(parents=True, exist_ok=True)
        vae_post.write_text(str(os.getpid()))
        gen_post = root / "outputs/publication_ext/generalization_postprocess.pid"
        gen_post.parent.mkdir(parents=True, exist_ok=True)
        gen_post.write_text(str(os.getpid()))
        extra_post = root / "outputs/publication_ext/reviewer_extra_postprocess.pid"
        extra_post.parent.mkdir(parents=True, exist_ok=True)
        extra_post.write_text(str(os.getpid()))

    def _with_root(self, root: Path):
        old_root, old_json, old_md = audit.ROOT, audit.OUT_JSON, audit.OUT_MD
        audit.ROOT = root
        audit.OUT_JSON = root / "results/reviewer_experiment_readiness_status.json"
        audit.OUT_MD = root / "results/reviewer_experiment_readiness_audit.md"
        return old_root, old_json, old_md

    def _restore_root(self, saved):
        audit.ROOT, audit.OUT_JSON, audit.OUT_MD = saved

    def test_open_until_strict_runs_and_vae_are_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._prepare_root(root, complete=False)
            saved = self._with_root(root)
            try:
                status = audit.build_status()
            finally:
                self._restore_root(saved)

        rows = {row["requirement"]: row for row in status["rows"]}
        self.assertEqual(status["overall"], "OPEN")
        self.assertEqual(rows["Prompt-to-artifact checklist exists"]["status"], "PASS")
        self.assertEqual(rows["Strict faithful core runs are complete"]["status"], "OPEN")
        self.assertEqual(rows["Strict faithful allocation sweeps are complete"]["status"], "OPEN")
        self.assertEqual(rows["VAE sensitivity has at least one completed alternative"]["status"], "OPEN")
        self.assertEqual(rows["Graph archive launchability preflight is recorded"]["status"], "PASS")
        self.assertEqual(rows["Downstream VAE-drift queue and collector are organized"]["status"], "PASS")
        self.assertEqual(rows["Generalization queue and collector are organized"]["status"], "PASS")
        self.assertEqual(rows["Reviewer-extra queue and collector are organized"]["status"], "PASS")
        self.assertEqual(rows["Next-wave reviewer experiments are prepared or running"]["status"], "PASS")
        self.assertEqual(rows["Trained baseline queue and collector are organized"]["status"], "PASS")

    def test_pass_only_when_all_gates_are_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._prepare_root(root, complete=True)
            saved = self._with_root(root)
            try:
                status = audit.build_status()
            finally:
                self._restore_root(saved)

        self.assertEqual(status["overall"], "PASS")
        self.assertTrue(all(row["status"] == "PASS" for row in status["rows"]))

    def test_manifest_requires_existing_config_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._prepare_root(root, complete=True)
            (root / "configs/reviewer_faithful/core/run_0.yaml").unlink()
            saved = self._with_root(root)
            try:
                status = audit.build_status()
            finally:
                self._restore_root(saved)

        rows = {row["requirement"]: row for row in status["rows"]}
        row = rows["Reviewer-faithful manifest has strict core and allocation configs"]
        self.assertEqual(status["overall"], "OPEN")
        self.assertEqual(row["status"], "OPEN")
        self.assertIn("core_0", row["note"])

    def test_readiness_requires_strict_protocol_config_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._prepare_root(root, complete=True)
            self._write_json(
                root,
                "results/drifting_faithfulness_status.json",
                {
                    "algorithm2_equivalence": {"status": "PASS", "max_abs_diff": 0.0},
                    "strict_protocol_configs": {
                        "status": "OPEN",
                        "checked": 10,
                        "failures": ["unit failure"],
                    },
                },
            )
            saved = self._with_root(root)
            try:
                status = audit.build_status()
            finally:
                self._restore_root(saved)

        rows = {row["requirement"]: row for row in status["rows"]}
        row = rows["Strict reviewer-faithful config protocol passes"]
        self.assertEqual(status["overall"], "OPEN")
        self.assertEqual(row["status"], "OPEN")
        self.assertIn("unit failure", row["note"])


if __name__ == "__main__":
    unittest.main()
