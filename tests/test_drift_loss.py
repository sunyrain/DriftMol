from __future__ import annotations

import torch
import unittest

from src.drifting.drift_latent_phi import compute_drift_field_paper, multi_temp_drift_loss


class DriftLossTest(unittest.TestCase):
    def test_paper_drift_field_shape_and_finite_values(self):
        torch.manual_seed(0)
        phi_gen = torch.randn(8, 16)
        phi_pos = torch.randn(12, 16)
        phi_unc = torch.randn(4, 16)

        drift = compute_drift_field_paper(
            phi_gen,
            phi_pos,
            temperature=0.05,
            phi_unc=phi_unc,
            cfg_w=1.0,
            normalize_distances=True,
        )

        self.assertEqual(drift.shape, phi_gen.shape)
        self.assertTrue(torch.isfinite(drift).all())

    def test_drift_component_scales_decompose_full_field(self):
        torch.manual_seed(1)
        phi_gen = torch.randn(6, 8)
        phi_pos = torch.randn(10, 8)

        full = compute_drift_field_paper(
            phi_gen,
            phi_pos,
            temperature=0.2,
            normalize_distances=True,
        )
        attraction_only = compute_drift_field_paper(
            phi_gen,
            phi_pos,
            temperature=0.2,
            normalize_distances=True,
            attraction_scale=1.0,
            repulsion_scale=0.0,
        )
        repulsion_only = compute_drift_field_paper(
            phi_gen,
            phi_pos,
            temperature=0.2,
            normalize_distances=True,
            attraction_scale=0.0,
            repulsion_scale=1.0,
        )
        zeroed = compute_drift_field_paper(
            phi_gen,
            phi_pos,
            temperature=0.2,
            normalize_distances=True,
            attraction_scale=0.0,
            repulsion_scale=0.0,
        )

        self.assertTrue(torch.allclose(full, attraction_only + repulsion_only, atol=1e-6))
        self.assertTrue(torch.allclose(zeroed, torch.zeros_like(zeroed), atol=1e-7))

    def test_multi_temp_drift_loss_backpropagates_to_generated_features(self):
        torch.manual_seed(0)
        phi_gen = torch.randn(8, 16, requires_grad=True)
        phi_pos = torch.randn(12, 16)

        loss = multi_temp_drift_loss(
            phi_gen,
            phi_pos,
            temperatures=[0.02, 0.05],
            normalize_drift=True,
            normalize_distances=True,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(phi_gen.grad)
        self.assertTrue(torch.isfinite(phi_gen.grad).all())
        self.assertGreater(phi_gen.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
