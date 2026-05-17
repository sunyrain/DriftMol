import subprocess
import sys
import unittest
import os


class GraphTemperatureDecodeTest(unittest.TestCase):
    def test_archive_discretize_logits_accepts_temperature(self):
        code = """
import torch
from src.utils import discretize_logits

node = torch.randn(3, 29, 6)
edge = torch.randn(3, 29, 29, 4)
for temp in (0.0, 0.5, 1.0):
    graphs = discretize_logits(node, edge, temperature=temp)
    assert len(graphs) == 3
    assert graphs[0]["node_type"].shape == (29,)
    assert graphs[0]["edge_type"].shape == (29, 29)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd="archive/graph_vae_line",
            env={**os.environ, "PYTHONPATH": "."},
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
