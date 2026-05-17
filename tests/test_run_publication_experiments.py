import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RunPublicationExperimentsTest(unittest.TestCase):
    def test_dry_run_does_not_write_status_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest = tmp_path / "manifest.json"
            status = tmp_path / "status.json"
            manifest.write_text(json.dumps({
                "entries": [
                    {
                        "group": "unit",
                        "name": "pub_unit_qed_s42",
                        "command": "echo should_not_run",
                        "output_dir": "outputs/publication/unit/pub_unit_qed_s42",
                    }
                ]
            }))
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_publication_experiments.py",
                    "--manifest",
                    str(manifest),
                    "--status-file",
                    str(status),
                    "--dry-run",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout)
            self.assertFalse(status.exists())


if __name__ == "__main__":
    unittest.main()
