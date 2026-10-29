from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from grpo.train import latest_checkpoint, resolve_resume_from_checkpoint


class TrainResumeTest(unittest.TestCase):
    def test_latest_checkpoint_picks_largest_step(self) -> None:
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "checkpoint-2").mkdir()
            (output_dir / "checkpoint-17").mkdir()
            (output_dir / "checkpoint-final").mkdir()
            (output_dir / "not-a-checkpoint").mkdir()

            self.assertEqual(latest_checkpoint(output_dir), output_dir / "checkpoint-17")

    def test_resume_from_checkpoint_auto_resolves_latest_output_checkpoint(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "outputs/run-a"
            (output_dir / "checkpoint-10").mkdir(parents=True)
            (output_dir / "checkpoint-70").mkdir()
            config = {
                "_project_dir": str(root),
                "training": {
                    "output_dir": "./outputs/run-a",
                    "resume_from_checkpoint": "auto",
                },
            }

            self.assertEqual(resolve_resume_from_checkpoint(config), str(output_dir / "checkpoint-70"))

    def test_resume_from_checkpoint_none_starts_fresh(self) -> None:
        with TemporaryDirectory() as tmp:
            config = {
                "_project_dir": tmp,
                "training": {
                    "output_dir": "./outputs/run-a",
                    "resume_from_checkpoint": None,
                },
            }

            self.assertIsNone(resolve_resume_from_checkpoint(config))


if __name__ == "__main__":
    unittest.main()
