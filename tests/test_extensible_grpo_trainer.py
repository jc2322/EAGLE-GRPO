from __future__ import annotations

import sys
import types
import unittest

from utils import extensible_grpo_trainer


class ExtensibleGrpoTrainerVersionTest(unittest.TestCase):
    def test_trl_1_6_uses_base_loss_for_token_advantages(self) -> None:
        original = sys.modules.get("trl")
        sys.modules["trl"] = types.SimpleNamespace(__version__="1.6.0")
        try:
            self.assertEqual(extensible_grpo_trainer.trl_major_minor(), (1, 6))
            self.assertTrue(extensible_grpo_trainer.trl_supports_token_advantages_in_base_loss())
        finally:
            if original is None:
                sys.modules.pop("trl", None)
            else:
                sys.modules["trl"] = original

    def test_trl_024_keeps_legacy_token_loss_path(self) -> None:
        original = sys.modules.get("trl")
        sys.modules["trl"] = types.SimpleNamespace(__version__="0.24.0")
        try:
            self.assertEqual(extensible_grpo_trainer.trl_major_minor(), (0, 24))
            self.assertFalse(extensible_grpo_trainer.trl_supports_token_advantages_in_base_loss())
            extensible_grpo_trainer.require_supported_trl_024()
        finally:
            if original is None:
                sys.modules.pop("trl", None)
            else:
                sys.modules["trl"] = original


if __name__ == "__main__":
    unittest.main()
