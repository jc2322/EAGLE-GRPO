from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from grpo.train import maybe_prepare_grpo_lora_model


class FakeModel:
    def __init__(self, name_or_path: str = "merged-model") -> None:
        self.name_or_path = name_or_path
        self.config = SimpleNamespace(_name_or_path=name_or_path, name_or_path=name_or_path)


class BaselineTrainTest(unittest.TestCase):
    def test_grpo_lora_is_disabled_without_explicit_lora_section(self) -> None:
        model = FakeModel()
        self.assertIs(maybe_prepare_grpo_lora_model(model, {}), model)

    def test_grpo_lora_is_applied_only_when_enabled_and_preserves_model_path(self) -> None:
        base_model = FakeModel("./models/main_image_structured_sft_merged")
        wrapped_model = FakeModel("peft-wrapper")
        with patch("sft.train_schema_sft.prepare_lora_model", return_value=wrapped_model) as prepare:
            result = maybe_prepare_grpo_lora_model(base_model, {"lora": {"enabled": True}})

        self.assertIs(result, wrapped_model)
        prepare.assert_called_once_with(base_model, {"lora": {"enabled": True}})
        self.assertEqual(result.name_or_path, "./models/main_image_structured_sft_merged")
        self.assertEqual(result.config._name_or_path, "./models/main_image_structured_sft_merged")
        self.assertEqual(result.config.name_or_path, "./models/main_image_structured_sft_merged")


if __name__ == "__main__":
    unittest.main()
