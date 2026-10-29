from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from utils.modeling import prepare_vllm_model_name_or_path, schema_token_rows_path


class FakeTokenizer:
    special_tokens_map = {"additional_special_tokens": ["<final_prompt>", "</final_prompt>"]}

    def __len__(self) -> int:
        return 12

    def get_added_vocab(self) -> dict[str, int]:
        return {"<final_prompt>": 10, "</final_prompt>": 11}

    def save_pretrained(self, path: str | Path) -> None:
        Path(path, "tokenizer_config.json").write_text("{}\n", encoding="utf-8")


class FakeProcessor:
    name_or_path = "fake-adapter-processor"

    def save_pretrained(self, path: str | Path) -> None:
        Path(path, "processor_config.json").write_text("{}\n", encoding="utf-8")


class FakeModel:
    def __init__(self) -> None:
        self.name_or_path = "base"
        self.config = SimpleNamespace(_name_or_path="base")
        self.save_calls: list[tuple[Path, dict[str, object]]] = []

    def save_pretrained(self, path: str | Path, *, safe_serialization: bool, max_shard_size: str) -> None:
        target = Path(path)
        self.save_calls.append((target, {"safe_serialization": safe_serialization, "max_shard_size": max_shard_size}))
        target.joinpath("config.json").write_text("{}\n", encoding="utf-8")
        target.joinpath("model.safetensors").write_text("fake\n", encoding="utf-8")


class FakePeftModel:
    peft_config = {"default": object()}

    def __init__(self) -> None:
        self.name_or_path = "peft"
        self.config = SimpleNamespace(_name_or_path="peft")
        self.base_model = FakeModel()
        self.events: list[str] = []

    def merge_adapter(self) -> None:
        self.events.append("merge")

    def unmerge_adapter(self) -> None:
        self.events.append("unmerge")

    def get_base_model(self) -> FakeModel:
        return self.base_model

    def save_pretrained(self, *_args, **_kwargs) -> None:
        raise AssertionError("vLLM cache should save the merged base model, not the PEFT adapter")


class ModelingConfigTest(unittest.TestCase):
    def test_schema_token_rows_can_be_disabled_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            adapter.mkdir()
            (adapter / "schema_token_rows.pt").write_bytes(b"placeholder")
            config = {
                "_project_dir": temp_dir,
                "model": {
                    "sft_adapter_path": str(adapter),
                    "apply_schema_token_rows": False,
                },
            }

            self.assertIsNone(schema_token_rows_path(config))

    def test_schema_token_rows_default_to_adapter_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            adapter.mkdir()
            rows_path = adapter / "schema_token_rows.pt"
            rows_path.write_bytes(b"placeholder")
            config = {"_project_dir": temp_dir, "model": {"sft_adapter_path": str(adapter)}}

            self.assertEqual(schema_token_rows_path(config), rows_path)

    def test_prepare_vllm_model_name_or_path_materializes_expanded_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir) / "base"
            base.mkdir()
            (base / "config.json").write_text("{}\n", encoding="utf-8")
            rows_path = Path(temp_dir) / "schema_token_rows.pt"
            rows_path.write_bytes(b"rows")
            config = {
                "_project_dir": temp_dir,
                "generation": {"use_vllm": True, "vllm_mode": "colocate"},
                "model": {
                    "model_name_or_path": str(base),
                    "torch_dtype": "bfloat16",
                    "attn_implementation": "sdpa",
                    "vllm_materialize_expanded_model": True,
                    "vllm_expanded_model_cache_dir": "./cache/vllm",
                    "vllm_expanded_model_max_shard_size": "2GB",
                },
            }
            model = FakeModel()

            result = prepare_vllm_model_name_or_path(
                config,
                model=model,
                tokenizer=FakeTokenizer(),
                processor=FakeProcessor(),
                rows_path=rows_path,
                expanded_vocab_required=True,
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result.is_dir())
            self.assertEqual(result.parent, Path(temp_dir, "cache", "vllm"))
            self.assertEqual(model.name_or_path, str(result))
            self.assertEqual(model.config._name_or_path, str(result))
            self.assertEqual(model.save_calls[0][1], {"safe_serialization": True, "max_shard_size": "2GB"})
            self.assertTrue((result / "model.safetensors").exists())
            self.assertTrue((result / "processor_config.json").exists())
            marker = json.loads((result / ".main_image_grpo_vllm_expanded_model_complete.json").read_text())
            self.assertEqual(marker["tokenizer_length"], 12)
            self.assertEqual(marker["schema_token_rows_path"], str(rows_path))

    def test_prepare_vllm_model_name_or_path_saves_merged_peft_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir) / "base"
            base.mkdir()
            adapter = Path(temp_dir) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            rows_path = adapter / "schema_token_rows.pt"
            rows_path.write_bytes(b"rows")
            config = {
                "_project_dir": temp_dir,
                "generation": {"use_vllm": True},
                "model": {
                    "model_name_or_path": str(base),
                    "sft_adapter_path": str(adapter),
                    "vllm_materialize_expanded_model": True,
                    "vllm_expanded_model_cache_dir": "./cache/vllm",
                },
            }
            model = FakePeftModel()

            result = prepare_vllm_model_name_or_path(
                config,
                model=model,
                tokenizer=FakeTokenizer(),
                processor=FakeProcessor(),
                rows_path=rows_path,
                expanded_vocab_required=True,
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(model.events, ["merge", "unmerge"])
            self.assertEqual(model.base_model.save_calls[0][0], result)
            self.assertEqual(model.name_or_path, str(result))
            marker = json.loads((result / ".main_image_grpo_vllm_expanded_model_complete.json").read_text())
            self.assertEqual(marker["sft_adapter_path"], str(adapter.resolve()))

    def test_prepare_vllm_model_name_or_path_reuses_complete_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir) / "base"
            base.mkdir()
            rows_path = Path(temp_dir) / "schema_token_rows.pt"
            rows_path.write_bytes(b"rows")
            config = {
                "_project_dir": temp_dir,
                "generation": {"use_vllm": True},
                "model": {
                    "model_name_or_path": str(base),
                    "vllm_materialize_expanded_model": True,
                    "vllm_expanded_model_cache_dir": "./cache/vllm",
                },
            }
            first_model = FakeModel()
            first = prepare_vllm_model_name_or_path(
                config,
                model=first_model,
                tokenizer=FakeTokenizer(),
                processor=FakeProcessor(),
                rows_path=rows_path,
                expanded_vocab_required=True,
            )
            second_model = FakeModel()
            second = prepare_vllm_model_name_or_path(
                config,
                model=second_model,
                tokenizer=FakeTokenizer(),
                processor=FakeProcessor(),
                rows_path=rows_path,
                expanded_vocab_required=True,
            )

            self.assertEqual(first, second)
            self.assertEqual(second_model.save_calls, [])
            self.assertEqual(second_model.name_or_path, str(second))

    def test_prepare_vllm_model_name_or_path_uses_explicit_path_without_saving(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            explicit = Path(temp_dir) / "expanded"
            config = {
                "_project_dir": temp_dir,
                "generation": {"use_vllm": True},
                "model": {
                    "model_name_or_path": "base",
                    "vllm_model_name_or_path": str(explicit),
                },
            }
            model = FakeModel()

            result = prepare_vllm_model_name_or_path(
                config,
                model=model,
                tokenizer=FakeTokenizer(),
                expanded_vocab_required=True,
            )

            self.assertEqual(result, explicit.resolve())
            self.assertEqual(model.name_or_path, str(explicit.resolve()))
            self.assertEqual(model.save_calls, [])

    def test_prepare_vllm_model_name_or_path_rejects_disabled_materialization_when_expanded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "_project_dir": temp_dir,
                "generation": {"use_vllm": True},
                "model": {
                    "model_name_or_path": "base",
                    "vllm_materialize_expanded_model": False,
                },
            }

            with self.assertRaisesRegex(ValueError, "same expanded tokenizer vocab"):
                prepare_vllm_model_name_or_path(
                    config,
                    model=FakeModel(),
                    tokenizer=FakeTokenizer(),
                    expanded_vocab_required=True,
                )


if __name__ == "__main__":
    unittest.main()
