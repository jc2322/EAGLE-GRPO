from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from utils.config import load_config, resolve_path, section
from utils.modeling import _processor_path_with_tokenizer_compat, torch_dtype
from utils.schema_token_training import apply_schema_token_rows


LOG = logging.getLogger(__name__)
AUXILIARY_FILES = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "chat_template.jinja",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the structured SFT LoRA adapter and schema token rows into a full Qwen-VL model."
    )
    parser.add_argument("--config", default="configs/sft.yaml", help="Project config used to resolve model defaults.")
    parser.add_argument("--base-model", default=None, help="Override model.model_name_or_path from the config.")
    parser.add_argument("--adapter-path", default="./outputs/main_image_structured_sft", help="SFT LoRA adapter directory.")
    parser.add_argument("--schema-token-rows", default=None, help="Override schema_token_rows.pt path.")
    parser.add_argument("--output-dir", default="./models/main_image_structured_sft_merged", help="Merged full model output directory.")
    parser.add_argument("--max-shard-size", default="5GB", help="Shard size passed to save_pretrained.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--torch-dtype", default=None, help="Override model.torch_dtype, e.g. bfloat16.")
    parser.add_argument("--attn-implementation", default=None, help="Override model.attn_implementation.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output directory.")
    return parser.parse_args()


def _resolve_required_path(value: str | Path | None, *, base_dir: Path, label: str) -> Path:
    path = resolve_path(value, base_dir=base_dir)
    if path is None or not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_expanded_tokenizer(adapter_path: Path, project_dir: Path, trust_remote_code: bool):
    from transformers import AutoTokenizer

    tokenizer_path = _processor_path_with_tokenizer_compat(adapter_path, project_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer, Path(tokenizer_path)


def load_base_model(base_model: Path | str, model_cfg: dict[str, Any], *, dtype: Any, trust_remote_code: bool, attn_implementation: str | None):
    import transformers

    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    errors: list[str] = []
    for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
        model_class = getattr(transformers, class_name, None)
        if model_class is None:
            continue
        try:
            LOG.info("Trying base model loader %s", class_name)
            return model_class.from_pretrained(str(base_model), **kwargs)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{class_name}: {exc}")
    raise RuntimeError("Could not load base model:\n" + "\n".join(errors))


def copy_auxiliary_files(*, base_model: Path, adapter_path: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for filename in AUXILIARY_FILES:
        for source_root in (adapter_path, base_model):
            source = source_root / filename
            if source.exists():
                shutil.copy2(source, output_dir / filename)
                copied.append(filename)
                break
    return copied


def write_metadata(
    *,
    output_dir: Path,
    base_model: Path | str,
    adapter_path: Path,
    schema_rows_path: Path,
    tokenizer: Any,
    copied_auxiliary_files: list[str],
    elapsed_seconds: float,
) -> None:
    metadata = {
        "base_model": str(base_model),
        "sft_adapter_path": str(adapter_path),
        "schema_token_rows_path": str(schema_rows_path),
        "tokenizer_length": len(tokenizer),
        "copied_auxiliary_files": copied_auxiliary_files,
        "elapsed_seconds": elapsed_seconds,
    }
    (output_dir / "merge_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def merge_sft_to_full_model(args: argparse.Namespace) -> Path:
    start = time.monotonic()
    config = load_config(args.config)
    project_dir = Path(config["_project_dir"])
    model_cfg = section(config, "model")

    base_model = Path(args.base_model or model_cfg["model_name_or_path"]).expanduser()
    if not base_model.is_absolute() and base_model.exists():
        base_model = base_model.resolve()
    adapter_path = _resolve_required_path(args.adapter_path, base_dir=project_dir, label="SFT adapter")
    schema_rows_path = _resolve_required_path(
        args.schema_token_rows or adapter_path / "schema_token_rows.pt",
        base_dir=project_dir,
        label="schema token rows",
    )
    output_dir = resolve_path(args.output_dir, base_dir=project_dir)
    if output_dir is None:
        raise ValueError("--output-dir must not be null")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trust_remote_code = bool(model_cfg.get("trust_remote_code", True)) if args.trust_remote_code is None else bool(args.trust_remote_code)
    dtype = torch_dtype(args.torch_dtype or model_cfg.get("torch_dtype", "bfloat16"))
    attn_implementation = args.attn_implementation or model_cfg.get("attn_implementation")

    LOG.info("Loading expanded tokenizer from %s", adapter_path)
    tokenizer, tokenizer_path = load_expanded_tokenizer(adapter_path, project_dir, trust_remote_code)
    LOG.info("Expanded tokenizer length: %d", len(tokenizer))

    LOG.info("Loading base model from %s", base_model)
    model = load_base_model(base_model, model_cfg, dtype=dtype, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation)
    current_vocab_size = int(model.get_input_embeddings().weight.shape[0])
    if current_vocab_size != len(tokenizer):
        LOG.info("Resizing token embeddings: %d -> %d", current_vocab_size, len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))

    LOG.info("Applying schema token rows from %s", schema_rows_path)
    apply_schema_token_rows(model, schema_rows_path)

    from peft import PeftModel

    LOG.info("Loading SFT LoRA adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    LOG.info("Merging adapter into base model")
    merged = model.merge_and_unload()
    merged.config.vocab_size = len(tokenizer)

    LOG.info("Saving merged full model to %s", output_dir)
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(output_dir)
    copied = copy_auxiliary_files(base_model=base_model, adapter_path=adapter_path, output_dir=output_dir)
    write_metadata(
        output_dir=output_dir,
        base_model=base_model,
        adapter_path=adapter_path,
        schema_rows_path=schema_rows_path,
        tokenizer=tokenizer,
        copied_auxiliary_files=copied,
        elapsed_seconds=time.monotonic() - start,
    )
    LOG.info("Merged model complete: %s", output_dir)
    LOG.info("Tokenizer source compatibility path: %s", tokenizer_path)
    return output_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    merge_sft_to_full_model(parse_args())


if __name__ == "__main__":
    main()
