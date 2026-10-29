from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import logging
import os
import shutil
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch

from utils.completion_schema import FINAL_PROMPT_TAGS, REASONING_TAGS
from utils.config import resolve_path, section
from utils.schema_token_training import apply_schema_token_rows


LOG = logging.getLogger(__name__)
FIX_MISTRAL_REGEX_KWARGS = {"fix_mistral_regex": True}


def torch_dtype(value: str | None):
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(str(value or "auto").lower(), "auto")



def _kbit_compute_dtype(model_cfg: dict[str, Any]):
    dtype = torch_dtype(model_cfg.get("torch_dtype"))
    return torch.bfloat16 if dtype == "auto" else dtype


def _align_output_head_dtype(model: Any, target_dtype: Any) -> None:
    if target_dtype in {None, "auto"}:
        return
    candidates: list[tuple[str, Any]] = []
    if hasattr(model, "get_output_embeddings"):
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None:
            candidates.append(("output_embeddings", output_embeddings))
    if hasattr(model, "named_modules"):
        candidates.extend((name, module) for name, module in model.named_modules() if name.endswith("lm_head"))

    seen: set[int] = set()
    for name, module in candidates:
        if id(module) in seen:
            continue
        seen.add(id(module))
        weight = getattr(module, "weight", None)
        if weight is None or not torch.is_floating_point(weight):
            continue
        if weight.dtype != target_dtype:
            old_dtype = weight.dtype
            module.to(dtype=target_dtype)
            LOG.info("Aligned %s dtype from %s to %s", name, old_dtype, target_dtype)
        _install_lm_head_dtype_hook(module, name)


def _install_lm_head_dtype_hook(module: Any, name: str) -> None:
    if getattr(module, "_main_image_grpo_dtype_hook", False):
        return
    weight = getattr(module, "weight", None)
    if weight is None or not torch.is_floating_point(weight):
        return

    def cast_hidden_states(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if not inputs:
            return inputs
        hidden_states = inputs[0]
        current_weight = getattr(_module, "weight", None)
        if (
            torch.is_tensor(hidden_states)
            and current_weight is not None
            and torch.is_floating_point(hidden_states)
            and torch.is_floating_point(current_weight)
            and hidden_states.dtype != current_weight.dtype
        ):
            return (hidden_states.to(dtype=current_weight.dtype), *inputs[1:])
        return inputs

    module.register_forward_pre_hook(cast_hidden_states)
    setattr(module, "_main_image_grpo_dtype_hook", True)
    LOG.info("Installed dtype-alignment hook on %s", name)


def _enable_input_require_grads(model: Any) -> None:
    """Keep gradients flowing through frozen input embeddings under checkpointing."""
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        LOG.info("Enabled input require grads for gradient checkpointing")
        return
    if not hasattr(model, "get_input_embeddings"):
        LOG.warning("Cannot enable input require grads: model has no get_input_embeddings")
        return
    embeddings = model.get_input_embeddings()
    if embeddings is None:
        LOG.warning("Cannot enable input require grads: input embeddings are unavailable")
        return
    if getattr(embeddings, "_main_image_grpo_input_grad_hook", False):
        return

    def make_outputs_require_grad(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if torch.is_tensor(output):
            output.requires_grad_(True)

    embeddings.register_forward_hook(make_outputs_require_grad)
    setattr(embeddings, "_main_image_grpo_input_grad_hook", True)
    LOG.info("Installed input embedding grad hook for gradient checkpointing")


def _dedupe(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for value in values:
        key = str(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _patch_tokenizer_config_for_mistral_regex(processor_path: str | Path) -> None:
    path = Path(processor_path)
    tokenizer_config = path / "tokenizer_config.json"
    if not tokenizer_config.exists():
        return
    try:
        config = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if config.get("fix_mistral_regex") is True:
        return
    config["fix_mistral_regex"] = True
    try:
        tokenizer_config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        LOG.info("Patched tokenizer_config fix_mistral_regex at %s", tokenizer_config)
    except OSError as exc:
        LOG.warning("Could not patch tokenizer_config fix_mistral_regex at %s: %s", tokenizer_config, exc)


def _from_pretrained_with_mistral_regex(loader: Any, path: str | Path, **kwargs: Any) -> Any:
    try:
        return loader.from_pretrained(str(path), **kwargs, **FIX_MISTRAL_REGEX_KWARGS)
    except TypeError as exc:
        if "fix_mistral_regex" not in str(exc):
            raise
        return loader.from_pretrained(str(path), **kwargs)


def _processor_path_with_tokenizer_compat(processor_path: str | Path, project_dir: Path) -> str:
    path = Path(processor_path)
    tokenizer_config = path / "tokenizer_config.json"
    if not tokenizer_config.exists():
        return str(processor_path)
    try:
        config = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return str(processor_path)
    extra = config.get("extra_special_tokens")
    if not isinstance(extra, list):
        return str(processor_path)

    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    compat_dir = project_dir / "work" / "processor_compat" / digest
    compat_dir.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() or child.name == "adapter_model.safetensors":
            continue
        target = compat_dir / child.name
        if child.name == "tokenizer_config.json":
            continue
        if not target.exists() or target.stat().st_mtime < child.stat().st_mtime:
            shutil.copy2(child, target)

    additional = config.get("additional_special_tokens")
    if not isinstance(additional, list):
        additional = []
    config["additional_special_tokens"] = _dedupe([*additional, *extra])
    config["fix_mistral_regex"] = True
    config.pop("extra_special_tokens", None)
    (compat_dir / "tokenizer_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOG.info("Using tokenizer compatibility copy at %s", compat_dir)
    return str(compat_dir)


def schema_tag_tokens() -> set[str]:
    tags = {"thinking", "reasoning", "think", "final_prompt", *REASONING_TAGS, *FINAL_PROMPT_TAGS}
    return {token for tag in tags for token in (f"<{tag}>", f"</{tag}>")}


def _strip_non_schema_specials(tokenizer: Any, text: str) -> str:
    removable = sorted(
        {str(token) for token in getattr(tokenizer, "all_special_tokens", []) if str(token) not in schema_tag_tokens()},
        key=len,
        reverse=True,
    )
    for token in removable:
        text = text.replace(token, "")
    return text


def decode_preserving_schema_tags(tokenizer: Any, token_ids: Any, **kwargs: Any) -> str:
    """Decode while preserving XML schema tags even when skipping specials.

    Schema tags are additional special tokens so generation can emit stable
    boundaries. Vanilla ``skip_special_tokens=True`` would also remove these
    tags, which breaks reward/schema parsing.
    """

    if not kwargs.get("skip_special_tokens", False):
        return tokenizer.decode(token_ids, **kwargs)
    patched_kwargs = dict(kwargs)
    patched_kwargs["skip_special_tokens"] = False
    return _strip_non_schema_specials(tokenizer, tokenizer.decode(token_ids, **patched_kwargs))


def patch_decode_to_preserve_schema_tags(tokenizer: Any) -> None:
    if getattr(tokenizer, "_main_image_grpo_preserve_schema_decode", False):
        return
    original_decode = tokenizer.decode
    original_batch_decode = tokenizer.batch_decode

    def decode_with_schema(*args: Any, **kwargs: Any) -> str:
        if not kwargs.get("skip_special_tokens", False):
            return original_decode(*args, **kwargs)
        patched_kwargs = dict(kwargs)
        patched_kwargs["skip_special_tokens"] = False
        return _strip_non_schema_specials(tokenizer, original_decode(*args, **patched_kwargs))

    def batch_decode_with_schema(*args: Any, **kwargs: Any) -> list[str]:
        if not kwargs.get("skip_special_tokens", False):
            return original_batch_decode(*args, **kwargs)
        patched_kwargs = dict(kwargs)
        patched_kwargs["skip_special_tokens"] = False
        return [_strip_non_schema_specials(tokenizer, item) for item in original_batch_decode(*args, **patched_kwargs)]

    tokenizer.decode = decode_with_schema
    tokenizer.batch_decode = batch_decode_with_schema
    setattr(tokenizer, "_main_image_grpo_preserve_schema_decode", True)
    LOG.info("Patched tokenizer decode to preserve schema tag special tokens")


def load_processor(config: dict[str, Any]):
    from transformers import AutoProcessor

    project_dir = Path(config["_project_dir"])
    model_cfg = section(config, "model")
    adapter_path = resolve_path(model_cfg.get("sft_adapter_path"), base_dir=project_dir)
    processor_path = model_cfg.get("processor_name_or_path")
    if not processor_path and adapter_path and adapter_path.exists():
        processor_path = str(adapter_path)
    if not processor_path:
        processor_path = model_cfg["model_name_or_path"]
    _patch_tokenizer_config_for_mistral_regex(processor_path)
    processor_path = _processor_path_with_tokenizer_compat(processor_path, project_dir)
    processor = _from_pretrained_with_mistral_regex(
        AutoProcessor,
        processor_path,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    tokenizer = getattr(processor, "tokenizer", None) or processor
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    patch_decode_to_preserve_schema_tags(tokenizer)
    if not hasattr(processor, "tokenizer"):
        with suppress(Exception):
            setattr(processor, "tokenizer", tokenizer)
    if hasattr(processor, "decode"):
        processor.decode = tokenizer.decode
    if hasattr(processor, "batch_decode"):
        processor.batch_decode = tokenizer.batch_decode
    return processor


def schema_token_rows_path(config: dict[str, Any]) -> Path | None:
    project_dir = Path(config["_project_dir"])
    model_cfg = section(config, "model")
    if not bool(model_cfg.get("apply_schema_token_rows", True)):
        return None
    explicit_path = resolve_path(model_cfg.get("schema_token_rows_path"), base_dir=project_dir)
    if explicit_path and explicit_path.exists():
        return explicit_path
    adapter_path = resolve_path(model_cfg.get("sft_adapter_path"), base_dir=project_dir)
    if adapter_path:
        candidate = adapter_path / "schema_token_rows.pt"
        if candidate.exists():
            return candidate
    return None


def _optional_bool(value: Any, *, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"auto", ""}:
        return default
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean or 'auto', got {value!r}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _path_stat_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "missing": True}
    return {
        "path": str(path.resolve()),
        "is_dir": path.is_dir(),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _source_model_fingerprint(model_name_or_path: str) -> dict[str, Any]:
    path = Path(str(model_name_or_path)).expanduser()
    if not path.exists():
        return {"model_id": str(model_name_or_path)}
    result: dict[str, Any] = {"path": str(path.resolve())}
    for filename in ("config.json", "generation_config.json", "preprocessor_config.json"):
        child = path / filename
        if child.exists():
            result[filename] = _path_stat_fingerprint(child)
    return result


def _adapter_fingerprint(config: dict[str, Any]) -> dict[str, Any] | None:
    project_dir = Path(config["_project_dir"])
    model_cfg = section(config, "model")
    adapter_path = resolve_path(model_cfg.get("sft_adapter_path"), base_dir=project_dir)
    if adapter_path is None:
        return None
    result: dict[str, Any] = {"path": str(adapter_path)}
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        child = adapter_path / filename
        if child.exists():
            result[filename] = _path_stat_fingerprint(child)
    return result


def _tokenizer_fingerprint(tokenizer: Any) -> dict[str, Any]:
    added_vocab = tokenizer.get_added_vocab() if hasattr(tokenizer, "get_added_vocab") else {}
    special_map = getattr(tokenizer, "special_tokens_map", {})
    return {
        "length": len(tokenizer),
        "added_vocab": sorted((str(token), int(token_id)) for token, token_id in dict(added_vocab).items()),
        "special_tokens_map": _jsonable(special_map),
    }


def _call_save_pretrained(obj: Any, path: Path, **kwargs: Any) -> None:
    save_pretrained = getattr(obj, "save_pretrained")
    try:
        signature = inspect.signature(save_pretrained)
    except (TypeError, ValueError):
        save_pretrained(path, **kwargs)
        return
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if accepts_kwargs:
        save_pretrained(path, **kwargs)
        return
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    save_pretrained(path, **supported)


def _set_model_name_or_path(model: Any, path: str | Path) -> None:
    value = str(path)
    setattr(model, "name_or_path", value)
    config = getattr(model, "config", None)
    if config is not None:
        setattr(config, "_name_or_path", value)
        setattr(config, "name_or_path", value)


def _expanded_vllm_model_cache_path(
    config: dict[str, Any],
    *,
    tokenizer: Any,
    rows_path: Path | None,
    processor: Any | None,
) -> Path:
    project_dir = Path(config["_project_dir"])
    model_cfg = section(config, "model")
    cache_root = resolve_path(
        str(model_cfg.get("vllm_expanded_model_cache_dir", "./work/vllm_expanded_models")),
        base_dir=project_dir,
    )
    if cache_root is None:
        raise ValueError("model.vllm_expanded_model_cache_dir must not be null")
    payload = {
        "version": 2,
        "source_model": _source_model_fingerprint(str(model_cfg["model_name_or_path"])),
        "sft_adapter": _adapter_fingerprint(config),
        "processor_name_or_path": str(getattr(processor, "name_or_path", "")),
        "tokenizer": _tokenizer_fingerprint(tokenizer),
        "schema_token_rows": _path_stat_fingerprint(rows_path),
        "torch_dtype": str(model_cfg.get("torch_dtype", "auto")),
        "attn_implementation": str(model_cfg.get("attn_implementation", "")),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return cache_root / digest[:20]


def _peft_base_model_for_save(model: Any) -> Any:
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    base_model = getattr(model, "base_model", None)
    nested = getattr(base_model, "model", None)
    return nested or base_model or model


def _save_model_for_vllm_cache(model: Any, target: Path, model_cfg: dict[str, Any]) -> None:
    save_kwargs = {
        "safe_serialization": True,
        "max_shard_size": str(model_cfg.get("vllm_expanded_model_max_shard_size", "5GB")),
    }
    is_peft = hasattr(model, "peft_config") and hasattr(model, "merge_adapter") and hasattr(model, "unmerge_adapter")
    if not is_peft:
        _call_save_pretrained(model, target, **save_kwargs)
        return

    LOG.info("Saving merged PEFT policy model for vLLM cache")
    model.merge_adapter()
    try:
        _call_save_pretrained(_peft_base_model_for_save(model), target, **save_kwargs)
    finally:
        model.unmerge_adapter()


def prepare_vllm_model_name_or_path(
    config: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any | None,
    processor: Any | None = None,
    rows_path: Path | None = None,
    expanded_vocab_required: bool = False,
) -> Path | None:
    generation = section(config, "generation")
    if not bool(generation.get("use_vllm", False)):
        return None

    project_dir = Path(config["_project_dir"])
    model_cfg = section(config, "model")
    explicit_path = resolve_path(model_cfg.get("vllm_model_name_or_path"), base_dir=project_dir)
    if explicit_path is not None:
        _set_model_name_or_path(model, explicit_path)
        LOG.info("Using explicit vLLM model path %s", explicit_path)
        return explicit_path

    materialize = _optional_bool(model_cfg.get("vllm_materialize_expanded_model", "auto"), default=None)
    if not expanded_vocab_required and materialize is not True:
        return None
    if materialize is False:
        raise ValueError(
            "vLLM rollout needs a model path with the same expanded tokenizer vocab as training. "
            "Set model.vllm_materialize_expanded_model: true/auto or provide model.vllm_model_name_or_path."
        )
    if tokenizer is None:
        raise ValueError("Cannot prepare expanded vLLM model path without the training tokenizer")
    if model_cfg.get("load_in_4bit") or model_cfg.get("load_in_8bit"):
        raise ValueError(
            "Automatic expanded vLLM model materialization must run from an unquantized policy model. "
            "For QLoRA, prebuild a BF16 expanded model and set model.vllm_model_name_or_path."
        )
    if not hasattr(model, "save_pretrained"):
        raise TypeError("Policy model does not support save_pretrained; cannot materialize expanded vLLM model")

    target = _expanded_vllm_model_cache_path(config, tokenizer=tokenizer, rows_path=rows_path, processor=processor)
    marker = target / ".main_image_grpo_vllm_expanded_model_complete.json"
    if marker.exists():
        LOG.info("Reusing expanded vLLM model cache at %s", target)
    else:
        target.mkdir(parents=True, exist_ok=True)
        LOG.info("Saving expanded vLLM policy model cache to %s", target)
        _save_model_for_vllm_cache(model, target, model_cfg)
        if processor is not None and hasattr(processor, "save_pretrained"):
            _call_save_pretrained(processor, target)
        elif tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            _call_save_pretrained(tokenizer, target)
        marker.write_text(
            json.dumps(
                {
                    "source_model": str(model_cfg["model_name_or_path"]),
                    "sft_adapter_path": str(resolve_path(model_cfg.get("sft_adapter_path"), base_dir=project_dir))
                    if model_cfg.get("sft_adapter_path")
                    else None,
                    "tokenizer_length": len(tokenizer),
                    "schema_token_rows_path": str(rows_path) if rows_path else None,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    _set_model_name_or_path(model, target)
    if str(generation.get("vllm_mode", "colocate")).lower() == "server":
        LOG.warning("TRL vLLM server mode must be launched with the same expanded model path: %s", target)
    return target


def load_policy_model(config: dict[str, Any], tokenizer: Any | None = None, processor: Any | None = None):
    import transformers
    from transformers import BitsAndBytesConfig

    model_cfg = section(config, "model")
    _patch_tokenizer_config_for_mistral_regex(model_cfg["model_name_or_path"])
    train_cfg = section(config, "training")
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "low_cpu_mem_usage": True,
        "torch_dtype": torch_dtype(model_cfg.get("torch_dtype")),
    }
    if model_cfg.get("attn_implementation"):
        kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    if model_cfg.get("load_in_4bit"):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = model_cfg.get("device_map", "auto")
    elif model_cfg.get("device_map") not in {None, "none", "null", False}:
        if str(model_cfg.get("device_map")).lower() != "auto" or importlib.util.find_spec("accelerate"):
            kwargs["device_map"] = model_cfg["device_map"]

    LOG.info("Loading policy model from %s", model_cfg["model_name_or_path"])
    errors: list[str] = []
    for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
        model_class = getattr(transformers, class_name, None)
        if model_class is None:
            continue
        try:
            LOG.info("Trying policy model loader %s", class_name)
            model = model_class.from_pretrained(model_cfg["model_name_or_path"], **kwargs)
            LOG.info("Loaded policy model with %s", class_name)
            break
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            errors.append(f"{class_name}: {exc}")
    else:
        raise RuntimeError("Could not load policy model:\n" + "\n".join(errors))

    project_dir = Path(config["_project_dir"])
    vocab_was_resized = False
    if tokenizer is not None:
        target_vocab_size = len(tokenizer)
        current_vocab_size = int(model.get_input_embeddings().weight.shape[0])
        if target_vocab_size != current_vocab_size:
            model.resize_token_embeddings(target_vocab_size)
            vocab_was_resized = True
    rows_path = schema_token_rows_path(config)
    if rows_path:
        if tokenizer is None:
            LOG.warning("schema_token_rows_path is set but no tokenizer was passed; assuming model vocab is already resized")
        apply_schema_token_rows(model, rows_path)
    adapter_path = resolve_path(model_cfg.get("sft_adapter_path"), base_dir=project_dir)
    if adapter_path:
        from peft import PeftModel, prepare_model_for_kbit_training

        LOG.info("Loading trainable SFT adapter from %s", adapter_path)
        if model_cfg.get("load_in_4bit") or model_cfg.get("load_in_8bit"):
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
        LOG.info("Loaded trainable SFT adapter")
        if bool(train_cfg.get("gradient_checkpointing", True)):
            _enable_input_require_grads(model)
        if model_cfg.get("load_in_4bit") or model_cfg.get("load_in_8bit"):
            _align_output_head_dtype(model, _kbit_compute_dtype(model_cfg))
    prepare_vllm_model_name_or_path(
        config,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        rows_path=rows_path,
        expanded_vocab_required=vocab_was_resized or bool(rows_path),
    )
    if not hasattr(model, "hf_device_map") and torch.cuda.is_available() and bool(model_cfg.get("move_to_cuda", True)):
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        device = f"cuda:{local_rank}" if local_rank >= 0 else "cuda"
        LOG.info("Moving policy model to %s", device)
        model = model.to(device)
        LOG.info("Policy model is on %s", device)
    return model
