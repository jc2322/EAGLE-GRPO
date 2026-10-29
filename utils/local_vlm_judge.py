from __future__ import annotations

import copy
import threading
import warnings
from pathlib import Path
from typing import Any

import torch

from utils.local_image_edit import load_image_ref
from utils.modeling import _from_pretrained_with_mistral_regex


_JUDGES: dict[tuple[str, str, str, str], "LocalTransformersVlmJudge"] = {}
_CACHE_LOCK = threading.Lock()


def judge_json(messages: list[dict[str, Any]], config: dict[str, Any], *, max_new_tokens: int) -> str:
    judge = load_judge(config)
    return judge.generate(messages, max_new_tokens=max_new_tokens)


def load_judge(config: dict[str, Any]) -> "LocalTransformersVlmJudge":
    model_path = resolve_model_path(config)
    dtype_name = str(config.get("torch_dtype") or "bfloat16")
    device_map = str(config.get("device_map") or "auto")
    attn = str(config.get("attn_implementation") or "")
    key = (str(model_path), dtype_name, device_map, attn)
    with _CACHE_LOCK:
        if key not in _JUDGES:
            _JUDGES[key] = LocalTransformersVlmJudge(config, model_path)
        return _JUDGES[key]


def resolve_model_path(config: dict[str, Any]) -> Path:
    configured = str(config.get("model_path") or config.get("model_name_or_path") or "").strip()
    if configured:
        return Path(configured).expanduser()
    model = str(config.get("model") or "").strip()
    if model == "Qwen3-VL-32B-Instruct":
        return Path("Qwen/Qwen3-VL-32B-Instruct")
    path = Path(model).expanduser()
    if path.exists():
        return path
    raise ValueError(f"Set judge model_path for local_transformers backend: {model}")


def resolve_torch_dtype(name: str) -> Any:
    normalized = str(name or "").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    if normalized in {"auto", ""}:
        return "auto"
    raise ValueError(f"Unsupported torch dtype for local VLM judge: {name}")


class LocalTransformersVlmJudge:
    def __init__(self, config: dict[str, Any], model_path: Path) -> None:
        from transformers import AutoProcessor

        self.config = dict(config)
        self.model_path = model_path
        self.processor = _from_pretrained_with_mistral_regex(
            AutoProcessor,
            str(model_path),
            trust_remote_code=bool(config.get("trust_remote_code", True)),
        )
        self.model = self._load_model(model_path, config)
        self.lock = threading.Lock()

    def _load_model(self, model_path: Path, config: dict[str, Any]) -> Any:
        import transformers

        kwargs: dict[str, Any] = {
            "trust_remote_code": bool(config.get("trust_remote_code", True)),
            "torch_dtype": resolve_torch_dtype(str(config.get("torch_dtype") or "bfloat16")),
            "low_cpu_mem_usage": True,
        }
        if config.get("attn_implementation"):
            kwargs["attn_implementation"] = config["attn_implementation"]
        device_map = config.get("device_map", "auto")
        if device_map not in {None, False, "false", "none", "null"}:
            kwargs["device_map"] = device_map

        errors: list[str] = []
        for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
            model_class = getattr(transformers, class_name, None)
            if model_class is None:
                continue
            try:
                model = model_class.from_pretrained(str(model_path), **kwargs)
                break
            except Exception as exc:  # pragma: no cover - model/runtime dependent
                errors.append(f"{class_name}: {exc}")
        else:
            raise RuntimeError("Could not load local VLM judge:\n" + "\n".join(errors))

        if not hasattr(model, "hf_device_map") and torch.cuda.is_available() and bool(config.get("move_to_cuda", True)):
            device = str(config.get("device") or "cuda")
            model = model.to(device)
        generation_config = deterministic_generation_config(model)
        if generation_config is not None:
            model.generation_config = generation_config
        return model.eval()

    def generate(self, messages: list[dict[str, Any]], *, max_new_tokens: int) -> str:
        normalized_messages = normalize_messages(messages, timeout=float(self.config.get("image_download_timeout", 60) or 60))
        text = self.processor.apply_chat_template(normalized_messages, tokenize=False, add_generation_prompt=True)
        images = extract_images(normalized_messages)
        inputs = self.processor(
            text=[text],
            images=images or None,
            padding=True,
            return_tensors="pt",
        )
        inputs = move_inputs_to_model(inputs, self.model)
        generation_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(self.config.get("do_sample", False)),
        }
        if generation_kwargs["do_sample"]:
            generation_kwargs["temperature"] = float(self.config.get("temperature", 0.1) or 0.1)
            generation_kwargs["top_p"] = float(self.config.get("top_p", 0.9) or 0.9)
        else:
            generation_config = deterministic_generation_config(self.model)
            if generation_config is not None:
                generation_kwargs["generation_config"] = generation_config
        with self.lock, torch.inference_mode():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"`generation_config` default values have been modified.*")
                warnings.filterwarnings("ignore", message=r"The following generation flags are not valid.*")
                generated = self.model.generate(**inputs, **generation_kwargs)
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            generated = generated[:, input_ids.shape[1] :]
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return str(decoded[0] if decoded else "")


def normalize_messages(messages: list[dict[str, Any]], *, timeout: float) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            normalized.append(dict(message))
            continue
        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "").strip()
            if part_type == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    image_url = image_url.get("url")
                parts.append({"type": "image", "image": load_image_ref(str(image_url or ""), timeout=timeout)})
            elif part_type == "image":
                image_value = part.get("image")
                if isinstance(image_value, str):
                    image_value = load_image_ref(image_value, timeout=timeout)
                parts.append({"type": "image", "image": image_value})
            else:
                parts.append(dict(part))
        normalized.append({**message, "content": parts})
    return normalized


def extract_images(messages: list[dict[str, Any]]) -> list[Any]:
    images: list[Any] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image" and part.get("image") is not None:
                images.append(part["image"])
    return images


def move_inputs_to_model(inputs: Any, model: Any) -> Any:
    if hasattr(inputs, "to"):
        try:
            return inputs.to(model.device)
        except Exception:
            pass
    device = getattr(model, "device", None)
    if device is None:
        return inputs
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in dict(inputs).items()}


def deterministic_generation_config(model: Any) -> Any:
    generation_config = copy.deepcopy(getattr(model, "generation_config", None))
    if generation_config is None:
        return None
    generation_config.do_sample = False
    for name, value in {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "min_p": None,
        "typical_p": 1.0,
    }.items():
        if hasattr(generation_config, name):
            setattr(generation_config, name, value)
    return generation_config
