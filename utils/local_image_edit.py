from __future__ import annotations

import base64
import io
import logging
import re
import sys
import threading
from pathlib import Path
from typing import Any

import requests
from PIL import Image


LOG = logging.getLogger(__name__)


LOCAL_IMAGE_MODEL_PATHS = {
    "Qwen-Image-Edit-2511": "Qwen/Qwen-Image-Edit-2511",
    "FLUX.2-klein-9B": "black-forest-labs/FLUX.2-klein-9B",
}

LOCAL_BACKENDS = {"local", "local_diffusers", "diffusers"}
QWEN_PROVIDERS = {"qwen", "qwen_image_edit", "qwen_image_edit_2511", "qwen_image_edit_plus"}
FLUX2_PROVIDERS = {"flux2", "flux2_klein", "flux_2_klein", "flux.2-klein-9b"}

_PIPELINES: dict[tuple[str, str, str, str], Any] = {}
_CACHE_LOCK = threading.Lock()
_GENERATION_LOCKS: dict[tuple[str, str, str, str], threading.Lock] = {}


def image_to_data_url(image: Image.Image, output_format: str = "png") -> str:
    fmt = _output_format(output_format)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format=fmt)
    mime = "image/jpeg" if fmt == "JPEG" else f"image/{fmt.lower()}"
    return "data:" + mime + ";base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def request_local_image_edit(prompt: str, image_url: str, config: dict[str, Any]) -> dict[str, Any]:
    image = load_image_ref(image_url, timeout=float(config.get("image_download_timeout", 60) or 60))
    provider = infer_provider(config)
    if provider == "qwen_image_edit_2511":
        output = run_qwen_image_edit(prompt, image, config)
    elif provider == "flux2_klein":
        output = run_flux2_klein(prompt, image, config)
    else:
        raise ValueError(f"Unsupported local image edit provider: {provider}")
    output_format = str(config.get("output_format") or "png")
    return {
        "data": [{"url": image_to_data_url(output, output_format)}],
        "usage": {
            "backend": "local_diffusers",
            "provider": provider,
            "model": str(config.get("model") or ""),
            "model_path": str(resolve_model_path(config)),
            "num_inference_steps": int(config.get("num_inference_steps", 0) or 0),
        },
    }


def load_image_ref(image_ref: str, *, timeout: float = 60.0) -> Image.Image:
    value = str(image_ref or "").strip()
    if not value:
        raise ValueError("missing image reference")
    if value.startswith("data:image/"):
        _, payload = value.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    if value.startswith(("http://", "https://")):
        response = requests.get(value, timeout=timeout)
        if not response.ok:
            raise RuntimeError(f"source image download failed: status={response.status_code} url={value}")
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    if value.startswith("file://"):
        value = value[len("file://") :]
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"source image not found: {path}")
    return Image.open(path).convert("RGB")


def infer_provider(config: dict[str, Any]) -> str:
    configured = str(config.get("provider") or config.get("pipeline") or config.get("backend_provider") or "").strip()
    normalized = configured.lower().replace("-", "_")
    if normalized in QWEN_PROVIDERS:
        return "qwen_image_edit_2511"
    if normalized in FLUX2_PROVIDERS:
        return "flux2_klein"
    text = f"{config.get('model', '')} {config.get('model_path', '')}".lower()
    if "flux.2" in text or "flux2" in text or "klein" in text:
        return "flux2_klein"
    if "qwen" in text:
        return "qwen_image_edit_2511"
    raise ValueError("Set image_edit.provider to qwen_image_edit_2511 or flux2_klein for local_diffusers backend.")


def resolve_model_path(config: dict[str, Any]) -> Path:
    configured = str(config.get("model_path") or "").strip()
    if configured:
        return Path(configured).expanduser()
    model = str(config.get("model") or "").strip()
    if model in LOCAL_IMAGE_MODEL_PATHS:
        return Path(LOCAL_IMAGE_MODEL_PATHS[model])
    path = Path(model).expanduser()
    if path.exists():
        return path
    raise ValueError(f"Set image_edit.model_path for local image model: {model}")


def disable_xformers_for_diffusers() -> None:
    """Force diffusers to skip optional xformers imports for local image edit.

    Some job images expose a global xformers build compiled against a different
    torch/triton stack. Diffusers treats xformers as optional, but importing it
    during FLUX2 pipeline import can fail before we ever instantiate the model.
    """
    for module_name in list(sys.modules):
        if module_name == "xformers" or module_name.startswith("xformers."):
            sys.modules.pop(module_name, None)
    try:
        import diffusers.utils.import_utils as import_utils

        import_utils._xformers_available = False
        import_utils._xformers_version = None
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Could not patch diffusers xformers availability: %s", exc)
    attention_processor = sys.modules.get("diffusers.models.attention_processor")
    if attention_processor is not None:
        setattr(attention_processor, "xformers", None)


def run_qwen_image_edit(prompt: str, image: Image.Image, config: dict[str, Any]) -> Image.Image:
    pipeline = load_pipeline("qwen_image_edit_2511", config)
    source = prepare_qwen_source_image(image, config)
    seed = int(config.get("seed", 0) or 0)
    import torch

    inputs: dict[str, Any] = {
        "image": [source],
        "prompt": str(prompt),
        "generator": torch.Generator(device="cpu").manual_seed(seed),
        "true_cfg_scale": float(config.get("true_cfg_scale", 4.0) or 4.0),
        "negative_prompt": str(config.get("negative_prompt", " ")),
        "num_inference_steps": int(config.get("num_inference_steps", 40) or 40),
        "guidance_scale": float(config.get("guidance_scale", 1.0) or 1.0),
        "num_images_per_prompt": int(config.get("num_images_per_prompt", 1) or 1),
    }
    return call_pipeline(pipeline, inputs)


def run_flux2_klein(prompt: str, image: Image.Image, config: dict[str, Any]) -> Image.Image:
    pipeline = load_pipeline("flux2_klein", config)
    multiple = max(1, int(config.get("size_multiple", 16) or 16))
    target_size = flux_target_size(image, config)
    width, height = align_size(target_size, multiple)
    seed = str(config.get("seed", 0)).strip().lower()
    import torch

    inputs: dict[str, Any] = {
        "prompt": str(prompt),
        "image": image.convert("RGB"),
        "width": width,
        "height": height,
        "guidance_scale": float(config.get("guidance_scale", 1.0) or 1.0),
        "num_inference_steps": int(config.get("num_inference_steps", 4) or 4),
    }
    if seed not in {"", "none", "null", "random"}:
        try:
            inputs["generator"] = torch.Generator(device=str(config.get("device") or "cuda")).manual_seed(int(seed))
        except Exception:
            inputs["generator"] = torch.manual_seed(int(seed))
    output = call_pipeline(pipeline, inputs)
    return enforce_output_size(output, target_size)


def load_pipeline(kind: str, config: dict[str, Any]) -> Any:
    model_path = resolve_model_path(config)
    device = str(config.get("device") or "cuda").strip()
    dtype_name = str(config.get("torch_dtype") or "bfloat16").strip()
    key = (kind, str(model_path), device, dtype_name)
    with _CACHE_LOCK:
        if key in _PIPELINES:
            return _PIPELINES[key]
        pipeline = create_pipeline(kind, model_path, device, dtype_name, config)
        _PIPELINES[key] = pipeline
        _GENERATION_LOCKS.setdefault(key, threading.Lock())
        return pipeline


def create_pipeline(kind: str, model_path: Path, device: str, dtype_name: str, config: dict[str, Any]) -> Any:
    import torch

    if bool(config.get("disable_xformers", False)):
        disable_xformers_for_diffusers()
    torch_dtype = resolve_torch_dtype(torch, dtype_name)
    if kind == "qwen_image_edit_2511":
        try:
            from diffusers import QwenImageEditPlusPipeline
        except ImportError as exc:
            raise ImportError("Qwen local image edit requires diffusers with QwenImageEditPlusPipeline.") from exc
        pipeline = QwenImageEditPlusPipeline.from_pretrained(str(model_path), torch_dtype=torch_dtype)
    elif kind == "flux2_klein":
        try:
            from diffusers import Flux2KleinPipeline
        except ImportError as exc:
            raise ImportError("FLUX.2-klein local image edit requires diffusers with Flux2KleinPipeline.") from exc
        pipeline = Flux2KleinPipeline.from_pretrained(str(model_path), torch_dtype=torch_dtype)
    else:
        raise ValueError(f"Unsupported local image edit pipeline: {kind}")

    if bool(config.get("enable_model_cpu_offload", False)):
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=bool(config.get("disable_progress_bar", True)))
    return pipeline


def call_pipeline(pipeline: Any, inputs: dict[str, Any]) -> Image.Image:
    key = pipeline_cache_key(pipeline)
    lock = _GENERATION_LOCKS.get(key, threading.Lock())
    with lock:
        result = pipeline(**inputs)
    images = getattr(result, "images", None)
    if not images:
        raise ValueError("local image edit pipeline returned no images")
    return images[0].convert("RGB")


def pipeline_cache_key(pipeline: Any) -> tuple[str, str, str, str]:
    for key, value in _PIPELINES.items():
        if value is pipeline:
            return key
    return ("unknown", str(id(pipeline)), "", "")


def resolve_torch_dtype(torch: Any, name: str) -> Any:
    normalized = str(name or "").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype for local image edit: {name}")


def prepare_qwen_source_image(image: Image.Image, config: dict[str, Any]) -> Image.Image:
    size = parse_size(str(config.get("size") or ""))
    if not size or not bool(config.get("resize_input", True)):
        return image.convert("RGB")
    mode = str(config.get("resize_mode") or "pad").strip().lower()
    if mode == "stretch":
        return image.convert("RGB").resize(size, resample=resample_lanczos())
    if mode == "contain":
        from PIL import ImageOps

        return ImageOps.contain(image.convert("RGB"), size, method=resample_lanczos())
    if mode != "pad":
        raise ValueError(f"Unsupported local image resize_mode: {mode}")
    from PIL import ImageOps

    contained = ImageOps.contain(image.convert("RGB"), size, method=resample_lanczos())
    canvas = Image.new("RGB", size, parse_rgb_color(str(config.get("pad_color") or "white")))
    left = (size[0] - contained.size[0]) // 2
    top = (size[1] - contained.size[1]) // 2
    canvas.paste(contained, (left, top))
    return canvas


def align_size(size: tuple[int, int], multiple: int) -> tuple[int, int]:
    return tuple(max(multiple, int(round(value / multiple) * multiple)) for value in size)  # type: ignore[return-value]


def flux_target_size(image: Image.Image, config: dict[str, Any]) -> tuple[int, int]:
    return parse_size(str(config.get("size") or "")) or image.size


def enforce_output_size(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    if image.size == target_size:
        return image.convert("RGB")
    return image.convert("RGB").resize(target_size, resample=resample_lanczos())


def parse_size(value: str) -> tuple[int, int] | None:
    match = re.match(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$", str(value or ""))
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        return None
    return width, height


def parse_rgb_color(value: str) -> tuple[int, int, int]:
    normalized = str(value or "").strip().lower()
    if normalized in {"white", ""}:
        return (255, 255, 255)
    if normalized == "black":
        return (0, 0, 0)
    match = re.match(r"^#?([0-9a-f]{6})$", normalized)
    if match:
        text = match.group(1)
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    raise ValueError(f"Unsupported RGB color: {value}")


def resample_lanczos() -> Any:
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def _output_format(value: str) -> str:
    normalized = str(value or "png").strip().lower()
    if normalized in {"jpg", "jpeg"}:
        return "JPEG"
    if normalized == "webp":
        return "WEBP"
    return "PNG"
