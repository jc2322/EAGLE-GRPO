from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import requests

from rewards.providers import build_edit_body, request_image_edit
from rewards.scoring import ContextualImageScorer
from prompts.schema import compile_final_prompt_for_image
from utils.config import resolve_path, section
from rewards.online_component_reward import ComponentReward


LOG = logging.getLogger(__name__)


COT_JUDGE_PROMPT = """You are a strict reasoning judge for a Shopee product-main-image planning VLM.

You receive product metadata, the original image, the model's <thinking> text, and its structured <final_prompt>.
Judge only the reasoning quality. Do not reward the final image or the final prompt wording directly.

The thinking should be a concise planning summary. It should identify the product and useful source/listing facts,
select grounded low-risk facts for the main image, mention preservation and unsupported-claim risks, justify layout/style,
and explain why optional commercial elements should be used or left empty.

Score calibration:
- Use 5-6 as the default for reasoning that is product-relevant and usable but generic, checklist-like, or only mildly specific to the image/listing.
- Use 7-8 for concise reasoning that correctly identifies the product, selects grounded useful facts, drops risky/noisy facts, and gives product-specific layout/style rationale.
- Reserve 8+ for reasoning that is clearly product-specific, not just fluent: it must name concrete source-image/listing facts, justify why specific final-prompt choices are useful, and identify the most likely failure risks for this product.
- Reserve 9+ for rare reasoning that is concise, highly product-specific, clearly explains tradeoffs, anticipates final-prompt risks such as wrong count/variant, logo/package text corruption, unsupported claims, extra product, or misleading callouts, and justifies why optional commercial elements should be used or left empty.
- Use 3-4 for vague, mechanical, partially wrong, or weakly grounded reasoning.
- Use 0-2 for absent, unrelated, mostly copied, or seriously misleading reasoning.
Do not give high scores for generic preservation/design checklists that could apply to any product.
Do not give high scores just because the reasoning is fluent, concise, or plausible. A generic but well-written planning paragraph should remain 5-6.

Score from 0 to 10 using the full scale:
- source_understanding_score: correct product/category/source-image/listing understanding.
- grounded_fact_selection_score: selects useful, true, low-risk facts and drops noisy facts.
- risk_control_score: recognizes identity/count/variant/logo/text/unsupported-claim risks.
- design_rationale_score: explains layout/style/visual hierarchy choices that fit the product and commerce goal.
- efficiency_score: concise, not mechanical, not a checklist dump.
- non_redundancy_score: does not copy final-prompt sentences or tag-by-tag action details.

Hard fail only if the reasoning is absent, unrelated to the product, or mostly copied from the final prompt.

Return JSON only:
{
  "hard_fail": false,
  "source_understanding_score": 0,
  "grounded_fact_selection_score": 0,
  "risk_control_score": 0,
  "design_rationale_score": 0,
  "efficiency_score": 0,
  "non_redundancy_score": 0,
  "long_overlap_with_final_prompt": false,
  "reason": "short concrete evidence"
}
"""


IMAGE_JUDGE_PROMPT = """You are a strict VLM judge for Shopee product main images with advertising-aesthetic criteria.

You receive the original product image, one image-edit candidate, product metadata, and the final prompt used to create it.
Judge the generated image as a marketplace main image. The goal is not only preservation, but faithful, attention-grabbing,
commercially persuasive product advertising.

Evaluation philosophy:
- First check product truth: the real product identity, count, variant, package shape, source marks, and grounded claims must remain faithful.
- Then evaluate advertising aesthetics in three stages:
  1. Perceptual Attention: the image is clear, quickly processable, visually salient, and professionally rendered.
  2. Formal Interest: color, typography, layout, spacing, hierarchy, and background create coherent visual interest.
  3. Desire Impact: the image communicates useful selling points, attractive promotional/iconographic cues, and marketplace conversion appeal.
- Promotional iconography means non-text visual selling signs such as badges, chips, arrows, highlights, insets, swatches, benefit icons, or compact callout forms.
  Reward them only when they are grounded, readable, restrained, and helpful. Penalize generic clutter, fake badges, excessive stickers, or unsupported claims.
- Do not give high scores just because the product is preserved. A safe but plain catalog image should be mid-range, not excellent.
- Do not give high scores to a flashy image if it harms product truth, readability, or commercial trust.

Score calibration:
- Use 5-6 as the default score for images that are faithful and usable but visually ordinary, template-like, or only mildly improved from a catalog image.
- Use 6-7 for clean, polished, faithful GPT-Image-style outputs that are attractive and readable but use a conventional marketplace layout or generic design language.
- Use 7.5-8.5 only when the image adds clear commercial value beyond polish: product-specific selling points are immediately understandable, visual hierarchy improves comprehension, and grounded callouts/icons/insets/background choices increase click appeal.
- Reserve 9+ for rare images with premium advertising impact: faithful product truth, strong first-glance stopping power, refined layout, excellent lighting/rendering, product-specific creative selling-point presentation, and persuasive but trustworthy commercial appeal.
- Use 3-4 for weak images with noticeable clutter, poor readability, weak salience, awkward layout, obvious artifacts, or questionable advertising choices.
- Use 0-2 for unusable, severely misleading, severely damaged, or commercially invalid main images.
- Apply this calibration mainly to advertising-aesthetic and commercial-quality dimensions. Product_fidelity_score may be high when preservation is genuinely excellent, but product preservation alone should not make the overall image excellent.
- Avoid compressing most passing images into 8-9. For GPT-Image-2 outputs, generic high production quality should not by itself produce excellent scores.
- Do not score technical polish, text readability, clean background, or product fidelity as excellent advertising. These are prerequisites. The highest scores require distinctive, grounded, product-specific advertising communication.

Hard gates:
- product_identity_gate: fail if the product became a different item.
- product_count_gate: fail if visible product count is clearly wrong.
- variant_gate: fail if visible variants/colorways/package variants are clearly missing, duplicated, or invented in a way that changes the listing.
- logo_package_text_gate: fail only for severe logo/package/source-text corruption that changes product identity or key commercial info.
- unsupported_claim_gate: fail for clear unsupported, risky, medical, safety, certification, or authenticity claims introduced by the edit.
- severe_visual_damage_gate: fail for broken product geometry, unreadable mess, extreme artifacts, or unusable composition.

Positive scores from 0 to 10, using the full scale:
- product_fidelity_score: product identity, count, variants, shape, packaging, source marks, and grounded facts remain faithful.
- product_salience_score: product is prominent, clear, and immediately recognizable at thumbnail size.
- eye_catching_attention_score: strong first-glance stopping power without becoming noisy or misleading.
- creative_selling_point_score: grounded selling points are presented through useful text, iconography, insets, chips, or visual emphasis rather than generic decoration.
- color_harmony_score: palette has pleasant hue, saturation, and contrast, and remains compatible with the product/category.
- layout_hierarchy_score: composition, spacing, alignment, safe zones, and text/product hierarchy are professional and easy to scan.
- background_scene_fit_score: background or scene improves product appeal without distracting, confusing category, or hiding details.
- visual_communication_score: viewer quickly understands what is being sold, the core benefit, and why the image is worth clicking.
- text_readability_score: visible text/callouts are legible, concise, well placed, and not excessive; score low for tiny, garbled, or cluttered text.
- commercial_desire_score: likely click/conversion appeal as a Shopee main image, including persuasive but trustworthy advertising impact.
- professional_polish_score: rendering quality, lighting, shadows, edges, texture, perspective, and finish are clean and commercially credible.

Return JSON only:
{
  "hard_fail": false,
  "hard_fail_reason": "",
  "product_identity_gate": "pass",
  "product_count_gate": "pass",
  "variant_gate": "pass",
  "logo_package_text_gate": "pass",
  "unsupported_claim_gate": "pass",
  "severe_visual_damage_gate": "pass",
  "product_fidelity_score": 0,
  "product_salience_score": 0,
  "eye_catching_attention_score": 0,
  "creative_selling_point_score": 0,
  "color_harmony_score": 0,
  "layout_hierarchy_score": 0,
  "background_scene_fit_score": 0,
  "visual_communication_score": 0,
  "text_readability_score": 0,
  "commercial_desire_score": 0,
  "professional_polish_score": 0,
  "reason": "short concrete evidence"
}
"""


def clamp01_from_10(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(10.0, score)) / 10.0


def extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                value = json.loads(raw[start : index + 1])
                if not isinstance(value, dict):
                    raise ValueError("JSON value is not an object")
                return value
    raise ValueError("Unterminated JSON object")


def weighted_score(payload: dict[str, Any], weights: dict[str, Any], fields: list[tuple[str, str, float]]) -> float:
    total = sum(float(weights.get(key, default)) for key, _field, default in fields)
    if total <= 0:
        return 0.0
    return sum(float(weights.get(key, default)) * clamp01_from_10(payload.get(field)) for key, field, default in fields) / total


def image_data_url_from_response(body: dict[str, Any]) -> str:
    data = body.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("image edit response has no data images")
    first = data[0]
    if not isinstance(first, dict):
        raise ValueError("image edit response image is not an object")
    if first.get("url"):
        return str(first["url"])
    b64_json = str(first.get("b64_json") or "").strip()
    if not b64_json:
        raise ValueError("image edit response has neither url nor b64_json")
    raw = base64.b64decode(b64_json)
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def image_data_url_from_url(url: str, *, timeout: float = 60.0) -> str:
    response = requests.get(url, timeout=timeout)
    if not response.ok:
        raise RuntimeError(f"Failed to download judge image: status={response.status_code} url={url}")
    mime = str(response.headers.get("content-type") or "").split(";", 1)[0].strip()
    if not mime.startswith("image/"):
        mime = "image/jpeg"
    return "data:" + mime + ";base64," + base64.b64encode(response.content).decode("ascii")


def image_data_url_from_file(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    return "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def is_remote_url(value: str) -> bool:
    return str(value or "").startswith(("http://", "https://"))


def image_bytes_from_data_url(value: str) -> tuple[bytes, str]:
    match = re.match(r"^data:(image/[A-Za-z0-9.+-]+);base64,(.*)$", str(value or ""), flags=re.DOTALL)
    if match is None:
        raise ValueError("generated image is neither a URL nor a base64 image data URL")
    mime, payload = match.groups()
    return base64.b64decode(payload), mime.lower()


def verify_image_bytes(image_bytes: bytes) -> None:
    from PIL import Image

    with Image.open(BytesIO(image_bytes)) as image:
        image.verify()
    with Image.open(BytesIO(image_bytes)) as image:
        image.convert("RGB").load()


def image_extension_for_mime(mime: str) -> str:
    return {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
    }.get(str(mime or "").lower(), "png")


def image_data_url_from_reference(value: str, *, timeout: float = 60.0) -> str:
    value = str(value or "").strip()
    if value.startswith("data:image/"):
        return value
    if is_remote_url(value):
        return image_data_url_from_url(value, timeout=timeout)
    path = Path(value).expanduser()
    if path.exists():
        return image_data_url_from_file(path)
    raise ValueError("image reference must be an image URL, local image path, or data URL")


def load_image_from_reference(value: str, *, timeout: float = 60.0):
    from PIL import Image

    data_url = image_data_url_from_reference(value, timeout=timeout)
    raw, _mime = image_bytes_from_data_url(data_url)
    return Image.open(BytesIO(raw)).convert("RGB")


def resize_image_reference_for_judge(
    value: str,
    output_path: Path,
    *,
    timeout: float = 60.0,
    max_side: int = 1024,
) -> dict[str, Any]:
    from PIL import Image

    image = load_image_from_reference(value, timeout=timeout)
    original_size = [int(image.width), int(image.height)]
    resized = image.copy()
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    resized.thumbnail((max_side, max_side), resampling)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(output_path, format="PNG", optimize=True)
    return {
        "path": str(output_path),
        "data_url": image_data_url_from_file(output_path),
        "original_size": original_size,
        "resized_size": [int(resized.width), int(resized.height)],
    }


def comparison_image_data_url(
    original_ref: str,
    generated_ref: str,
    *,
    timeout: float = 60.0,
    max_side: int = 768,
) -> str:
    from PIL import Image, ImageDraw, ImageFont

    def fit(image: Image.Image) -> Image.Image:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (max_side, max_side), "white")
        x = (max_side - image.width) // 2
        y = (max_side - image.height) // 2
        canvas.paste(image, (x, y))
        return canvas

    label_h = 48
    gap = 16
    original = fit(load_image_from_reference(original_ref, timeout=timeout))
    generated = fit(load_image_from_reference(generated_ref, timeout=timeout))
    canvas = Image.new("RGB", (max_side * 2 + gap, max_side + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    draw.text((16, 12), "Original product image", fill=(20, 20, 20), font=font)
    draw.text((max_side + gap + 16, 12), "Generated candidate", fill=(20, 20, 20), font=font)
    canvas.paste(original, (0, label_h))
    canvas.paste(generated, (max_side + gap, label_h))
    out = BytesIO()
    canvas.save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def safe_filename_part(value: Any, default: str = "sample") -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or default)).strip("_")
    return safe or default


@dataclass(frozen=True)
class WorkerResult:
    request_id: str
    ok: bool
    value: Any = None
    error: str = ""


@dataclass(frozen=True)
class JudgeTask:
    task_id: str
    request_id: str
    kind: str
    request: dict[str, Any]
    image_payload: dict[str, Any] | None = None


def run_parallel(items: list[dict[str, Any]], worker_count: int, fn: Callable[[dict[str, Any]], Any]) -> dict[str, WorkerResult]:
    results: dict[str, WorkerResult] = {}
    if not items:
        return results
    with ThreadPoolExecutor(max_workers=max(1, int(worker_count))) as executor:
        future_to_item = {executor.submit(fn, item): item for item in items}
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            request_id = str(item["request_id"])
            try:
                results[request_id] = WorkerResult(request_id=request_id, ok=True, value=future.result())
            except Exception as exc:  # noqa: BLE001
                results[request_id] = WorkerResult(request_id=request_id, ok=False, error=f"{type(exc).__name__}: {exc}")
    return results


class DecoupledCotImageRewardStrategy:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.online_cfg = section(config, "online_reward")
        self.image_scorer = ContextualImageScorer.from_config(config)
        self._openai_compatible_clients: dict[tuple[str, str, float], Any] = {}
        self._judge_tmp_root: Path | None = None

    def score(self, requests: list[dict[str, Any]]) -> list[ComponentReward]:
        total_start = time.monotonic()
        for request in requests:
            self.image_scorer.weights(request.get("market"), request.get("category_l1"))
        rewards = [self._initial_reward(request) for request in requests]
        by_id = {reward.request_id: reward for reward in rewards}
        valid_requests = [request for request in requests if bool(request.get("schema_valid", False))]

        image_start = time.monotonic()
        image_results = run_parallel(valid_requests, int(self.online_cfg.get("image_worker_count", 5)), self._generate_image)
        LOG.info(
            "online_reward image_edit batch: requests=%d ok=%d seconds=%.3f",
            len(valid_requests),
            sum(1 for result in image_results.values() if result.ok),
            time.monotonic() - image_start,
        )
        judge_tasks: list[JudgeTask] = []
        for request_id, result in image_results.items():
            reward = by_id[request_id]
            if not result.ok:
                reward.active = False
                reward.drop_reason = f"image_pipeline_failed: {result.error}"
                reward.details["failure_policy"] = "remove_system_fail_after_retry"
                continue
            image_payload = dict(result.value)
            reward.details["image_judge"] = {key: value for key, value in image_payload.items() if not key.startswith("_")}
            request = image_payload.get("_request")
            if not isinstance(request, dict):
                reward.active = False
                reward.drop_reason = "image_pipeline_failed: missing request payload"
                continue
            if bool(self.online_cfg.get("enable_cot_reward", True)):
                judge_tasks.append(
                    JudgeTask(
                        task_id=f"{request_id}:cot",
                        request_id=request_id,
                        kind="cot",
                        request=request,
                    )
                )
            else:
                reward.details["cot_judge"] = {"skipped": True, "reason": "disabled_by_config"}
            judge_tasks.append(
                JudgeTask(
                    task_id=f"{request_id}:image",
                    request_id=request_id,
                    kind="image",
                    request=request,
                    image_payload=image_payload,
                )
            )

        judge_start = time.monotonic()
        judge_results = self._run_judge_tasks(judge_tasks)
        LOG.info(
            "online_reward judge batch: requests=%d cot=%d image=%d ok=%d seconds=%.3f",
            len(judge_tasks),
            sum(1 for task in judge_tasks if task.kind == "cot"),
            sum(1 for task in judge_tasks if task.kind == "image"),
            sum(1 for result in judge_results.values() if result.ok),
            time.monotonic() - judge_start,
        )
        cot_ok = 0
        image_ok = 0
        for task in judge_tasks:
            reward = by_id[task.request_id]
            result = judge_results.get(task.task_id)
            if result is None:
                reward.active = False
                reward.drop_reason = f"{task.kind}_judge_failed: missing judge result"
                reward.details["failure_policy"] = "remove_system_fail_after_retry"
                continue
            if task.kind == "cot":
                if not result.ok:
                    reward.active = False
                    reward.drop_reason = f"cot_judge_failed: {result.error}"
                    reward.details["failure_policy"] = "remove_system_fail_after_retry"
                    continue
                cot_payload = result.value
                reward.cot_reward = self._cot_reward(cot_payload)
                reward.details["cot_judge"] = cot_payload
                cot_ok += 1
                continue
            if not result.ok:
                reward.active = False
                reward.drop_reason = f"image_judge_failed: {result.error}"
                reward.details["failure_policy"] = "remove_system_fail_after_retry"
                continue
            image_payload = result.value
            reward.image_reward = self._image_reward(image_payload, task.request)
            reward.details["image_judge"] = image_payload
            if bool(image_payload.get("judge_failed", False)):
                reward.active = False
                reward.drop_reason = f"image_judge_failed: {image_payload.get('judge_error', '')}"
                reward.details["failure_policy"] = "remove_system_fail_after_retry"
                continue
            image_ok += 1
        for reward in rewards:
            if reward.active and reward.schema_valid and "image_judge" in reward.details:
                reward.scalar_reward = self._scalar_reward(reward.cot_reward, reward.image_reward)
        LOG.info(
            "online_reward cot_judge batch: requests=%d ok=%d seconds=%.3f",
            sum(1 for task in judge_tasks if task.kind == "cot"),
            cot_ok,
            time.monotonic() - judge_start,
        )
        LOG.info(
            "online_reward image_judge batch: requests=%d ok=%d seconds=%.3f",
            sum(1 for task in judge_tasks if task.kind == "image"),
            image_ok,
            time.monotonic() - judge_start,
        )
        LOG.info(
            "online_reward image_pipeline batch: requests=%d ok=%d seconds=%.3f",
            sum(1 for task in judge_tasks if task.kind == "image"),
            image_ok,
            time.monotonic() - image_start,
        )
        LOG.info(
            "online_reward score batch: requests=%d valid=%d active=%d seconds=%.3f",
            len(requests),
            len(valid_requests),
            sum(1 for reward in rewards if reward.active),
            time.monotonic() - total_start,
        )
        return rewards

    def _run_judge_tasks(self, tasks: list[JudgeTask]) -> dict[str, WorkerResult]:
        results: dict[str, WorkerResult] = {}
        if not tasks:
            return results
        worker_count = int(
            self.online_cfg.get(
                "judge_worker_count",
                max(int(self.online_cfg.get("cot_worker_count", 5)), int(self.online_cfg.get("image_worker_count", 5))),
            )
        )
        with ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
            future_to_task = {executor.submit(self._run_judge_task, task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    results[task.task_id] = WorkerResult(request_id=task.task_id, ok=True, value=future.result())
                except Exception as exc:  # noqa: BLE001
                    results[task.task_id] = WorkerResult(
                        request_id=task.task_id,
                        ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
        return results

    def _run_judge_task(self, task: JudgeTask) -> dict[str, Any]:
        if task.kind == "cot":
            return self._judge_cot(task.request)
        if task.kind != "image":
            raise ValueError(f"unknown judge task kind: {task.kind}")
        if task.image_payload is None:
            raise ValueError("image judge task missing image payload")
        return self._judge_image_from_generated(task.request, task.image_payload)

    def _initial_reward(self, request: dict[str, Any]) -> ComponentReward:
        schema_valid = bool(request.get("schema_valid", False))
        schema_invalid_reward = float(self.online_cfg.get("schema_invalid_reward", 0.0))
        return ComponentReward(
            request_id=str(request["request_id"]),
            row_key=str(request.get("row_key") or ""),
            sample_index=int(request.get("sample_index", 0)),
            raw_completion=str(request.get("raw_completion") or ""),
            schema_valid=schema_valid,
            schema_error=str(request.get("schema_error") or ""),
            active=True,
            cot_reward=schema_invalid_reward if not schema_valid else 0.0,
            image_reward=schema_invalid_reward if not schema_valid else 0.0,
            scalar_reward=schema_invalid_reward if not schema_valid else 0.0,
            format_invalid=not schema_valid,
            details={"completion_debug": request["completion_debug"]} if request.get("completion_debug") else {},
        )

    def _judge_cfg(self, kind: str) -> dict[str, Any]:
        default_cfg = section(self.online_cfg, "judge")
        section_name = "cot_judge" if kind == "cot" else "image_judge"
        specific_cfg = section(self.online_cfg, section_name)
        legacy_model_key = "cot_model" if kind == "cot" else "image_judge_model"
        merged = {**default_cfg, **specific_cfg}
        merged["backend"] = str(
            merged.get("backend") or self.online_cfg.get("judge_backend") or "api"
        ).strip()
        merged["model"] = str(
            merged.get("model")
            or self.online_cfg.get(legacy_model_key)
            or self.online_cfg.get("model")
            or "gpt-5.4"
        ).strip()
        return merged

    def _judge_model_name(self, kind: str) -> str:
        cfg = self._judge_cfg(kind)
        return str(cfg.get("model") or cfg.get("model_path") or "judge-model")

    def _judge_json(self, kind: str, *, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        cfg = self._judge_cfg(kind)
        backend = str(cfg.get("backend") or "api").strip().lower()
        if backend in {"api", "openai"}:
            return self._api_chat_json(cfg=cfg, messages=messages, max_tokens=max_tokens)
        if backend in {
            "vllm",
            "vllm_openai",
            "openai_compatible",
            "openai-compatible",
            "gemini",
            "gemini_openai",
            "gemini-openai",
        }:
            return self._openai_compatible_chat_json(cfg=cfg, messages=messages, max_tokens=max_tokens)
        if backend in {"local", "local_transformers", "transformers", "hf"}:
            from utils.local_vlm_judge import judge_json

            text = judge_json(messages, cfg, max_new_tokens=int(cfg.get("max_new_tokens") or max_tokens))
            return extract_json_object(text)
        raise ValueError(f"Unsupported {kind} judge backend: {backend}")

    def _chat_json(self, *, model: str, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        return self._api_chat_json(cfg={"model": model}, messages=messages, max_tokens=max_tokens)

    def _chat_response_text(self, response: Any) -> str:
        content = response.choices[0].message.content
        if isinstance(content, list):
            return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        return str(content or "")

    def _messages_with_optional_system(self, messages: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
        system_prompt = str(
            cfg.get("system_prompt")
            or self.online_cfg.get("judge_system_prompt")
            or self.online_cfg.get("system_prompt")
            or ""
        ).strip()
        if not system_prompt:
            return messages
        if messages and str(messages[0].get("role") or "").lower() == "system":
            return messages
        return [{"role": "system", "content": system_prompt}, *messages]

    def _retry_section(self) -> dict[str, Any]:
        retry_cfg = section(self.online_cfg, "retry")
        return retry_cfg if retry_cfg else {}

    def _retry_attempts(self, key: str, default: int = 3) -> int:
        retry_cfg = self._retry_section()
        return max(1, int(retry_cfg.get(f"{key}_attempts", retry_cfg.get("attempts", default)) or default))

    def _retry_backoff_seconds(self, key: str, default: float = 2.0) -> float:
        retry_cfg = self._retry_section()
        return max(0.0, float(retry_cfg.get(f"{key}_backoff_seconds", retry_cfg.get("backoff_seconds", default)) or default))

    def _is_non_retryable_image_edit_error(self, exc: BaseException) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        return (
            ("400" in text or "badrequest" in text)
            and any(token in text for token in ("safety", "content_policy", "moderation", "rejected", "policy"))
        )

    def _with_retry(
        self,
        key: str,
        fn: Callable[[], Any],
        *,
        default_attempts: int = 3,
        default_backoff: float = 2.0,
        non_retryable: Callable[[BaseException], bool] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        attempts = self._retry_attempts(key, default_attempts)
        backoff = self._retry_backoff_seconds(key, default_backoff)
        errors: list[str] = []
        for attempt in range(1, attempts + 1):
            try:
                value = fn()
                return value, {"attempts": attempt, "retry_errors": errors}
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                errors.append(error)
                if attempt >= attempts or (non_retryable is not None and non_retryable(exc)):
                    raise
                sleep_seconds = backoff * (2 ** (attempt - 1))
                LOG.warning(
                    "online_reward %s attempt %d/%d failed; retrying in %.2fs: %s",
                    key,
                    attempt,
                    attempts,
                    sleep_seconds,
                    error,
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
        raise RuntimeError(f"{key} retry exhausted without result")

    def _api_chat_json(self, *, cfg, messages, max_tokens):
        return self._openai_compatible_chat_json(cfg=cfg, messages=messages, max_tokens=max_tokens)

    def _openai_compatible_client(self, cfg: dict[str, Any]):
        base_url_env = str(cfg.get("base_url_env") or "JUDGE_BASE_URL").strip()
        base_url = str(
            os.getenv(base_url_env) or cfg.get("base_url") or cfg.get("api_base") or cfg.get("endpoint") or ""
        ).strip()
        if not base_url:
            raise ValueError(
                "Set judge.base_url or export "
                f"{base_url_env} for vllm_openai/openai_compatible backend"
            )
        api_key_env = str(cfg.get("api_key_env") or "VLLM_API_KEY").strip()
        api_key = str(os.getenv(api_key_env) or cfg.get("api_key") or os.getenv("OPENAI_API_KEY") or "EMPTY")
        timeout = float(cfg.get("timeout", self.online_cfg.get("timeout", 300)) or 300)
        key = (base_url, api_key_env + ":" + api_key, timeout)
        clients = getattr(self, "_openai_compatible_clients", None)
        if clients is None:
            clients = {}
            self._openai_compatible_clients = clients
        if key not in clients:
            from openai import OpenAI

            clients[key] = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        return clients[key]

    def _openai_compatible_chat_json(
        self, *, cfg: dict[str, Any], messages: list[dict[str, Any]], max_tokens: int
    ) -> dict[str, Any]:
        model_env = str(cfg.get("model_env") or "JUDGE_MODEL").strip()
        model = str(
            os.getenv(model_env)
            or cfg.get("model")
            or cfg.get("served_model_name")
            or self.online_cfg.get("model")
            or "judge-model"
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._messages_with_optional_system(messages, cfg),
            "temperature": float(cfg.get("temperature", self.online_cfg.get("temperature", 0.0))),
            "max_tokens": int(cfg.get("max_tokens") or cfg.get("max_new_tokens") or max_tokens),
        }
        if model.lower().startswith("gpt-5"):
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        if bool(cfg.get("response_format_json", self.online_cfg.get("response_format_json", True))):
            kwargs["response_format"] = {"type": "json_object"}
        reasoning_effort = cfg.get("reasoning_effort")
        has_reasoning_effort = reasoning_effort is not None and bool(str(reasoning_effort).strip())
        if has_reasoning_effort:
            kwargs["reasoning_effort"] = str(reasoning_effort).strip()
        extra_body = cfg.get("extra_body")
        merged_extra_body = dict(extra_body) if isinstance(extra_body, dict) else {}
        if has_reasoning_effort:
            google_body = dict(merged_extra_body.get("google") or {})
            google_body.pop("thinking_config", None)
            if google_body:
                merged_extra_body["google"] = google_body
            else:
                merged_extra_body.pop("google", None)
        thinking_config: dict[str, Any] = {}
        if not has_reasoning_effort and "thinking_budget" in cfg:
            thinking_config["thinking_budget"] = int(cfg["thinking_budget"])
        if not has_reasoning_effort and "include_thoughts" in cfg:
            thinking_config["include_thoughts"] = bool(cfg["include_thoughts"])
        if not has_reasoning_effort and thinking_config:
            google_body = dict(merged_extra_body.get("google") or {})
            existing_thinking = google_body.get("thinking_config")
            if isinstance(existing_thinking, dict):
                thinking_config = {**existing_thinking, **thinking_config}
            google_body["thinking_config"] = thinking_config
            merged_extra_body["google"] = google_body
        if merged_extra_body:
            kwargs["extra_body"] = merged_extra_body
        extra_headers = cfg.get("extra_headers")
        if isinstance(extra_headers, dict) and extra_headers:
            kwargs["extra_headers"] = {str(key): str(value) for key, value in extra_headers.items()}
        response = self._openai_compatible_client(cfg).chat.completions.create(**kwargs)
        return extract_json_object(self._chat_response_text(response))


    def _image_edit_cfg(self) -> dict[str, Any]:
        configured = section(self.online_cfg, "image_edit")
        if configured:
            return configured
        return section(self.online_cfg, "gpt_image2")

    def _image_edit_model_name(self) -> str:
        return str(self._image_edit_cfg().get("model") or self._image_edit_cfg().get("provider") or "image-edit-model")

    def _generated_image_refs(self, generated_ref: str, request: dict[str, Any]) -> dict[str, Any]:
        # Inline media is private to the judge request, not duplicated into audits.
        public_ref = generated_ref if is_remote_url(generated_ref) else ""
        if not public_ref and bool(self.online_cfg.get("save_generated_images", False)):
            image_bytes, mime = image_bytes_from_data_url(generated_ref)
            folder = self._online_reward_work_dir() / "generated_images"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (safe_filename_part(request.get("request_id")) + "." + image_extension_for_mime(mime))
            path.write_bytes(image_bytes)
            public_ref = str(path)
        return {"generated_image_url": public_ref,
                "generated_image_source": "provider_url" if is_remote_url(generated_ref) else "inline",
                "_generated_ref_for_judge": generated_ref}

    def _online_reward_work_dir(self) -> Path:
        if self._judge_tmp_root is not None:
            return self._judge_tmp_root.parent
        project_dir = Path(self.config["_project_dir"])
        reward_cfg = section(self.config, "reward")
        configured = reward_cfg.get("work_dir")
        if configured is not None and str(configured).strip():
            work_dir = resolve_path(str(configured), base_dir=project_dir)
        else:
            train_cfg = section(self.config, "training")
            output_dir = resolve_path(str(train_cfg.get("output_dir", "./outputs/baseline")), base_dir=project_dir)
            work_dir = (output_dir or (project_dir / "outputs" / "baseline")) / "online_reward"
        if work_dir is None:
            work_dir = project_dir / "work" / "online_reward"
        return work_dir

    def _judge_resized_root(self) -> Path:
        if self._judge_tmp_root is None:
            self._judge_tmp_root = self._online_reward_work_dir() / "judge_resized_tmp"
        return self._judge_tmp_root

    def _chat_image_part(self, url: str, *, model: str, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
        cfg = cfg or {}
        if not str(url).startswith(("http://", "https://", "data:image/")):
            url = image_data_url_from_file(url)
        chat_image_format = str(cfg.get("chat_image_format") or self.online_cfg.get("chat_image_format") or "auto").strip().lower()
        if chat_image_format == "auto":
            model_key = model.lower()
            chat_image_format = "image" if any(key in model_key for key in ("qwen", "molmo")) else "image_url"
        if chat_image_format == "image":
            image_value = str(url)
            if bool(cfg.get("inline_chat_images", self.online_cfg.get("inline_chat_images", False))) and image_value.startswith(("http://", "https://")):
                timeout = float(cfg.get("image_download_timeout", self.online_cfg.get("image_download_timeout", 60)) or 60)
                image_value = image_data_url_from_url(image_value, timeout=timeout)
            return {"type": "image", "image": image_value}
        return {"type": "image_url", "image_url": {"url": str(url)}}

    def _metadata_text(self, request: dict[str, Any]) -> str:
        return f"""Product title: {request.get('title', '')}
Product description: {request.get('description', '')}
Category: {request.get('category', request.get('category_l1', ''))}
Region: {request.get('market', request.get('grass_region', ''))}
""".strip()

    def _judge_cot(self, request: dict[str, Any]) -> dict[str, Any]:
        model = self._judge_model_name("cot")
        cfg = self._judge_cfg("cot")
        text = f"""{COT_JUDGE_PROMPT}

{self._metadata_text(request)}

Thinking:
{request.get('reasoning', '')}

Structured final_prompt:
{request.get('final_prompt', '')}
"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    self._chat_image_part(str(request.get("original_image_url") or ""), model=model, cfg=cfg),
                ],
            }
        ]
        start = time.monotonic()
        payload, retry_meta = self._with_retry(
            "judge",
            lambda: self._judge_json(
                "cot",
                messages=messages,
                max_tokens=int(self.online_cfg.get("cot_max_completion_tokens", 1400)),
            ),
            default_attempts=3,
            default_backoff=2.0,
        )
        payload["cot_judge_seconds"] = time.monotonic() - start
        payload["cot_judge_attempts"] = int(retry_meta.get("attempts", 1))
        if retry_meta.get("retry_errors"):
            payload["cot_judge_retry_errors"] = retry_meta["retry_errors"]
        return payload

    def _generate_image(self, request: dict[str, Any]) -> dict[str, Any]:
        total_start = time.monotonic()
        compiled_prompt = compile_final_prompt_for_image(str(request.get("raw_completion") or ""))
        image_url = str(request.get("original_image_url") or "").strip()
        if not image_url:
            raise ValueError("missing original_image_url")
        image_config = dict(self.config)
        image_config["gpt_image2_inference"] = {**section(self.config, "gpt_image2_inference"), **self._image_edit_cfg()}
        body = build_edit_body(compiled_prompt, image_url, image_config)
        edit_start = time.monotonic()
        validate_generated = bool(self._retry_section().get("validate_generated_image", True))

        def run_image_edit_once() -> tuple[dict[str, Any], str]:
            response = request_image_edit(body, image_config)
            generated = image_data_url_from_response(response)
            if validate_generated and str(generated).startswith("data:image/"):
                image_bytes, _mime = image_bytes_from_data_url(generated)
                verify_image_bytes(image_bytes)
            return response, generated

        (response_body, generated_ref), image_edit_retry = self._with_retry(
            "image_edit",
            run_image_edit_once,
            default_attempts=3,
            default_backoff=2.0,
            non_retryable=self._is_non_retryable_image_edit_error,
        )
        image_edit_seconds = time.monotonic() - edit_start
        generated_refs = self._generated_image_refs(generated_ref, request)
        image_payload = dict(generated_refs)
        image_payload["original_image_url"] = image_url
        image_payload["compiled_prompt"] = compiled_prompt
        image_payload["image_edit_model"] = self._image_edit_model_name()
        image_payload["image_edit_usage"] = response_body.get("usage", {}) if isinstance(response_body, dict) else {}
        image_payload["image_edit_seconds"] = image_edit_seconds
        image_payload["image_edit_attempts"] = int(image_edit_retry.get("attempts", 1))
        if image_edit_retry.get("retry_errors"):
            image_payload["image_edit_retry_errors"] = image_edit_retry["retry_errors"]
        image_payload["image_validation_enabled"] = validate_generated
        image_payload["image_pipeline_seconds"] = time.monotonic() - total_start
        image_payload["_request"] = request
        return image_payload

    def _judge_image_from_generated(self, request: dict[str, Any], image_payload: dict[str, Any]) -> dict[str, Any]:
        total_start = time.monotonic()
        generated_url = str(image_payload["generated_image_url"])
        generated_ref_for_judge = str(image_payload.get("_generated_ref_for_judge") or generated_url)
        compiled_prompt = str(image_payload.get("compiled_prompt") or "")
        try:
            judge_start = time.monotonic()
            judge_payload, judge_retry = self._with_retry(
                "judge",
                lambda: self._judge_image(request, compiled_prompt, generated_ref_for_judge),
                default_attempts=3,
                default_backoff=2.0,
            )
            judge_payload["image_judge_seconds"] = time.monotonic() - judge_start
            judge_payload["image_judge_attempts"] = int(judge_retry.get("attempts", 1))
            if judge_retry.get("retry_errors"):
                judge_payload["image_judge_retry_errors"] = judge_retry["retry_errors"]
        except Exception as exc:  # noqa: BLE001
            judge_payload = {
                "hard_fail": True,
                "hard_fail_reason": "image_judge_failed",
                "product_identity_gate": "fail",
                "product_count_gate": "fail",
                "variant_gate": "fail",
                "logo_package_text_gate": "fail",
                "unsupported_claim_gate": "fail",
                "severe_visual_damage_gate": "fail",
                "judge_failed": True,
                "judge_error": f"{type(exc).__name__}: {exc}",
                "reason": "Image judge request failed before scoring.",
            }
        judge_payload.update({key: value for key, value in image_payload.items() if not key.startswith("_")})
        judge_payload["image_pipeline_seconds"] = float(image_payload.get("image_pipeline_seconds") or 0.0) + (
            time.monotonic() - total_start
        )
        return judge_payload

    def _generate_and_judge_image(self, request: dict[str, Any]) -> dict[str, Any]:
        image_payload = self._generate_image(request)
        return self._judge_image_from_generated(request, image_payload)

    def _judge_image(self, request: dict[str, Any], compiled_prompt: str, generated_url: str) -> dict[str, Any]:
        model = self._judge_model_name("image")
        cfg = self._judge_cfg("image")
        text = f"""{IMAGE_JUDGE_PROMPT}

{self._metadata_text(request)}

Compiled final prompt sent to {self._image_edit_model_name()}:
{compiled_prompt}

Image 1 is the original product image. Image 2 is the generated candidate to judge.
"""
        max_images = int(cfg.get("max_images_per_prompt") or self.online_cfg.get("max_images_per_prompt") or 0)
        original_url = str(request.get("original_image_url") or "")
        cleanup_dirs: list[Path] = []
        judge_meta: dict[str, Any] = {}
        if max_images == 1:
            timeout = float(cfg.get("image_download_timeout", self.online_cfg.get("image_download_timeout", 60)) or 60)
            comparison_url = comparison_image_data_url(original_url, generated_url, timeout=timeout)
            text = text.replace(
                "Image 1 is the original product image. Image 2 is the generated candidate to judge.",
                "The single image is a side-by-side comparison: left is the original product image, right is the generated candidate to judge.",
            )
            content = [
                {"type": "text", "text": text},
                self._chat_image_part(comparison_url, model=model, cfg=cfg),
            ]
        else:
            judge_original_url = original_url
            judge_generated_url = generated_url
            resize_inputs = bool(cfg.get("resize_inputs", self.online_cfg.get("resize_judge_images", True)))
            if resize_inputs:
                timeout = float(cfg.get("image_download_timeout", self.online_cfg.get("image_download_timeout", 60)) or 60)
                max_side = int(cfg.get("resize_max_side", self.online_cfg.get("judge_resize_max_side", 1024)) or 1024)
                request_id = safe_filename_part(request.get("request_id"))
                temp_dir = self._judge_resized_root() / request_id
                cleanup_dirs.append(temp_dir)
                original_resized = resize_image_reference_for_judge(
                    original_url,
                    temp_dir / "original.png",
                    timeout=timeout,
                    max_side=max_side,
                )
                generated_resized = resize_image_reference_for_judge(
                    generated_url,
                    temp_dir / "generated.png",
                    timeout=timeout,
                    max_side=max_side,
                )
                judge_original_url = str(original_resized["data_url"])
                judge_generated_url = str(generated_resized["data_url"])
                judge_meta.update(
                    {
                        "judge_resized_inputs": True,
                        "judge_resize_max_side": max_side,
                        "judge_original_resized_path": original_resized["path"],
                        "judge_generated_resized_path": generated_resized["path"],
                        "judge_original_size": original_resized["original_size"],
                        "judge_original_resized_size": original_resized["resized_size"],
                        "judge_generated_size": generated_resized["original_size"],
                        "judge_generated_resized_size": generated_resized["resized_size"],
                    }
                )
            content = [
                {"type": "text", "text": text},
                self._chat_image_part(judge_original_url, model=model, cfg=cfg),
                self._chat_image_part(judge_generated_url, model=model, cfg=cfg),
            ]
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        try:
            payload = self._judge_json(
                "image",
                messages=messages,
                max_tokens=int(self.online_cfg.get("image_max_completion_tokens", 2200)),
            )
            payload.update(judge_meta)
            return payload
        finally:
            if bool(cfg.get("cleanup_resized_inputs", self.online_cfg.get("cleanup_judge_resized_images", True))):
                for path in cleanup_dirs:
                    shutil.rmtree(path, ignore_errors=True)

    def _cot_reward(self, payload: dict[str, Any]) -> float:
        if bool(payload.get("hard_fail", False)) or bool(payload.get("long_overlap_with_final_prompt", False)):
            return 0.0
        weights = section(self.online_cfg, "cot_reward_weights")
        return weighted_score(
            payload,
            weights,
            [
                ("source_understanding", "source_understanding_score", 0.12),
                ("grounded_fact_selection", "grounded_fact_selection_score", 0.30),
                ("risk_control", "risk_control_score", 0.30),
                ("design_rationale", "design_rationale_score", 0.22),
                ("efficiency", "efficiency_score", 0.04),
                ("non_redundancy", "non_redundancy_score", 0.02),
            ],
        )

    def _image_reward(self, payload: dict[str, Any], request: dict[str, Any]) -> float:
        result = self.image_scorer.score(payload, request)
        payload["contextual_scoring"] = result
        return result["reward"]

    def _scalar_reward(self, cot_reward: float, image_reward: float) -> float:
        scalar_cfg = section(self.online_cfg, "scalar_logging_weights")
        cot_weight = float(scalar_cfg.get("cot", 0.1))
        image_weight = float(scalar_cfg.get("image", 0.9))
        total = max(cot_weight + image_weight, 1e-9)
        return (cot_weight * cot_reward + image_weight * image_reward) / total
