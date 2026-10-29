from __future__ import annotations

import importlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from prompts.schema import validate_sft_prompt
from utils.completion_schema import parse_completion
from utils.config import resolve_path, section
from utils.reward_protocol import completion_text, completion_text_from_token_ids, json_safe


LOG = logging.getLogger(__name__)


@dataclass
class ComponentReward:
    request_id: str
    row_key: str
    sample_index: int
    raw_completion: str
    schema_valid: bool
    schema_error: str = ""
    active: bool = True
    drop_reason: str = ""
    cot_reward: float = 0.0
    image_reward: float = 0.0
    scalar_reward: float = 0.0
    cot_advantage: float = 0.0
    image_advantage: float = 0.0
    format_advantage: float = 0.0
    format_invalid: bool = False
    group_diagnostics: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentRewardBatch:
    rewards: list[ComponentReward]


class ComponentRewardStore:
    def __init__(self) -> None:
        self.latest: ComponentRewardBatch | None = None

    def set_latest(self, batch: ComponentRewardBatch) -> None:
        self.latest = batch

    def consume_latest(self) -> ComponentRewardBatch:
        if self.latest is None:
            raise RuntimeError("No component reward batch is available for token advantage computation")
        batch = self.latest
        self.latest = None
        return batch


class OnlineRewardStrategy(Protocol):
    def score(self, requests: list[dict[str, Any]]) -> list[ComponentReward]:
        ...


def _schema_marker_count(text: str) -> int:
    lowered = str(text or "").lower()
    return sum(
        lowered.count(marker)
        for marker in ("<thinking", "</thinking", "<reasoning", "</reasoning", "<think", "</think", "<final_prompt", "</final_prompt")
    )


def _preview_text(text: str, limit: int = 1000) -> str:
    value = str(text or "")
    return value[:limit]


def import_object(path: str) -> Any:
    module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Import path must look like module.Object, got {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def default_online_reward_work_dir(config: dict[str, Any]) -> Path:
    project_dir = Path(config["_project_dir"])
    train_cfg = section(config, "training")
    output_dir = resolve_path(str(train_cfg.get("output_dir", "./outputs/baseline")), base_dir=project_dir)
    if output_dir is None:
        raise ValueError("training.output_dir must not be null when resolving online reward work_dir")
    return output_dir / "online_reward"


def resolve_online_reward_work_dir(config: dict[str, Any], reward_cfg: dict[str, Any]) -> Path:
    project_dir = Path(config["_project_dir"])
    configured = reward_cfg.get("work_dir")
    if configured is None or str(configured).strip() == "":
        return default_online_reward_work_dir(config)
    work_dir = resolve_path(str(configured), base_dir=project_dir)
    if work_dir is None:
        return default_online_reward_work_dir(config)
    return work_dir


def zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(var)
    if std <= 1e-6:
        return [0.0 for _ in values]
    return [(value - mean) / (std + 1e-6) for value in values]


def reward_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "range": 0.0}
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(var)
    return {
        "mean": float(mean),
        "std": float(std),
        "min": float(min(values)),
        "max": float(max(values)),
        "range": float(max(values) - min(values)),
    }


def normalize_component_advantages(
    rewards: list[ComponentReward],
    *,
    min_successful_samples: int,
    min_successful_ratio: float,
) -> None:
    groups: dict[str, list[ComponentReward]] = {}
    for reward in rewards:
        groups.setdefault(reward.row_key, []).append(reward)

    for group_rewards in groups.values():
        active = [reward for reward in group_rewards if reward.active]
        required = max(int(min_successful_samples), int(math.ceil(len(group_rewards) * float(min_successful_ratio))))
        cot_stats = reward_stats([float(reward.cot_reward) for reward in active])
        image_stats = reward_stats([float(reward.image_reward) for reward in active])
        scalar_stats = reward_stats([float(reward.scalar_reward) for reward in active])
        group_diagnostics = {
            "row_key": str(group_rewards[0].row_key) if group_rewards else "",
            "total_samples": int(len(group_rewards)),
            "active_samples": int(len(active)),
            "inactive_samples": int(len(group_rewards) - len(active)),
            "required_active_samples": int(required),
            "min_successful_samples": int(min_successful_samples),
            "min_successful_ratio": float(min_successful_ratio),
            "cot_reward": cot_stats,
            "image_reward": image_stats,
            "scalar_reward": scalar_stats,
            "cot_advantage_zero_std": bool(cot_stats["std"] <= 1e-6),
            "image_advantage_zero_std": bool(image_stats["std"] <= 1e-6),
            "scalar_advantage_zero_std": bool(scalar_stats["std"] <= 1e-6),
            "any_advantage_zero_std": bool(
                cot_stats["std"] <= 1e-6 or image_stats["std"] <= 1e-6 or scalar_stats["std"] <= 1e-6
            ),
            "group_dropped_too_few_successes": bool(len(active) < required),
        }
        for reward in group_rewards:
            reward.group_diagnostics = dict(group_diagnostics)
        if len(active) < required:
            reason = f"dropped_group_too_few_successes: active={len(active)} required={required} total={len(group_rewards)}"
            for reward in group_rewards:
                reward.active = False
                reward.drop_reason = reward.drop_reason or reason
                reward.cot_advantage = 0.0
                reward.image_advantage = 0.0
                reward.format_advantage = 0.0
            continue

        cot_advantages = zscore([reward.cot_reward for reward in active])
        image_advantages = zscore([reward.image_reward for reward in active])
        scalar_advantages = zscore([reward.scalar_reward for reward in active])
        for reward, cot_adv, image_adv, scalar_adv in zip(active, cot_advantages, image_advantages, scalar_advantages):
            reward.cot_advantage = cot_adv
            reward.image_advantage = image_adv
            reward.format_advantage = scalar_adv


class OnlineComponentRewardProvider:
    """TRL reward callable that stores component rewards for token-level credit."""

    def __init__(self, config: dict[str, Any], store: ComponentRewardStore, processing_class: Any | None = None) -> None:
        self.config = config
        self.store = store
        self.processing_class = processing_class
        reward_cfg = section(config, "reward")
        project_dir = Path(config["_project_dir"])
        strategy_path = str(reward_cfg.get("strategy") or "rewards.strategy.DecoupledCotImageRewardStrategy")
        strategy_cls = import_object(strategy_path)
        self.strategy: OnlineRewardStrategy = strategy_cls(config)
        self.work_dir = resolve_online_reward_work_dir(config, reward_cfg)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._call_index = 0
        self.__name__ = "online_component_reward"

    def __call__(self, prompts: list[Any], completions: list[Any], trainer_state: Any = None, **kwargs: Any) -> list[float]:
        total_start = time.monotonic()
        del prompts
        step = int(getattr(trainer_state, "global_step", 0) or 0)
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0)
        call_index = self._call_index
        self._call_index += 1
        completion_ids_list = kwargs.get("completion_ids") if isinstance(kwargs.get("completion_ids"), list) else []
        requests: list[dict[str, Any]] = []
        for index, completion in enumerate(completions):
            completion_ids = completion_ids_list[index] if index < len(completion_ids_list) else None
            completion_raw = completion_text(completion)
            token_raw = completion_text_from_token_ids(completion_ids, self.processing_class)
            raw = completion_raw
            raw_source = "completion"
            if token_raw and (not raw or _schema_marker_count(token_raw) > _schema_marker_count(raw)):
                raw = token_raw
                raw_source = "completion_ids"
            schema_valid = True
            schema_error = ""
            reasoning = ""
            final_prompt = ""
            try:
                parsed = parse_completion(raw)
                reasoning = parsed.reasoning
                final_prompt = parsed.final_prompt
                parse_completion(raw, require_structured_children=True)
                validate_sft_prompt(raw, overlap_ngram=10_000)
            except ValueError as exc:
                schema_valid = False
                schema_error = str(exc)
                if raw_source != "completion_ids" and token_raw and token_raw != raw:
                    try:
                        parsed = parse_completion(token_raw)
                        parse_completion(token_raw, require_structured_children=True)
                        validate_sft_prompt(token_raw, overlap_ngram=10_000)
                    except ValueError:
                        parsed = None
                    else:
                        raw = token_raw
                        raw_source = "completion_ids_after_parse_failure"
                        reasoning = parsed.reasoning
                        final_prompt = parsed.final_prompt
                        schema_valid = True
                        schema_error = ""
            request = {
                "request_id": f"step-{step:06d}__rank-{rank:04d}__call-{call_index:06d}__sample-{index:06d}",
                "step": step,
                "rank": rank,
                "sample_index": index,
                "raw_completion": raw,
                "schema_valid": schema_valid,
                "schema_error": schema_error,
                "reasoning": reasoning,
                "final_prompt": final_prompt,
            }
            if not schema_valid or raw_source != "completion" or not raw:
                ids = []
                if hasattr(completion_ids, "tolist"):
                    ids = completion_ids.tolist()
                elif isinstance(completion_ids, list):
                    ids = completion_ids
                request["completion_debug"] = {
                    "raw_source": raw_source,
                    "completion_type": type(completion).__name__,
                    "completion_preview": json_safe(completion),
                    "completion_ids_len": len(ids) if isinstance(ids, list) else 0,
                    "completion_ids_preview": ids[:32] if isinstance(ids, list) else [],
                    "completion_raw_len": len(completion_raw),
                    "completion_raw_preview": _preview_text(completion_raw),
                    "token_raw_len": len(token_raw),
                    "token_raw_preview": _preview_text(token_raw),
                    "completion_schema_markers": _schema_marker_count(completion_raw),
                    "token_schema_markers": _schema_marker_count(token_raw),
                }
            for key, values in kwargs.items():
                if isinstance(values, list) and index < len(values):
                    request[key] = json_safe(values[index])
            request["row_key"] = str(request.get("row_key") or f"row-{index:08d}")
            requests.append(request)

        strategy_start = time.monotonic()
        rewards = self.strategy.score(requests)
        strategy_seconds = time.monotonic() - strategy_start
        rewards_by_id = {reward.request_id: reward for reward in rewards}
        ordered_rewards: list[ComponentReward] = []
        schema_invalid_reward = float(section(self.config, "online_reward").get("schema_invalid_reward", 0.0))
        for request in requests:
            reward = rewards_by_id.get(str(request["request_id"]))
            if reward is None:
                reward = ComponentReward(
                    request_id=str(request["request_id"]),
                    row_key=str(request.get("row_key") or ""),
                    sample_index=int(request["sample_index"]),
                    raw_completion=str(request.get("raw_completion") or ""),
                    schema_valid=bool(request.get("schema_valid", False)),
                    active=False,
                    drop_reason="strategy_missing_reward",
                )
            if not reward.schema_valid:
                reward.format_invalid = True
                reward.cot_reward = min(reward.cot_reward, schema_invalid_reward)
                reward.image_reward = min(reward.image_reward, schema_invalid_reward)
                reward.scalar_reward = schema_invalid_reward
            ordered_rewards.append(reward)

        group_cfg = section(self.config, "online_reward").get("rollout_group", {}) or {}
        normalize_start = time.monotonic()
        normalize_component_advantages(
            ordered_rewards,
            min_successful_samples=int(group_cfg.get("min_successful_samples", 1)),
            min_successful_ratio=float(group_cfg.get("min_successful_ratio", 0.0)),
        )
        normalize_seconds = time.monotonic() - normalize_start
        self._write_audit(step, rank, call_index, ordered_rewards)
        self._log_timing_summary(
            step=step,
            rank=rank,
            call_index=call_index,
            rewards=ordered_rewards,
            strategy_seconds=strategy_seconds,
            normalize_seconds=normalize_seconds,
            total_seconds=time.monotonic() - total_start,
        )
        self.store.set_latest(ComponentRewardBatch(ordered_rewards))
        return [float(reward.scalar_reward) if reward.active else 0.0 for reward in ordered_rewards]

    def _log_timing_summary(
        self,
        *,
        step: int,
        rank: int,
        call_index: int,
        rewards: list[ComponentReward],
        strategy_seconds: float,
        normalize_seconds: float,
        total_seconds: float,
    ) -> None:
        def mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        cot_seconds: list[float] = []
        image_edit_seconds: list[float] = []
        media_upload_seconds: list[float] = []
        image_judge_seconds: list[float] = []
        image_pipeline_seconds: list[float] = []
        for reward in rewards:
            cot = reward.details.get("cot_judge")
            if isinstance(cot, dict) and cot.get("cot_judge_seconds") is not None:
                cot_seconds.append(float(cot["cot_judge_seconds"]))
            image = reward.details.get("image_judge")
            if not isinstance(image, dict):
                continue
            for key, bucket in (
                ("image_edit_seconds", image_edit_seconds),
                ("media_upload_seconds", media_upload_seconds),
                ("image_judge_seconds", image_judge_seconds),
                ("image_pipeline_seconds", image_pipeline_seconds),
            ):
                if image.get(key) is not None:
                    bucket.append(float(image[key]))
        LOG.info(
            "online_component_reward timing: step=%d rank=%d call=%d samples=%d active=%d "
            "total=%.3fs strategy=%.3fs normalize=%.3fs avg_cot_judge=%.3fs "
            "avg_image_edit=%.3fs avg_media_upload=%.3fs avg_image_judge=%.3fs avg_image_pipeline=%.3fs",
            step,
            rank,
            call_index,
            len(rewards),
            sum(1 for reward in rewards if reward.active),
            total_seconds,
            strategy_seconds,
            normalize_seconds,
            mean(cot_seconds),
            mean(image_edit_seconds),
            mean(media_upload_seconds),
            mean(image_judge_seconds),
            mean(image_pipeline_seconds),
        )

    def _write_audit(self, step: int, rank: int, call_index: int, rewards: list[ComponentReward]) -> None:
        path = self.work_dir / f"component_rewards_step-{step:06d}_rank-{rank:04d}_call-{call_index:06d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for reward in rewards:
                handle.write(
                    json_dumps(
                        {
                            "request_id": reward.request_id,
                            "row_key": reward.row_key,
                            "sample_index": reward.sample_index,
                            "raw_completion": reward.raw_completion,
                            "schema_valid": reward.schema_valid,
                            "schema_error": reward.schema_error,
                            "active": reward.active,
                            "drop_reason": reward.drop_reason,
                            "cot_reward": reward.cot_reward,
                            "image_reward": reward.image_reward,
                            "scalar_reward": reward.scalar_reward,
                            "cot_advantage": reward.cot_advantage,
                            "image_advantage": reward.image_advantage,
                            "format_advantage": reward.format_advantage,
                            "format_invalid": reward.format_invalid,
                            "group_diagnostics": reward.group_diagnostics,
                            "details": reward.details,
                        }
                    )
                    + "\n"
                )


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
