from __future__ import annotations

import argparse
import logging
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from contextlib import suppress

from utils.config import load_config, resolve_path, section
from utils.data import build_dataset
from utils.extensible_grpo_trainer import make_extensible_grpo_trainer_class
from utils.gpu_memory import maybe_log_cuda_memory
from utils.modeling import load_policy_model, load_processor
from utils.training import make_grpo_config


LOG = logging.getLogger(__name__)


class ExactMessageFilter(logging.Filter):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def filter(self, record: logging.LogRecord) -> bool:
        return self.text not in record.getMessage()


def install_log_filters() -> None:
    floops_message = "Could not estimate the number of tokens of the input"
    message_filter = ExactMessageFilter(floops_message)
    for logger_name in ("", "transformers", "transformers.trainer", "transformers.trainer_utils"):
        logger = logging.getLogger(logger_name)
        logger.addFilter(message_filter)
        for handler in logger.handlers:
            handler.addFilter(message_filter)


def destroy_distributed_process_group() -> None:
    with suppress(Exception):
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def maybe_prepare_grpo_lora_model(model, config: dict):
    lora_cfg = section(config, "lora")
    if not lora_cfg or not bool(lora_cfg.get("enabled", False)):
        return model
    from sft.train_schema_sft import prepare_lora_model

    name_or_path = getattr(model, "name_or_path", None)
    config_name_or_path = getattr(getattr(model, "config", None), "_name_or_path", None)
    model = prepare_lora_model(model, config)
    if name_or_path is not None:
        setattr(model, "name_or_path", name_or_path)
    model_config = getattr(model, "config", None)
    if model_config is not None:
        if config_name_or_path is not None:
            setattr(model_config, "_name_or_path", config_name_or_path)
        if name_or_path is not None:
            setattr(model_config, "name_or_path", name_or_path)
    LOG.info("Prepared fresh trainable GRPO LoRA adapter")
    return model


def import_object(path: str):
    import importlib

    module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Import path must look like module.Object, got {path!r}")
    return getattr(importlib.import_module(module_name), attr)


def checkpoint_step(path: Path) -> int:
    name = path.name
    if not name.startswith("checkpoint-"):
        return -1
    try:
        return int(name.rsplit("-", 1)[-1])
    except ValueError:
        return -1


def latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    candidates = [path for path in output_dir.iterdir() if path.is_dir() and checkpoint_step(path) >= 0]
    if not candidates:
        return None
    return max(candidates, key=checkpoint_step)


def resolve_resume_from_checkpoint(config: dict) -> str | None:
    train_cfg = section(config, "training")
    raw = train_cfg.get("resume_from_checkpoint")
    if raw is None:
        return None
    if isinstance(raw, bool):
        if not raw:
            return None
        raw = "auto"
    value = str(raw).strip()
    if not value or value.lower() in {"none", "null", "false", "0", "no", "off"}:
        return None
    project_dir = Path(config["_project_dir"])
    output_dir = resolve_path(str(train_cfg.get("output_dir", "./outputs/baseline")), base_dir=project_dir)
    if output_dir is None:
        raise ValueError("training.output_dir must not be null when resolving resume_from_checkpoint")
    if value.lower() == "auto":
        checkpoint = latest_checkpoint(output_dir)
        if checkpoint is None:
            LOG.info("resume_from_checkpoint=auto but no checkpoint-* found under %s; starting fresh", output_dir)
            return None
        LOG.info("Resuming GRPO from latest checkpoint: %s", checkpoint)
        return str(checkpoint)
    checkpoint = resolve_path(value, base_dir=project_dir)
    if checkpoint is None:
        raise ValueError(f"Could not resolve resume_from_checkpoint={value!r}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"resume_from_checkpoint does not exist: {checkpoint}")
    LOG.info("Resuming GRPO from configured checkpoint: %s", checkpoint)
    return str(checkpoint)


def patch_peft_tensor_parallel_resume() -> None:
    """Avoid PEFT/Transformers TP import mismatches when resuming single-GPU LoRA.

    PEFT 0.19 may try to import tensor-parallel helper classes while loading a
    LoRA checkpoint. Our GRPO policy is not tensor-parallel-sharded, so this TP
    sharding hook should be a no-op. Some Transformers builds do not expose the
    symbols PEFT imports there, which otherwise breaks checkpoint resume.
    """

    try:
        import peft.utils.save_and_load as peft_save_and_load
    except Exception as exc:  # pragma: no cover - defensive for non-PEFT configs
        LOG.warning("Could not import PEFT save/load utilities for resume patch: %s", exc)
        return

    current = getattr(peft_save_and_load, "_maybe_shard_state_dict_for_tp", None)
    if current is None:
        return
    if getattr(current, "_main_image_grpo_no_tp_patch", False):
        return

    def _no_tp_shard_state_dict_for_resume(model, peft_model_state_dict, adapter_name):
        return peft_model_state_dict

    setattr(_no_tp_shard_state_dict_for_resume, "_main_image_grpo_no_tp_patch", True)
    peft_save_and_load._maybe_shard_state_dict_for_tp = _no_tp_shard_state_dict_for_resume
    LOG.info("Patched PEFT tensor-parallel checkpoint sharding as no-op for single-GPU resume")


def train(config_path: str) -> None:
    install_log_filters()
    config = load_config(config_path)
    train_cfg = section(config, "training")
    log_gpu_memory = train_cfg.get("log_gpu_memory", True)
    project_dir = Path(config["_project_dir"])
    maybe_log_cuda_memory(log_gpu_memory, "startup", logger=LOG)
    processor = load_processor(config)
    dataset = build_dataset(config, processor)
    maybe_log_cuda_memory(log_gpu_memory, "after_dataset", logger=LOG)
    model = load_policy_model(config, tokenizer=processor.tokenizer, processor=processor)
    maybe_log_cuda_memory(log_gpu_memory, "after_policy_model_load", logger=LOG)
    model = maybe_prepare_grpo_lora_model(model, config)
    maybe_log_cuda_memory(log_gpu_memory, "after_grpo_lora", logger=LOG)
    args = make_grpo_config(config)

    reward_cfg = section(config, "reward")
    if reward_cfg.get("provider", "online_component") != "online_component":
        raise ValueError("This release uses the contextual online_component reward provider")
    from rewards.online_component_reward import ComponentRewardStore, OnlineComponentRewardProvider

    reward_store = ComponentRewardStore()
    reward_provider = OnlineComponentRewardProvider(config, reward_store, processing_class=processor)
    advantage_provider_path = str(
        reward_cfg.get("advantage_provider") or "utils.element_kernel_advantage.ElementKernelAdvantageProvider"
    )
    advantage_provider = import_object(advantage_provider_path)(reward_store, config=config)
    trainer_class = make_extensible_grpo_trainer_class()
    trainer = trainer_class(
        model=model,
        args=args,
        reward_funcs=reward_provider,
        train_dataset=dataset,
        processing_class=processor,
        advantage_provider=advantage_provider,
    )
    maybe_log_cuda_memory(log_gpu_memory, "after_trainer_init", logger=LOG)
    try:
        resume_from_checkpoint = resolve_resume_from_checkpoint(config)
        if resume_from_checkpoint is None:
            trainer.train()
        else:
            patch_peft_tensor_parallel_resume()
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        maybe_log_cuda_memory(log_gpu_memory, "after_train", logger=LOG)
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)
        LOG.info("Saved GRPO run artifacts to %s", args.output_dir)
    finally:
        destroy_distributed_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train main-image GRPO from the configured policy and reward setup.")
    parser.add_argument("--config", default="configs/grpo_gpt.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    train(args.config)


if __name__ == "__main__":
    main()
