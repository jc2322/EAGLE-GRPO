from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from utils.config import resolve_path, section


def configure_trl_vllm_import(config: dict[str, Any]) -> None:
    """Prevent incompatible ambient vLLM installs from breaking non-vLLM GRPO.

    TRL 0.24 imports vLLM symbols while importing GRPOTrainer/GRPOConfig if a
    vLLM package is visible on sys.path. Some job images expose newer global
    vLLM builds whose APIs no longer match TRL 0.24. When config explicitly
    disables vLLM rollout, make TRL behave as if vLLM is unavailable.
    """
    generation = section(config, "generation")
    if bool(generation.get("use_vllm", False)):
        return
    try:
        import trl
        import trl.import_utils as trl_import_utils
    except Exception:
        return
    version = str(getattr(trl, "__version__", ""))
    if not version.startswith("0."):
        return
    trl_import_utils._vllm_available = False
    trl_import_utils._vllm_version = None


def make_grpo_config(config: dict[str, Any]):
    project_dir = Path(config["_project_dir"])
    train = section(config, "training")
    generation = section(config, "generation")
    configure_trl_vllm_import(config)
    from trl import GRPOConfig

    output_dir = resolve_path(str(train.get("output_dir", "./outputs/baseline")), base_dir=project_dir)
    if output_dir is None:
        raise ValueError("training.output_dir must not be null")
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(train.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(train.get("gradient_accumulation_steps", 8)),
        "max_steps": int(train.get("max_steps", 200)),
        "learning_rate": float(train.get("learning_rate", 1e-6)),
        "weight_decay": float(train.get("weight_decay", 0.0)),
        "warmup_steps": int(train.get("warmup_steps", 10)),
        "lr_scheduler_type": str(train.get("lr_scheduler_type", "cosine")),
        "max_grad_norm": float(train.get("max_grad_norm", 1.0)),
        "logging_steps": int(train.get("logging_steps", 1)),
        "save_steps": int(train.get("save_steps", 50)),
        "save_total_limit": int(train.get("save_total_limit", 3)),
        "bf16": bool(train.get("bf16", True)),
        "fp16": bool(train.get("fp16", False)),
        "gradient_checkpointing": bool(train.get("gradient_checkpointing", True)),
        "use_liger_kernel": bool(train.get("use_liger_kernel", False)),
        "remove_unused_columns": False,
        "report_to": [] if str(train.get("report_to", "none")).lower() == "none" else train.get("report_to"),
        "seed": int(train.get("seed", 42)),
        "disable_dropout": bool(train.get("disable_dropout", True)),
        "log_completions": bool(train.get("log_completions", True)),
        "beta": float(train.get("beta", 0.01)),
        "epsilon": float(train.get("epsilon", 0.2)),
        "loss_type": str(train.get("loss_type", "grpo")),
        "scale_rewards": str(train.get("scale_rewards", "group")),
        "num_iterations": int(train.get("num_iterations", 1)),
        "importance_sampling_level": str(train.get("importance_sampling_level", "token")),
        "mask_truncated_completions": bool(train.get("mask_truncated_completions", True)),
        "generation_batch_size": (
            int(train["generation_batch_size"]) if train.get("generation_batch_size") is not None else None
        ),
        "steps_per_generation": (
            int(train["steps_per_generation"]) if train.get("steps_per_generation") is not None else None
        ),
        "num_generations": int(generation.get("num_generations", 8)),
        "max_prompt_length": int(generation.get("max_prompt_length", train.get("max_prompt_length", 8192))),
        "max_completion_length": int(generation.get("max_completion_length", 2048)),
        "temperature": float(generation.get("temperature", 0.9)),
        "top_p": float(generation.get("top_p", 0.95)),
        "min_p": float(generation["min_p"]) if generation.get("min_p") is not None else None,
        "top_k": int(generation.get("top_k", 50)),
        "repetition_penalty": float(generation.get("repetition_penalty", 1.05)),
        "use_vllm": bool(generation.get("use_vllm", False)),
        "vllm_mode": generation.get("vllm_mode"),
        "vllm_model_impl": generation.get("vllm_model_impl"),
        "vllm_enable_sleep_mode": (
            bool(generation["vllm_enable_sleep_mode"])
            if generation.get("vllm_enable_sleep_mode") is not None
            else None
        ),
        "vllm_server_base_url": generation.get("vllm_server_base_url"),
        "vllm_server_host": generation.get("vllm_server_host"),
        "vllm_server_port": (
            int(generation["vllm_server_port"]) if generation.get("vllm_server_port") is not None else None
        ),
        "vllm_server_timeout": (
            float(generation["vllm_server_timeout"]) if generation.get("vllm_server_timeout") is not None else None
        ),
        "vllm_group_port": (
            int(generation["vllm_group_port"]) if generation.get("vllm_group_port") is not None else None
        ),
        "vllm_max_model_len": (
            int(generation["vllm_max_model_len"]) if generation.get("vllm_max_model_len") is not None else None
        ),
        "vllm_max_model_length": (
            int(generation["vllm_max_model_length"]) if generation.get("vllm_max_model_length") is not None else None
        ),
        "vllm_gpu_memory_utilization": (
            float(generation["vllm_gpu_memory_utilization"])
            if generation.get("vllm_gpu_memory_utilization") is not None
            else None
        ),
        "vllm_tensor_parallel_size": (
            int(generation["vllm_tensor_parallel_size"])
            if generation.get("vllm_tensor_parallel_size") is not None
            else None
        ),
        "vllm_importance_sampling_correction": (
            bool(generation["vllm_importance_sampling_correction"])
            if generation.get("vllm_importance_sampling_correction") is not None
            else False
        ),
        "vllm_importance_sampling_mode": generation.get("vllm_importance_sampling_mode"),
        "vllm_importance_sampling_clip_min": (
            float(generation["vllm_importance_sampling_clip_min"])
            if generation.get("vllm_importance_sampling_clip_min") is not None
            else None
        ),
        "vllm_importance_sampling_clip_max": (
            float(generation["vllm_importance_sampling_clip_max"])
            if generation.get("vllm_importance_sampling_clip_max") is not None
            else None
        ),
        "vllm_structured_outputs_regex": generation.get("vllm_structured_outputs_regex"),
        "vllm_guided_decoding_regex": generation.get("vllm_guided_decoding_regex"),
        "pad_to_multiple_of": (
            int(generation["pad_to_multiple_of"]) if generation.get("pad_to_multiple_of") is not None else None
        ),
        "deepspeed": train.get("deepspeed"),
    }
    signature = inspect.signature(GRPOConfig.__init__).parameters
    return GRPOConfig(**{key: value for key, value in kwargs.items() if key in signature and value is not None})
