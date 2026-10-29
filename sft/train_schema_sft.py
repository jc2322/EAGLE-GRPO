from __future__ import annotations

import argparse
import csv
import faulthandler
import json
import logging
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from prompts.schema import validate_sft_prompt
from prompts.tokenizer_special_tokens import add_schema_special_tokens, schema_special_tokens
from utils.config import load_config, resolve_path, section
from utils.data import ImageCache, build_user_prompt, non_empty, read_csv
from utils.modeling import load_policy_model, load_processor
from utils.schema_token_training import (
    are_input_output_embeddings_tied,
    enable_schema_token_row_training,
    initialize_token_rows_from_subtokens,
    save_schema_token_rows,
    schema_subtoken_ids,
    schema_token_ids,
)


LOG = logging.getLogger(__name__)


def build_lora_config(config: dict[str, Any]):
    from peft import LoraConfig

    lora_cfg = section(config, "lora")
    target_modules = lora_cfg.get("target_modules")
    if isinstance(target_modules, str):
        target_modules = [item.strip() for item in target_modules.split(",") if item.strip()]
    return LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
        target_modules=target_modules,
    )


def prepare_lora_model(model: Any, config: dict[str, Any]) -> Any:
    lora_cfg = section(config, "lora")
    if not bool(lora_cfg.get("enabled", True)):
        return model
    from peft import get_peft_model, prepare_model_for_kbit_training

    model_cfg = section(config, "model")
    train_cfg = section(config, "training")
    if model_cfg.get("load_in_4bit") or model_cfg.get("load_in_8bit"):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        )
    model = get_peft_model(model, build_lora_config(config))
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def setup_schema_tokens(processor: Any, model: Any, config: dict[str, Any]) -> list[int]:
    model_cfg = section(config, "model")
    if not bool(model_cfg.get("add_schema_special_tokens", True)):
        return []
    tokenizer = processor.tokenizer
    old_vocab_size = len(tokenizer)
    tokens = schema_special_tokens()
    subtoken_ids = schema_subtoken_ids(tokenizer, tokens)
    added = add_schema_special_tokens(tokenizer)
    if added:
        LOG.info("Added %d schema special tokens; tokenizer size %d -> %d", added, old_vocab_size, len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))
    token_ids = schema_token_ids(tokenizer, tokens)
    if added:
        initialize_token_rows_from_subtokens(model, token_ids, subtoken_ids, tokens)
    LOG.info("Schema token rows prepared; tied input/output embeddings: %s", are_input_output_embeddings_tied(model))
    return token_ids


def apply_schema_row_training(model: Any, token_ids: list[int], config: dict[str, Any]) -> None:
    model_cfg = section(config, "model")
    if not token_ids or not bool(model_cfg.get("train_schema_token_rows", True)):
        return
    enable_schema_token_row_training(model, token_ids, train_lm_head=True)
    LOG.info("Enabled row-only training for %d schema tokens on embed_tokens and untied lm_head", len(token_ids))


def build_sft_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    return build_sft_rows_from_csv(config, "train_csv")


def build_sft_rows_from_csv(config: dict[str, Any], csv_config_key: str) -> list[dict[str, Any]]:
    project_dir = Path(config["_project_dir"])
    data_cfg = section(config, "data")
    csv_path = resolve_path(data_cfg.get(csv_config_key), base_dir=project_dir)
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(f"SFT CSV not found for data.{csv_config_key}: {csv_path}")
    rows = read_csv(csv_path)
    max_key = "max_eval_samples" if csv_config_key == "validation_csv" else "max_samples"
    if data_cfg.get(max_key):
        rows = rows[: int(data_cfg[max_key])]
    elif csv_config_key != "train_csv" and data_cfg.get("max_samples"):
        rows = rows[: int(data_cfg["max_samples"])]
    elif data_cfg.get("max_samples"):
        rows = rows[: int(data_cfg["max_samples"])]
    cache_dir = resolve_path(str(data_cfg.get("image_cache_dir", "./work/schema_sft_images")), base_dir=project_dir)
    image_cache = ImageCache(
        cache_dir or project_dir / "work/schema_sft_images",
        int(data_cfg.get("image_timeout_seconds", 30)),
        retries=int(data_cfg.get("image_retries", 3)),
        retry_sleep_seconds=float(data_cfg.get("image_retry_sleep_seconds", 1.0)),
        min_pixels=data_cfg.get("image_min_pixels"),
        max_pixels=data_cfg.get("image_max_pixels"),
    )
    image_column = str(data_cfg.get("image_column", "original_image_url"))
    prompt_column = str(data_cfg.get("prompt_column", "prompt"))
    skip_bad_images = bool(data_cfg.get("skip_bad_images", True))
    log_every = max(1, int(data_cfg.get("image_preflight_log_every", 100) or 100))

    dataset_rows: list[dict[str, Any]] = []
    skipped_images = 0
    LOG.info(
        "Preflighting %d SFT row(s) from %s; image cache=%s; progress every %d rows",
        len(rows),
        csv_path,
        image_cache.directory,
        log_every,
    )
    for index, row in enumerate(rows):
        if (index + 1) % log_every == 0 or index + 1 == len(rows):
            LOG.info(
                "SFT image preflight: starting row %d/%d; usable=%d skipped_images=%d",
                index + 1,
                len(rows),
                len(dataset_rows),
                skipped_images,
            )
        target = non_empty(row, prompt_column)
        image_ref = non_empty(row, image_column)
        title = non_empty(row, str(data_cfg.get("title_column", "title")))
        if not target or not image_ref or not title:
            continue
        validate_sft_prompt(target)
        try:
            image_path = image_cache.materialize(image_ref)
        except Exception as exc:  # noqa: BLE001
            if not skip_bad_images:
                raise
            skipped_images += 1
            LOG.warning(
                "Skipping row %s from %s because image could not be materialized: %s",
                non_empty(row, "request_key") or non_empty(row, "row_key") or index,
                image_ref,
                exc,
            )
            continue
        dataset_rows.append(
            {
                "row_key": non_empty(row, "row_key") or f"sft-row-{index:08d}",
                "image_path": str(image_path),
                "image_ref": image_ref,
                "user_prompt": build_user_prompt(config, row),
                "target": target,
            }
        )
    if skipped_images:
        LOG.warning("Skipped %d row(s) from %s due to image materialization failures", skipped_images, csv_path)
    if not dataset_rows:
        raise ValueError(f"No usable SFT rows found in {csv_path}")
    return dataset_rows


def build_eval_rows(config: dict[str, Any]) -> list[dict[str, Any]] | None:
    data_cfg = section(config, "data")
    if not data_cfg.get("validation_csv"):
        return None
    return build_sft_rows_from_csv(config, "validation_csv")


def message_content_text(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def build_messages(system_prompt: str, user_prompt: str, image_path: str, target: str | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": message_content_text(system_prompt)})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_prompt},
            ],
        }
    )
    if target is not None:
        messages.append({"role": "assistant", "content": message_content_text(target)})
    return messages


@dataclass
class VisionSftCollator:
    processor: Any
    system_prompt: str
    max_length: int
    image_cache: ImageCache | None = None

    def _render(self, row: dict[str, Any], *, include_target: bool) -> str:
        messages = build_messages(
            self.system_prompt,
            row["user_prompt"],
            row["image_path"],
            row["target"] if include_target else None,
        )
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not include_target,
        )

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images: list[Image.Image] = []
        full_texts: list[str] = []
        prompt_texts: list[str] = []
        for row in rows:
            if self.image_cache is not None:
                image = self.image_cache.load(str(row.get("image_ref") or row["image_path"]))
            else:
                image = Image.open(row["image_path"]).convert("RGB")
            images.append(image)
            full_texts.append(self._render(row, include_target=True))
            prompt_texts.append(self._render(row, include_target=False))

        full_inputs = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = full_inputs["input_ids"].clone()
        attention_mask = full_inputs.get("attention_mask")
        prompt_attention = prompt_inputs.get("attention_mask")
        for row_index in range(labels.shape[0]):
            prompt_len = int(prompt_attention[row_index].sum().item()) if prompt_attention is not None else 0
            labels[row_index, :prompt_len] = -100
        if attention_mask is not None:
            labels[attention_mask.eq(0)] = -100
        full_inputs["labels"] = labels
        return full_inputs


def make_training_args(config: dict[str, Any], *, has_eval: bool):
    from transformers import TrainingArguments

    project_dir = Path(config["_project_dir"])
    train_cfg = section(config, "training")
    output_dir = resolve_path(str(train_cfg.get("output_dir", "./outputs/main_image_structured_sft")), base_dir=project_dir)
    if output_dir is None:
        raise ValueError("training.output_dir must not be null")
    output_dir.mkdir(parents=True, exist_ok=True)
    import inspect

    signature = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 8)),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 1)),
        "max_steps": int(train_cfg.get("max_steps", -1)),
        "learning_rate": float(train_cfg.get("learning_rate", 2e-5)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
        "lr_scheduler_type": str(train_cfg.get("lr_scheduler_type", "cosine")),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
        "logging_steps": int(train_cfg.get("logging_steps", 5)),
        "save_steps": int(train_cfg.get("save_steps", 200)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 2)),
        "seed": int(train_cfg.get("seed", 42)),
        "bf16": bool(train_cfg.get("bf16", True)),
        "fp16": bool(train_cfg.get("fp16", False)),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "remove_unused_columns": bool(train_cfg.get("remove_unused_columns", False)),
        "dataloader_num_workers": int(train_cfg.get("dataloader_num_workers", 0)),
        "report_to": [] if str(train_cfg.get("report_to", "none")).lower() == "none" else train_cfg.get("report_to"),
    }
    eval_key = "eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"
    kwargs[eval_key] = str(train_cfg.get("eval_strategy", "steps")) if has_eval else "no"
    if "eval_steps" in signature:
        kwargs["eval_steps"] = int(train_cfg.get("eval_steps", train_cfg.get("save_steps", 200)))
    if "save_safetensors" in signature:
        kwargs["save_safetensors"] = bool(train_cfg.get("save_safetensors", True))
    if "optim" in signature and train_cfg.get("optim"):
        kwargs["optim"] = str(train_cfg["optim"])
    if "ddp_find_unused_parameters" in signature:
        kwargs["ddp_find_unused_parameters"] = bool(train_cfg.get("ddp_find_unused_parameters", False))
    return TrainingArguments(**{key: value for key, value in kwargs.items() if key in signature})


def save_trainer_history(log_history: list[dict[str, Any]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trainer_log_history.json").write_text(
        json.dumps(log_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fieldnames = sorted({key for row in log_history for key in row})
    if fieldnames:
        with (output_dir / "trainer_log_history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(log_history)


def _polyline(points: list[tuple[float, float]], *, width: int, height: int, pad: int) -> str:
    if not points:
        return ""
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_span = max(max_x - min_x, 1e-9)
    y_span = max(max_y - min_y, 1e-9)
    coords = []
    for x, y in points:
        sx = pad + (x - min_x) / x_span * (width - 2 * pad)
        sy = height - pad - (y - min_y) / y_span * (height - 2 * pad)
        coords.append(f"{sx:.1f},{sy:.1f}")
    return " ".join(coords)


def save_loss_curve(log_history: list[dict[str, Any]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    train_points = [(float(row.get("step", 0)), float(row["loss"])) for row in log_history if "loss" in row and row.get("step") is not None]
    eval_points = [
        (float(row.get("step", 0)), float(row["eval_loss"]))
        for row in log_history
        if "eval_loss" in row and row.get("step") is not None
    ]
    if not train_points and not eval_points:
        return
    width, height, pad = 900, 520, 60
    all_points = train_points + eval_points
    xs = [x for x, _ in all_points]
    ys = [y for _, y in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    train_line = _polyline(train_points, width=width, height=height, pad=pad)
    eval_line = _polyline(eval_points, width=width, height=height, pad=pad)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{pad}" y="34" font-family="Arial" font-size="22" font-weight="700">SFT Loss Curve</text>
  <text x="{pad}" y="{height - 18}" font-family="Arial" font-size="13">step {min_x:.0f} to {max_x:.0f}</text>
  <text x="{width - 220}" y="34" font-family="Arial" font-size="13" fill="#2563eb">train loss</text>
  <text x="{width - 120}" y="34" font-family="Arial" font-size="13" fill="#dc2626">eval loss</text>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#9ca3af"/>
  <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#9ca3af"/>
  <text x="10" y="{pad + 5}" font-family="Arial" font-size="12">{max_y:.4f}</text>
  <text x="10" y="{height - pad}" font-family="Arial" font-size="12">{min_y:.4f}</text>
"""
    if train_line:
        svg += f'  <polyline points="{train_line}" fill="none" stroke="#2563eb" stroke-width="2.5"/>\n'
    if eval_line:
        svg += f'  <polyline points="{eval_line}" fill="none" stroke="#dc2626" stroke-width="2.5"/>\n'
    svg += "</svg>\n"
    (output_dir / "loss_curve.svg").write_text(svg, encoding="utf-8")


def enable_debug_stack_dumps(config: dict[str, Any]) -> None:
    """Emit Python thread stacks when a configured training startup watchdog fires."""

    seconds = int(section(config, "training").get("debug_stack_dump_seconds", 0) or 0)
    if seconds <= 0:
        return
    faulthandler.enable()
    faulthandler.dump_traceback_later(seconds, repeat=True)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    LOG.info("Enabled training stack watchdog: every %d seconds; SIGUSR1 also dumps all Python threads", seconds)


def train(config_path: str) -> None:
    from datasets import Dataset
    from transformers import Trainer

    config = load_config(config_path)
    train_cfg = section(config, "training")
    enable_debug_stack_dumps(config)
    processor = load_processor(config)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = load_policy_model(config)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    token_ids = setup_schema_tokens(processor, model, config)
    model = prepare_lora_model(model, config)
    apply_schema_row_training(model, token_ids, config)

    LOG.info("Building SFT validation rows")
    eval_rows = build_eval_rows(config)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None
    LOG.info("Building SFT training rows")
    train_rows = build_sft_rows(config)
    LOG.info("Creating in-memory Hugging Face train dataset with %d row(s)", len(train_rows))
    dataset = Dataset.from_list(train_rows)
    LOG.info("Creating TrainingArguments")
    args = make_training_args(config, has_eval=eval_dataset is not None)
    data_cfg = section(config, "data")
    cache_dir = resolve_path(str(data_cfg.get("image_cache_dir", "./work/schema_sft_images")), base_dir=Path(config["_project_dir"]))
    image_cache = ImageCache(
        cache_dir or Path(config["_project_dir"]) / "work/schema_sft_images",
        int(data_cfg.get("image_timeout_seconds", 30)),
        retries=int(data_cfg.get("image_retries", 3)),
        retry_sleep_seconds=float(data_cfg.get("image_retry_sleep_seconds", 1.0)),
        min_pixels=data_cfg.get("image_min_pixels"),
        max_pixels=data_cfg.get("image_max_pixels"),
    )
    collator = VisionSftCollator(
        processor=processor,
        system_prompt=str(section(config, "data").get("system_prompt") or "").strip(),
        max_length=int(train_cfg.get("max_length", 4096)),
        image_cache=image_cache,
    )
    LOG.info("Constructing Hugging Face Trainer (dataloader_num_workers=%d)", args.dataloader_num_workers)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
    )
    LOG.info("Calling trainer.train(); planned max_steps=%d, num_train_epochs=%s", args.max_steps, args.num_train_epochs)
    train_result = trainer.train()
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        trainer.save_metrics("eval", eval_metrics)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    save_trainer_history(trainer.state.log_history, args.output_dir)
    save_loss_curve(trainer.state.log_history, args.output_dir)
    if token_ids and bool(section(config, "model").get("train_schema_token_rows", True)):
        rows_path = Path(args.output_dir) / "schema_token_rows.pt"
        save_schema_token_rows(model, schema_special_tokens(), rows_path)
        LOG.info("Saved schema token rows to %s", rows_path)
    LOG.info("Saved structured schema SFT output to %s", args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train structured schema SFT for main-image prompt generation.")
    parser.add_argument("--config", default="configs/sft.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    train(args.config)


if __name__ == "__main__":
    main()
