from __future__ import annotations

import csv
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image, ImageFile

from utils.config import resolve_path, section

ImageFile.LOAD_TRUNCATED_IMAGES = True


LOG = logging.getLogger(__name__)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def non_empty(row: dict[str, Any], key: str) -> str:
    value = str(row.get(key) or "").strip()
    return "" if value.lower() in {"", "nan", "none", "null"} else value


class ImageCache:
    def __init__(
        self,
        directory: str | Path,
        timeout_seconds: int = 30,
        *,
        retries: int = 3,
        retry_sleep_seconds: float = 1.0,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = int(timeout_seconds)
        self.retries = max(1, int(retries))
        self.retry_sleep_seconds = float(retry_sleep_seconds)
        self.min_pixels = int(min_pixels) if min_pixels else None
        self.max_pixels = int(max_pixels) if max_pixels else None

    def materialize(self, image_ref: str) -> Path:
        if not image_ref.startswith(("http://", "https://")):
            path = Path(image_ref).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            return path
        suffix = Path(urlparse(image_ref).path).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        path = self.directory / f"{hashlib.sha256(image_ref.encode()).hexdigest()}{suffix}"
        if not path.exists() or path.stat().st_size == 0:
            last_error: Exception | None = None
            for attempt in range(self.retries):
                try:
                    response = requests.get(
                        image_ref,
                        timeout=self.timeout_seconds,
                        headers={"User-Agent": "main-image-grpo/1.0"},
                    )
                    response.raise_for_status()
                    break
                except requests.RequestException as exc:
                    last_error = exc
                    if attempt + 1 < self.retries:
                        time.sleep(self.retry_sleep_seconds * (attempt + 1))
            else:
                raise RuntimeError(f"Failed to download image after {self.retries} attempt(s): {image_ref}") from last_error
            tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
            try:
                tmp_path.write_bytes(response.content)
                tmp_path.replace(path)
            finally:
                tmp_path.unlink(missing_ok=True)
        with Image.open(path) as image:
            image.verify()
        return path

    def materialize_for_policy(self, image_ref: str) -> Path:
        path = self.materialize(image_ref)
        if not self.min_pixels and not self.max_pixels:
            return path
        with Image.open(path) as image:
            loaded = image.convert("RGB")
        resized = self._resize_if_needed(loaded)
        if resized.size == loaded.size:
            return path
        suffix = path.suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        fingerprint = "|".join(
            [
                image_ref,
                f"min={self.min_pixels or ''}",
                f"max={self.max_pixels or ''}",
                f"size={resized.size[0]}x{resized.size[1]}",
            ]
        )
        output_dir = self.directory / "policy_resized"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{hashlib.sha256(fingerprint.encode()).hexdigest()}{suffix}"
        if output_path.exists() and output_path.stat().st_size > 0:
            with Image.open(output_path) as image:
                image.verify()
            return output_path
        tmp_path = output_path.with_name(f"{output_path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            save_kwargs: dict[str, Any] = {}
            if suffix in {".jpg", ".jpeg", ".webp"}:
                save_kwargs["quality"] = 95
            save_format = {
                ".jpg": "JPEG",
                ".jpeg": "JPEG",
                ".png": "PNG",
                ".webp": "WEBP",
            }[suffix]
            resized.save(tmp_path, format=save_format, **save_kwargs)
            tmp_path.replace(output_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return output_path

    def load(self, image_ref: str) -> Image.Image:
        path = self.materialize(image_ref)
        with Image.open(path) as image:
            loaded = image.convert("RGB")
        return self._resize_if_needed(loaded)

    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        pixels = width * height
        if self.max_pixels and pixels > self.max_pixels:
            scale = (self.max_pixels / float(pixels)) ** 0.5
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            return image.resize(new_size, Image.Resampling.LANCZOS)
        if self.min_pixels and pixels < self.min_pixels:
            scale = (self.min_pixels / float(pixels)) ** 0.5
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            return image.resize(new_size, Image.Resampling.BICUBIC)
        return image


def build_user_prompt(config: dict[str, Any], row: dict[str, Any]) -> str:
    data_cfg = section(config, "data")
    template = str(data_cfg["user_template"])
    return template.format(
        title=non_empty(row, str(data_cfg.get("title_column", "title"))),
        description=non_empty(row, str(data_cfg.get("description_column", "description"))),
        category=non_empty(row, str(data_cfg.get("category_column", "l3_category"))),
        grass_region=non_empty(row, str(data_cfg.get("region_column", "grass_region"))),
    )


def build_dataset(config: dict[str, Any], processor: Any):
    from datasets import Dataset, Features, Image as DatasetImage, Value

    project_dir = Path(config["_project_dir"])
    data_cfg = section(config, "data")
    csv_path = resolve_path(str(data_cfg["train_csv"]), base_dir=project_dir)
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")
    rows = read_csv(csv_path)
    if data_cfg.get("max_samples"):
        rows = rows[: int(data_cfg["max_samples"])]
    LOG.info("Building GRPO dataset from %s with %d raw row(s)", csv_path, len(rows))
    cache_dir = resolve_path(str(data_cfg.get("image_cache_dir", "./work/image_cache")), base_dir=project_dir)
    image_cache = ImageCache(
        cache_dir or project_dir / "work/image_cache",
        int(data_cfg.get("image_timeout_seconds", 30)),
        retries=int(data_cfg.get("image_retries", 3)),
        retry_sleep_seconds=float(data_cfg.get("image_retry_sleep_seconds", 1.0)),
        min_pixels=data_cfg.get("image_min_pixels"),
        max_pixels=data_cfg.get("image_max_pixels"),
    )

    system_prompt = str(data_cfg.get("system_prompt") or "").strip()
    image_column = str(data_cfg.get("image_column", "original_image_url"))
    item_id_column = str(data_cfg.get("item_id_column", "item_id"))
    dataset_rows: list[dict[str, Any]] = []
    log_every = max(1, int(data_cfg.get("dataset_log_every", 100) or 100))
    for index, row in enumerate(rows):
        if index == 0 or (index + 1) % log_every == 0:
            LOG.info("Materializing GRPO dataset row %d/%d", index + 1, len(rows))
        image_ref = non_empty(row, image_column)
        title = non_empty(row, str(data_cfg.get("title_column", "title")))
        if not image_ref or not title:
            continue
        market = non_empty(row, str(data_cfg.get("market_column", "market")))
        category_l1 = non_empty(row, str(data_cfg.get("category_l1_column", "category_l1")))
        if config.get("weight_head") and (not market or not category_l1):
            raise ValueError(f"Row {index}: market and category_l1 are required for contextual rewards")
        image_path = image_cache.materialize_for_policy(image_ref)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": build_user_prompt(config, row)})
        dataset_rows.append(
            {
                "prompt": messages,
                "image": str(image_path),
                "row_key": non_empty(row, "row_key") or f"row-{index:08d}",
                "item_id": non_empty(row, item_id_column),
                "market": market,
                "category_l1": category_l1,
                "original_image_url": image_ref,
                "title": title,
                "description": non_empty(row, str(data_cfg.get("description_column", "description"))),
                "category": non_empty(row, str(data_cfg.get("category_column", "l3_category"))),
                "grass_region": non_empty(row, str(data_cfg.get("region_column", "grass_region"))),
            }
        )
    if not dataset_rows:
        raise ValueError(f"No usable rows found in {csv_path}")
    LOG.info("Built GRPO dataset with %d usable row(s)", len(dataset_rows))
    features = Features(
        {
            "prompt": [
                {
                    "role": Value("string"),
                    "content": Value("string"),
                }
            ],
            "image": DatasetImage(),
            "row_key": Value("string"),
            "item_id": Value("string"),
            "market": Value("string"),
            "category_l1": Value("string"),
            "original_image_url": Value("string"),
            "title": Value("string"),
            "description": Value("string"),
            "category": Value("string"),
            "grass_region": Value("string"),
        }
    )
    return Dataset.from_list(dataset_rows, features=features)
