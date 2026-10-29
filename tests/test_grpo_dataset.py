from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from utils.data import build_dataset


class _NoTemplateProcessor:
    def apply_chat_template(self, *_args, **_kwargs):
        raise AssertionError("GRPO dataset should keep raw conversational prompts for TRL multimodal training")


class GrpoDatasetTest(unittest.TestCase):
    def test_build_dataset_keeps_raw_conversational_prompt_and_image_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "source.png"
            Image.new("RGB", (8, 8), "white").save(image_path)
            csv_path = root / "train.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "original_image_url",
                        "title",
                        "description",
                        "l3_category",
                        "grass_region",
                        "item_id",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "original_image_url": str(image_path),
                        "title": "Test product",
                        "description": "Short description",
                        "l3_category": "Category",
                        "grass_region": "SG",
                        "item_id": "123",
                    }
                )

            dataset = build_dataset(
                {
                    "_project_dir": str(root),
                    "data": {
                        "train_csv": str(csv_path),
                        "image_cache_dir": str(root / "cache"),
                        "system_prompt": "System text",
                        "user_template": "Title: {title}\nRegion: {grass_region}",
                    },
                },
                _NoTemplateProcessor(),
            )

            row = dataset[0]
            self.assertIsInstance(row["prompt"], list)
            self.assertEqual(row["prompt"][0], {"role": "system", "content": "System text"})
            self.assertEqual(row["prompt"][1]["role"], "user")
            self.assertIn("Title: Test product", row["prompt"][1]["content"])
            self.assertEqual(row["item_id"], "123")
            self.assertEqual(row["image"].size, (8, 8))

    def test_build_dataset_uses_resized_policy_image_when_max_pixels_is_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "large.png"
            Image.new("RGB", (100, 80), "white").save(image_path)
            csv_path = root / "train.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "original_image_url",
                        "title",
                        "description",
                        "l3_category",
                        "grass_region",
                        "item_id",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "original_image_url": str(image_path),
                        "title": "Large product",
                        "description": "Short description",
                        "l3_category": "Category",
                        "grass_region": "SG",
                        "item_id": "456",
                    }
                )

            dataset = build_dataset(
                {
                    "_project_dir": str(root),
                    "data": {
                        "train_csv": str(csv_path),
                        "image_cache_dir": str(root / "cache"),
                        "image_max_pixels": 2500,
                        "system_prompt": "",
                        "user_template": "Title: {title}",
                    },
                },
                _NoTemplateProcessor(),
            )

            row = dataset[0]
            self.assertEqual(row["original_image_url"], str(image_path))
            self.assertEqual(row["image"].size, (55, 44))
            self.assertTrue((root / "cache" / "policy_resized").exists())


if __name__ == "__main__":
    unittest.main()
