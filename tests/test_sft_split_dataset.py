from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sft.split_dataset import split_dataset

try:
    from sft.train_schema_sft import build_sft_rows_from_csv
except ModuleNotFoundError:  # pragma: no cover - lightweight local env
    build_sft_rows_from_csv = None


class SftSplitDatasetTest(unittest.TestCase):
    def test_split_dataset_keeps_all_variants_for_item_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_csv = tmp_path / "schema_sft_train.csv"
            fieldnames = ["item_id", "row_key", "teacher_variant_id", "selected_design_strategy", "prompt"]
            rows = []
            for item_index in range(10):
                for variant_index in range(4):
                    rows.append(
                        {
                            "item_id": f"item-{item_index}",
                            "row_key": f"row-{item_index}",
                            "teacher_variant_id": f"variant_{variant_index + 1:03d}",
                            "selected_design_strategy": "balanced_clean",
                            "prompt": "<thinking>x</thinking><final_prompt></final_prompt>",
                        }
                    )
            with input_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            summary = split_dataset(input_csv, tmp_path / "splits", seed=123)
            self.assertEqual(summary["total_items"], 10)
            self.assertEqual(summary["total_rows"], 40)
            seen: dict[str, str] = {}
            for split in ("train", "val", "test"):
                with (tmp_path / "splits" / f"{split}.csv").open("r", encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        previous = seen.setdefault(row["item_id"], split)
                        self.assertEqual(previous, split)
            self.assertEqual(len(seen), 10)
            self.assertTrue((tmp_path / "splits" / "split_summary.json").exists())

    def test_build_sft_rows_skips_bad_images_when_configured(self) -> None:
        if build_sft_rows_from_csv is None:
            self.skipTest("torch is not installed in this environment")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_csv = tmp_path / "train.csv"
            fieldnames = [
                "row_key",
                "item_id",
                "original_image_url",
                "title",
                "description",
                "grass_region",
                "global_category1",
                "global_category2",
                "global_category3",
                "prompt",
            ]
            prompt = (
                "<thinking>Short reasoning about the product and safe layout.</thinking>\n\n"
                "<final_prompt>"
                "<preservation_constraints>Preserve product.</preservation_constraints>"
                "<product_presentation>Show product clearly.</product_presentation>"
                "<headline>Main title.</headline>"
                "<subtitle></subtitle>"
                "<feature_callouts></feature_callouts>"
                "<specification_information></specification_information>"
                "<quantity_bundle_information></quantity_bundle_information>"
                "<variant_information></variant_information>"
                "<price_promotion_information></price_promotion_information>"
                "<seller_logistics_information></seller_logistics_information>"
                "<trust_information></trust_information>"
                "<usage_information></usage_information>"
                "<comparison_visual></comparison_visual>"
                "<supporting_visuals></supporting_visuals>"
                "<composition_layout>Centered layout.</composition_layout>"
                "<scene_style>Clean background.</scene_style>"
                "<lighting_rendering>Soft light.</lighting_rendering>"
                "<negative_constraints>No unsupported claims.</negative_constraints>"
                "</final_prompt>"
            )
            with input_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "row_key": "row-1",
                        "item_id": "item-1",
                        "original_image_url": "https://example.invalid/image.jpg",
                        "title": "Product",
                        "description": "",
                        "grass_region": "SG",
                        "global_category1": "",
                        "global_category2": "",
                        "global_category3": "",
                        "prompt": prompt,
                    }
                )
            config = {
                "_project_dir": str(tmp_path),
                "data": {
                    "train_csv": str(input_csv),
                    "image_column": "original_image_url",
                    "title_column": "title",
                    "description_column": "description",
                    "category_column": "global_category3",
                    "region_column": "grass_region",
                    "prompt_column": "prompt",
                    "image_cache_dir": str(tmp_path / "cache"),
                    "skip_bad_images": True,
                },
            }
            with patch("sft.train_schema_sft.ImageCache.materialize", side_effect=RuntimeError("dns failed")):
                with self.assertRaisesRegex(ValueError, "No usable SFT rows"):
                    build_sft_rows_from_csv(config, "train_csv")


if __name__ == "__main__":
    unittest.main()
