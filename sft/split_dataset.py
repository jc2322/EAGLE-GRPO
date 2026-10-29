from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalized_key(row: dict[str, str], item_column: str) -> str:
    item_id = str(row.get(item_column) or "").strip()
    if item_id:
        return item_id
    row_key = str(row.get("row_key") or "").strip()
    if row_key:
        return row_key
    raise ValueError(f"Row is missing both {item_column!r} and row_key")


def split_item_keys(
    keys: list[str],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("Split ratios must sum to a positive value")
    train_ratio, val_ratio, test_ratio = (train_ratio / total_ratio, val_ratio / total_ratio, test_ratio / total_ratio)
    shuffled = list(keys)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        raise ValueError("No item keys to split")
    train_n = int(round(n * train_ratio))
    val_n = int(round(n * val_ratio))
    if train_n + val_n > n:
        val_n = max(0, n - train_n)
    test_n = n - train_n - val_n
    if n >= 3:
        if train_n == 0:
            train_n, test_n = 1, max(0, test_n - 1)
        if val_ratio > 0 and val_n == 0:
            val_n = 1
            if test_n > 0:
                test_n -= 1
            else:
                train_n -= 1
        if test_ratio > 0 and test_n == 0:
            test_n = 1
            if train_n > 1:
                train_n -= 1
            else:
                val_n -= 1
    assignment: dict[str, str] = {}
    for key in shuffled[:train_n]:
        assignment[key] = "train"
    for key in shuffled[train_n : train_n + val_n]:
        assignment[key] = "val"
    for key in shuffled[train_n + val_n :]:
        assignment[key] = "test"
    return assignment


def summarize(rows_by_split: dict[str, list[dict[str, str]]], item_column: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split, rows in rows_by_split.items():
        item_ids = {normalized_key(row, item_column) for row in rows}
        strategies = Counter(str(row.get("selected_design_strategy") or "") for row in rows)
        variants = Counter(str(row.get("teacher_variant_id") or "") for row in rows)
        summary[split] = {
            "rows": len(rows),
            "items": len(item_ids),
            "selected_design_strategy": dict(sorted(strategies.items())),
            "teacher_variant_id": dict(sorted(variants.items())),
        }
    return summary


def split_dataset(
    input_csv: Path,
    output_dir: Path,
    *,
    item_column: str = "item_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    rows, fieldnames = read_csv(input_csv)
    if not rows:
        raise ValueError(f"No rows found in {input_csv}")
    if item_column not in fieldnames and "row_key" not in fieldnames:
        raise ValueError(f"{input_csv} must contain {item_column!r} or row_key")

    by_item: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_item[normalized_key(row, item_column)].append(row)

    assignment = split_item_keys(
        sorted(by_item),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    rows_by_split: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for item_key, item_rows in by_item.items():
        rows_by_split[assignment[item_key]].extend(item_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in rows_by_split.items():
        write_csv(output_dir / f"{split}.csv", split_rows, fieldnames)

    summary = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "item_column": item_column,
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "total_rows": len(rows),
        "total_items": len(by_item),
        "splits": summarize(rows_by_split, item_column),
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Split schema SFT data by item_id into train/val/test CSVs.")
    parser.add_argument("--input-csv", default="data/sft/responses.csv")
    parser.add_argument("--output-dir", default="data/schema_sft")
    parser.add_argument("--item-column", default="item_id")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = split_dataset(
        Path(args.input_csv).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
        item_column=args.item_column,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
