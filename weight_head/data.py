"""Public pair schema; real CTR records and judge outputs are not distributed."""
import json
import math
from pathlib import Path
import torch
from rewards.schema import score_vector, context_key
from weight_head.model import context_indices


def load_pairs(path):
    path = Path(path)
    if path.suffix == '.parquet':
        import pandas as pd
        raw = pd.read_parquet(path).to_dict('records')
    else:
        with path.open() as handle:
            raw = [json.loads(line) for line in handle if line.strip()]
    if not raw:
        raise ValueError(f'Empty pair dataset: {path}')
    records = []
    for i, row in enumerate(raw):
        row = dict(row)
        for key in ('product_id', 'market', 'category_l1'):
            row[key] = context_key(row.get(key), key)
        row['scores_a'] = score_vector(row['scores_a'])
        row['scores_b'] = score_vector(row['scores_b'])
        row['target'] = float(row['target'])
        row['sample_weight'] = float(row.get('sample_weight', 1.0))
        if not math.isfinite(row['target']) or not 0 <= row['target'] <= 1:
            raise ValueError(f'Row {i}: target must be P(CTR_A > CTR_B), in [0, 1]')
        if not math.isfinite(row['sample_weight']) or row['sample_weight'] <= 0:
            raise ValueError(f'Row {i}: sample_weight must be positive and finite')
        records.append(row)
    return records


def product_key(row):
    # IDs may be reused across markets, but a product cannot cross L1 categories.
    return row['market'], row['product_id']


def validate_splits(pretrain, finetune, pretrain_validation, finetune_validation):
    training = {product_key(r) for r in pretrain + finetune}
    validation = {product_key(r) for r in pretrain_validation + finetune_validation}
    if training & validation:
        raise ValueError('Training/validation products overlap across stages')
    contexts = {}
    for row in pretrain + finetune + pretrain_validation + finetune_validation:
        key = product_key(row)
        if key in contexts and contexts[key] != row['category_l1']:
            raise ValueError('Inconsistent L1 category for the same market/product')
        contexts[key] = row['category_l1']


def tensor_batch(records, vocab, device, *, unknown='error'):
    ids = {}
    product_ids = []
    for row in records:
        key = product_key(row)
        if key not in ids:
            ids[key] = len(ids)
        product_ids.append(ids[key])
    context = [context_indices(vocab, r['market'], r['category_l1'], unknown=unknown) for r in records]
    floating = lambda values: torch.tensor(values, dtype=torch.float32, device=device)
    integer = lambda values: torch.tensor(values, dtype=torch.long, device=device)
    return {'scores_a': floating([r['scores_a'] for r in records]),
            'scores_b': floating([r['scores_b'] for r in records]),
            'target': floating([r['target'] for r in records]),
            'sample_weight': floating([r['sample_weight'] for r in records]),
            'markets': integer([x[0] for x in context]), 'categories': integer([x[1] for x in context]),
            'product_index': integer(product_ids), 'products': len(ids)}


def product_macro_mean(values, batch):
    totals = values.new_zeros(batch['products'])
    weights = values.new_zeros(batch['products'])
    totals.index_add_(0, batch['product_index'], values * batch['sample_weight'])
    weights.index_add_(0, batch['product_index'], batch['sample_weight'])
    return (totals / weights).mean()
