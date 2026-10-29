"""11-criterion version of the V2-M8 market/category residual-gating model."""
from pathlib import Path
import math
import torch
from torch import nn
from rewards.schema import DIMENSIONS, context_key

FORMAT_VERSION = 1

class M8WeightHead(nn.Module):
    def __init__(self, market_count, category_count, embedding_dim=8, initial_gate=0.2):
        super().__init__()
        if min(market_count, category_count, embedding_dim) < 1 or not 0 < initial_gate < 1:
            raise ValueError('Invalid vocabulary, embedding dimension, or gate initialization')
        self.embedding_dim = int(embedding_dim)
        self.global_weights = nn.Parameter(torch.zeros(len(DIMENSIONS)))
        self.market_embedding = nn.Embedding(market_count, embedding_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(category_count, embedding_dim, padding_idx=0)
        for layer in (self.market_embedding, self.category_embedding):
            nn.init.normal_(layer.weight, std=0.02)
            with torch.no_grad():
                layer.weight[0].zero_()
        self.residual = nn.Linear(2 * embedding_dim, len(DIMENSIONS), bias=False)
        self.gate = nn.Linear(2 * embedding_dim, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(initial_gate / (1 - initial_gate)))

    def coefficients(self, markets, categories):
        e = torch.cat((self.market_embedding(markets), self.category_embedding(categories)), dim=-1)
        return self.global_weights + self.gate(e).sigmoid() * self.residual(e)

    def forward(self, scores_a, scores_b, markets, categories):
        weights = self.coefficients(markets, categories)
        return (((scores_a - scores_b) / 10.0) * weights).sum(-1), weights


def build_vocab(records):
    result = {}
    for key in ('market', 'category_l1'):
        values = sorted({context_key(r[key], key) for r in records})
        result[key] = {value: i + 1 for i, value in enumerate(values)}
    return result


def context_indices(vocab, market, category_l1, *, unknown='error'):
    if unknown not in {'error', 'zero_embedding'}:
        raise ValueError('unknown_context must be error or zero_embedding')
    keys = (context_key(market, 'market'), context_key(category_l1, 'category_l1'))
    indices = []
    for name, key in zip(('market', 'category_l1'), keys):
        if key not in vocab[name] and unknown == 'error':
            raise ValueError(f'Unseen {name}: {key!r}; supply a compatible trained head')
        indices.append(vocab[name].get(key, 0))
    return tuple(indices)


def save_checkpoint(path, model, vocab, *, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'format_version': FORMAT_VERSION, 'architecture': 'M8ResidualGating11',
        'dimensions': list(DIMENSIONS), 'score_scale': 10.0,
        'embedding_dim': model.embedding_dim, 'vocab': vocab,
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'metadata': metadata or {},
    }, path)


def load_checkpoint(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if (data.get('format_version') != FORMAT_VERSION or
        data.get('architecture') != 'M8ResidualGating11' or
        data.get('dimensions') != list(DIMENSIONS) or data.get('score_scale') != 10.0):
        raise ValueError('Incompatible checkpoint: expected ordered 11-dimension M8 release format')
    vocab = data['vocab']
    for name in ('market', 'category_l1'):
        ids = sorted(vocab[name].values())
        if ids != list(range(1, len(ids) + 1)):
            raise ValueError(f'Invalid vocabulary indices for {name}')
        if any(context_key(k, name) != k for k in vocab[name]):
            raise ValueError(f'Vocabulary {name} is not canonical')
    model = M8WeightHead(len(vocab['market']) + 1, len(vocab['category_l1']) + 1,
                         embedding_dim=int(data['embedding_dim']))
    model.load_state_dict(data['state_dict'], strict=True)
    for value in model.state_dict().values():
        if not torch.isfinite(value).all():
            raise ValueError('Checkpoint contains nonfinite parameters')
    return model, vocab, data.get('metadata', {})
