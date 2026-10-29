"""Frozen contextual aggregation, shared by RL reward and offline scoring."""
from functools import lru_cache
from pathlib import Path
import math
import torch
from rewards.schema import DIMENSIONS, SCORE_FIELDS, FIDELITY_GATES, score_vector, context_key
from weight_head.model import load_checkpoint, context_indices

class ContextualImageScorer:
    def __init__(self, checkpoint, *, unknown_context='error', denominator_epsilon=1e-8):
        self.model, self.vocab, _ = load_checkpoint(checkpoint)
        self.model.eval().requires_grad_(False)
        if unknown_context not in {'error', 'zero_embedding'}:
            raise ValueError('Invalid unknown_context policy')
        self.unknown_context = unknown_context
        self.epsilon = float(denominator_epsilon)
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError('denominator_epsilon must be positive and finite')

    @classmethod
    def from_config(cls, config):
        cfg = config.get('weight_head', {})
        path = cfg.get('checkpoint')
        if not path:
            raise ValueError('weight_head.checkpoint is required; fixed-weight fallback is disabled')
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = Path(config.get('_project_dir', '.')) / path
        return cls(path, unknown_context=cfg.get('unknown_context', 'error'),
                   denominator_epsilon=cfg.get('denominator_epsilon', 1e-8))

    @lru_cache(maxsize=16384)
    def weights(self, market, category_l1):
        mi, ci = context_indices(self.vocab, market, category_l1, unknown=self.unknown_context)
        with torch.inference_mode():
            w = self.model.coefficients(torch.tensor([mi]), torch.tensor([ci]))[0].double()
        total = float(w.sum())
        # Signed coefficients are retained, not clipped or softmaxed. A positive
        # denominator preserves the preference-logit direction during aggregation.
        if not torch.isfinite(w).all() or not math.isfinite(total) or total <= self.epsilon:
            raise ValueError('Weight sum must be positive and safely nonzero for reward aggregation')
        return tuple(float(x) for x in w)

    def score(self, payload, context):
        market = context_key(context.get('market'), 'market')
        category = context_key(context.get('category_l1'), 'category_l1')
        weights = self.weights(market, category)
        gates = {key: str(payload.get(key, '')).strip().lower() for key in FIDELITY_GATES}
        if any(value not in {'pass', 'fail'} for value in gates.values()):
            raise ValueError('All six fidelity gates must explicitly be pass or fail')
        hard_fail = payload.get('hard_fail', False)
        if not isinstance(hard_fail, bool):
            raise ValueError('hard_fail must be a JSON boolean')
        passed = not hard_fail and all(value == 'pass' for value in gates.values())
        # Judges may omit component scores on a hard failure, but not on a pass.
        scores = score_vector({name: payload[name] for name in SCORE_FIELDS}) if passed else None
        overall = math.fsum(w * s for w, s in zip(weights, scores)) / math.fsum(weights) if passed else 0.0
        if not math.isfinite(overall):
            raise ValueError('Nonfinite contextual image score')
        return {'market': market, 'category_l1': category, 'dimensions': list(DIMENSIONS),
                'weights': list(weights), 'scores': scores, 'passed_gates': passed,
                'overall': overall, 'reward': overall / 10.0}
