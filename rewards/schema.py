"""The single score order shared by judges, CTR training, and reward aggregation."""
import math
from collections.abc import Mapping

DIMENSIONS = (
    'product_fidelity', 'product_salience', 'eye_catching_attention',
    'creative_selling_point', 'color_harmony', 'layout_hierarchy',
    'background_scene_fit', 'visual_communication', 'text_readability',
    'commercial_desire', 'professional_polish',
)
SCORE_FIELDS = tuple(name + '_score' for name in DIMENSIONS)
FIDELITY_GATES = (
    'product_identity_gate', 'product_count_gate', 'variant_gate',
    'logo_package_text_gate', 'unsupported_claim_gate', 'severe_visual_damage_gate',
)

def score_vector(scores):
    """Named mappings are preferred; positional vectors follow DIMENSIONS exactly."""
    if isinstance(scores, Mapping):
        if set(scores) == set(DIMENSIONS):
            values = [scores[k] for k in DIMENSIONS]
        elif set(scores) == set(SCORE_FIELDS):
            values = [scores[k] for k in SCORE_FIELDS]
        else:
            raise ValueError('Scores must contain exactly the 11 named quality dimensions')
    else:
        values = list(scores)
    if len(values) != len(DIMENSIONS):
        raise ValueError('Expected 11 quality scores; legacy 8-dimensional scores are incompatible')
    values = [float(x) for x in values]
    if any(not math.isfinite(x) or not 0 <= x <= 10 for x in values):
        raise ValueError('Quality scores must be finite and in [0, 10]')
    return values


def context_key(value, name):
    if value is None or isinstance(value, bool):
        raise ValueError(f'{name} is required')
    text = str(value).strip()
    if text.lower() in {'', 'nan', 'none', 'null'}:
        raise ValueError(f'{name} is required')
    # Both release training and serving use the same canonical market spelling.
    return text.upper() if name == 'market' else text
