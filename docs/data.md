# Input contracts

No training/evaluation records, images, or checkpoints are included. Example IDs
and values here are synthetic. Use one stable naming convention for category IDs
in every training and scoring input; do not substitute an L3 category for L1.

## Prompt-writer SFT and GRPO

CSV columns used by the supplied configurations:

| Column | Meaning |
|---|---|
| `item_id` | Stable product ID; use globally unique IDs for product-level splitting |
| `original_image_url` | Image URL or local image path |
| `title` | Product title |
| `description` | Optional listing text |
| `market` | Market identifier; the weight head normalizes it to uppercase |
| `category_l1` | Top-level category identifier; must match weight-head vocabulary |
| `prompt` | SFT only: teacher `<thinking>…</thinking><final_prompt>…</final_prompt>` completion |
| `row_key` | Optional GRPO rollout-group key, unique per input product row |

All 18 child tags in `prompts/schema.py` are emitted in order. The seven required
tags have content; the eleven optional tags may be empty. The two constraint tags
are retained for image editing but excluded from element-credit mapping.

`data.market_column` and `data.category_l1_column` can map differently named input
columns. These columns are carried into the reward request as `market` and
`category_l1`; the descriptive `category` and legacy `grass_region` fields remain
available for prompt formatting. Missing required context stops GRPO data loading.

The product split helper groups by `item_id`. If identifiers are only unique
within a market, build a composite ID before splitting. Split verification and
image-level near-duplicate checks remain the responsibility of the data preparer.

## CTR preference pairs

Each JSONL record has this structure (score dictionaries are preferred over lists):

```json
{
  "product_id": "synthetic-product-1",
  "market": "AA",
  "category_l1": "Home",
  "scores_a": {
    "product_fidelity": 8, "product_salience": 7,
    "eye_catching_attention": 7, "creative_selling_point": 7,
    "color_harmony": 8, "layout_hierarchy": 8,
    "background_scene_fit": 7, "visual_communication": 8,
    "text_readability": 8, "commercial_desire": 7,
    "professional_polish": 8
  },
  "scores_b": [7, 6, 6, 6, 7, 7, 6, 7, 7, 6, 7],
  "target": 0.75,
  "sample_weight": 1.0
}
```

`target` is the posterior probability that **image A has higher CTR than image B**,
computed upstream from the historical experiment's click/impression observations.
It is neither raw CTR nor relative CTR uplift. If swapping A and B, swap score
vectors and replace `target` by `1-target`. Supply posterior probabilities from
your documented experiment estimator; this release does not invent priors or
convert uplift percentages into probabilities. `sample_weight` is positive and
defaults to 1. The trainer normalizes these weights within each product.

Pairs must compare images for the same product/context. Stage 1 uses original
versus optimized images; stage 2 uses two optimized images. Validation products
must be absent from both stages' training sets. The loader checks this, along with
context consistency and finite score/target ranges. It does not load a final test
set or perform architecture selection on it. Full-batch M8 training mirrors the
reference approach; choose CPU/GPU and memory capacity for your data size.

The authoritative order, also serialized in checkpoints, is:

1. `product_fidelity`
2. `product_salience`
3. `eye_catching_attention`
4. `creative_selling_point`
5. `color_harmony`
6. `layout_hierarchy`
7. `background_scene_fit`
8. `visual_communication`
9. `text_readability`
10. `commercial_desire`
11. `professional_polish`

All input scores are on a 0–10 scale. Pairwise training divides the score
difference by ten internally; do not pre-normalize the input a second time.

## Scoring input

`rewards.score` accepts JSONL records with `item_id` (optional), `market`,
`category_l1`, and `judge`. `judge` contains all eleven `<dimension>_score` keys
and the six pass/fail fields:

- `product_identity_gate`
- `product_count_gate`
- `variant_gate`
- `logo_package_text_gate`
- `unsupported_claim_gate`
- `severe_visual_damage_gate`

A hard-failed candidate may omit scores, but all gate outcomes must be explicit.
Missing gates, invalid scores, and unknown contexts are not silently mapped to
zero reward. `hard_fail`, when provided, must be a JSON boolean.

`rewards.judge_images` instead accepts `original_image_url`,
`generated_image_url`, `market`, `category_l1`, `title`, and `description`.
Image references can be local files or accessible URLs. `compiled_prompt` is
optional. Do not include method identities in judge-visible listing metadata.
Output `overall` is in the component score units; `reward` is `overall/10`.
