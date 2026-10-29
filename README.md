# EAGLE-GRPO

Training code for **EAGLE-GRPO: Element-Aware Group Learning for E-Commerce Image
Optimization**.

The release contains structured prompt-writer SFT, SFT adapter merging,
element-aware GRPO, and an 11-dimensional CTR-supervised contextual weight head.
It excludes proprietary datasets, trained checkpoints, experiment outputs, the
paper sources, and benchmark reports.

## Pipeline

1. Train the structured prompt writer with SFT. Learn LoRA parameters and the
   input/output embedding rows for the schema tags, then merge them into a full model.
2. Train the M8 weight head on original/optimized image pairs, then fine-tune it
   on optimized/optimized pairs. Freeze the selected head.
3. Sample CoT text and tagged editing prompts from the SFT-initialized writer.
   A frozen image editor renders each prompt. The VLM judge scores CoT and images.
4. For each image, obtain **11 weights from its market and L1 category**. The
   same market/category combination uses the same weights for every candidate.
   Compute the contextual weighted mean of the 11 image scores, with fidelity gates.
5. EAGLE decomposes the centered **image reward** using field kernels, retains
   unmapped credit in the residual, and maps credit to present design-token spans.
   CoT advantage updates CoT content independently. Update a fresh GRPO LoRA adapter.

The original V2-M8 implementation used a different, eight-dimensional score
schema. This release adapts its residual-gating architecture and two-stage
training objective to the paper's **11 quality dimensions**. An old eight-dimensional
checkpoint is not compatible and is rejected. Train a new head with aligned
11-dimensional judge scores; this repository does not ship or claim to reproduce
historical model results without the original data and checkpoints.

## Layout

| Directory | Contents |
|---|---|
| `sft/` | Product-level splitting, structured SFT, adapter/schema-row merging |
| `grpo/` | Policy-training entry point |
| `utils/` | Kernel credit, token masks, trainer integration, model loading, local editor worker |
| `weight_head/` | M8 architecture, pair data validation, two-stage training, checkpoint format |
| `rewards/` | Judge prompts, public editor/API adapters, shared contextual scorer |
| `prompts/` | 18-field schema: 16 design fields and two constraint fields |
| `configs/` | SFT, GPT-Image-2 GRPO, FLUX.2 [klein] 9B GRPO, weight-head recipes |
| `tests/` | CPU mathematical, data-interface, checkpoint, and integration tests |

## Installation

Use a separate Python environment. Install PyTorch/CUDA appropriate for your
hardware, then install this checkout in editable mode:

```bash
python -m pip install -e '.[train,test]'
# Default GRPO configs use colocated vLLM rollout:
python -m pip install -e '.[vllm]'
```

The release pins TRL 1.6.0 and vLLM 0.13.0. CPU checks and trainer/config imports
were exercised with PyTorch 2.9.0, Transformers 4.57.3, PEFT 0.19.1, and
Accelerate 1.4.0. vLLM 0.13.0 depends on PyTorch 2.9.0; install compatible CUDA
builds together. The inherited trainer also contains the older TRL 0.24 loss
path, but it is not the default installation. Do not mix arbitrary TRL/vLLM
versions. To use Transformers generation instead, set `generation.use_vllm: false`.

For local FLUX, use a separate editor environment with the `flux` extra and a
Diffusers release exposing `Flux2KleinPipeline`. Set
`IMAGE_EDIT_WORKER_PYTHON` to that environment's Python executable. Assign the
editor to an available GPU or configure CPU offloading in `configs/grpo_flux.yaml`.
The worker inherits GPU visibility; it is not automatically placed on another GPU.
Configure GPU memory and batch sizes for your hardware.

All commands below run from the repository root. Model names can be Hugging Face
IDs or local model directories. No model downloads, image generation, or API
requests occur when importing the library or running unit tests.

## 1. Structured SFT and merge

Prepare CSV input as described in [the data contract](docs/data.md). `prompt`
contains the teacher completion, including `<thinking>` and `<final_prompt>`.
Writer instructions are stored as readable blocks in `configs/sft.yaml`.

```bash
python -m sft.split_dataset \
  --input-csv data/sft/responses.csv --output-dir data/sft --seed 42
python -m sft.train_schema_sft --config configs/sft.yaml
python -m sft.merge_sft_to_full_model \
  --config configs/sft.yaml \
  --adapter-path outputs/main_image_structured_sft \
  --output-dir models/sft_merged
```

The merge includes the schema-token input embeddings and untied output-layer
rows (`schema_token_rows.pt`) as well as LoRA parameters. GRPO starts from this
merged model with a **new** LoRA adapter. Do not discard the added token rows or
continue training the SFT adapter in place.

## 2. Train the contextual weight head

Provide four JSONL files in `configs/weight_head.yaml`: training and validation
pairs for each stage. Parquet is optional (`pip install -e '.[parquet]'`).
Each pair contains product ID, market, L1 category, two aligned 11-dimensional
VLM score vectors in `[0,10]`, a CTR preference probability, and an optional
positive sample weight. See [data.md](docs/data.md) for exact fields.

```bash
python -m weight_head.train --config configs/weight_head.yaml
```

The head concatenates eight-dimensional market/category embeddings and computes

```text
w(context) = w0 + sigmoid(v^T e(context) + b) W e(context)
p(A preferred to B) = sigmoid(w(context)^T (scores_A - scores_B) / 10)
```

Training uses weighted soft-label BCE averaged within each product, then equally
across products, plus global/residual L2 penalties. The second stage starts from
the first stage's validation-selected parameters with a fresh AdamW optimizer.
Both checkpoints are selected on validation BCE, not on test data. The supplied
learning rates/penalties are a starting recipe, not a new empirical result.
Validation products must be disjoint from **both** training stages.

`outputs/weight_head/finetune.pt` stores the model, exact dimension order, score
scale, context vocabularies, and recipe. The scorer validates these before use.
Unknown contexts raise an error by default. An explicit `zero_embedding` policy
is available, but missing context fields always raise an error.

M8 coefficients are **signed**, as in the paper. Inference computes
`Overall = sum(w_j * score_j) / sum(w_j)` and `image_reward = Overall / 10`.
The release rejects nonfinite or non-positive/near-zero total weights, rather
than silently replacing them with uniform weights, clipping them, or applying
softmax. Signed weights do not mathematically guarantee an Overall in `[0,10]`;
validate the head's weight sums and score range before deployment. Fidelity-gate
failures receive zero image reward; missing/malformed judge observations are errors.

## 3. EAGLE-GRPO training

Configure the head checkpoint under `weight_head.checkpoint`. Supply your own
API credentials via environment variables; `.env.example` lists the names.
The public adapters use OpenAI-compatible chat judging and the image-edit API;
there are no internal gateways or media-upload dependencies.

```bash
export JUDGE_BASE_URL='https://generativelanguage.googleapis.com/v1beta/openai/'
# Set JUDGE_API_KEY securely in your shell/session.
# For GPT-Image-2, also set IMAGE_API_KEY and optionally IMAGE_BASE_URL.
python -m grpo.train --config configs/grpo_gpt.yaml

# Or, with a configured local editor worker:
python -m grpo.train --config configs/grpo_flux.yaml
```

The example recipes use 20 completions/product, generation batches of 80,
`eta=0.5`, `lambda_reg=0.1`, learning rate `1e-6`, and seed 42. Update budgets are
200 for GPT-Image-2 and 270 for FLUX.2 [klein] 9B. Both reward branches default to
Gemini-2.5-Flash. CoT and image token updates remain separate; the auxiliary
0.1/0.9 scalar is used for logging and normalized malformed-completion penalties,
not as a replacement for the two token-level reward branches.

No extra editor/judge calls are used for kernel credit assignment. CoT tokens,
constraint fields, delimiters, and design spans retain their distinct mask rules.
Absent fields receive no tokens; their unmapped credit stays in the residual.
If hidden-state extraction fails for only part of a field group, the entire
field group uses hashed representations rather than mixing incompatible spaces.
`eta=0` disables element redistribution within the same EAGLE filtering path.
The separate `utils.decoupled_advantage.DecoupledAdvantageProvider` can be selected
as a uniform-image-advantage control; its normalization population can differ on
invalid or no-design completions.

## Shared scoring at evaluation time

Evaluation uses **the same `ContextualImageScorer` as RL**, the same head
checkpoint, the same score order, and the same fidelity gates. It does not retrain
the head or use a different context-weight table. Two small entry points are
provided; neither includes a benchmark or historical results.

For already computed judge JSON:

```bash
python -m rewards.score --checkpoint outputs/weight_head/finetune.pt \
  --input data/judge_outputs.jsonl --output outputs/scored.jsonl
```

For source/candidate image pairs to be scored by a VLM:

```bash
python -m rewards.judge_images --config configs/grpo_gpt.yaml \
  --input data/images_to_score.jsonl --output outputs/judged.jsonl --repeats 3
```

Use `--judge-model` and `JUDGE_BASE_URL`/`JUDGE_API_KEY` to select an independent
evaluation judge. This utility averages repeated image scores and records each
judgment. It does not implement a human study, significance testing, or a claim
of agreement with human raters.

## Verification

```bash
python -m pytest -q
```

Tests use synthetic products and mocked external APIs. They cover two-stage
head training/checkpoint loading, pair reversal, dimension-order mismatch,
context propagation, training/evaluation aggregation equality, gates, token
masks, schema-row handling, and mean-advantage preservation. They do not establish
GPU throughput, live API compatibility for every provider, or reproduce the
paper's reported numerical results.
