# Training-code release design

Approved scope: structured SFT and merge, EAGLE-GRPO training, and a
CTR-supervised, 11-dimensional residual-gating weight head. No proprietary data,
model weights, benchmark reports, or historical results are included.

Preserve the existing GRPO token-mask, kernel decomposition, and presence-aware
advantage mapping. Isolate public image-editor/judge transports from training.
Use one ordered score schema and one frozen contextual scorer in RL and offline
scoring. Carry market and L1 category from CSV rows to reward requests explicitly.
Reject incompatible (including legacy 8-dimensional) checkpoints and missing
context. Preserve signed M8 coefficients; reject non-positive/near-zero sums
rather than silently changing coefficients or falling back to fixed weights.

The head trains on original/optimized image pairs, then optimized/optimized pairs,
using product-macro weighted soft-label BCE, global/residual L2, and held-out
validation for checkpoint selection. Data must provide CTR posterior preference
probabilities, or valid click/impression counts from which they are derived.
Vocabulary is fitted only on training contexts. The inference checkpoint records
criterion order, vocabulary, architecture, and recipe. No numerical results are
claimed for this released 11-dimensional adaptation without training it.

Validation: CPU synthetic two-stage training; checkpoint round-trip; identical
training/evaluation scoring; pair-order symmetry; context and dimension errors;
fidelity gates; masking and mean-advantage preservation; CLI/config imports.
GPU training and external paid API calls are not run as part of code packaging.

Implementation sequence: extract training dependencies; add M8/scoring/data
interfaces; add portable launch configs and documentation; test; inspect staged
files for data and credentials; publish to the requested repository.
