# Isolated J-lens rehearsal environment

## Current status

This directory is a **compatibility scaffold, not an interpretation result**.
It has its own `pyproject.toml`, `.python-version`, and lock file. Its default
environment deliberately has no dependencies. Creating it does not install
PyTorch/J-lens, download Qwen weights, download a fitted lens, or contact an
API.

The source audit is pinned to Anthropic's Jacobian-lens commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e` (reviewed 2026-08-25). See
[`compatibility_snapshot.json`](compatibility_snapshot.json) for the
machine-readable record.

## What source compatibility does—and does not—establish

- The current upstream walkthrough explicitly targets `Qwen/Qwen3.5-4B`, uses
  `AutoModelForCausalLM`, and points to a pre-fitted 1,000-prompt lens. The HF
  adapter includes multimodal-wrapper layouts such as `model.language_model`.
  This is strong source-level evidence that the intended combination exists.
- It is not a local runtime check. No model, lens, Torch build, or Transformers
  build has been installed or loaded here.
- The upstream package declares Python `>=3.10`, `transformers>=5.5`, and
  otherwise unbounded core dependencies. Its current lock happens to resolve
  Torch 2.12.0 and Transformers 5.9.0; that snapshot is recorded, not treated
  as a universal Windows/CUDA recipe.
- Upstream describes this as an unmaintained reference implementation. Version
  pins and a smoke test are therefore required before doing research work.
- The paper warns that a readable/verbalizable representation is not a full
  account of all model computation. J-lens evidence must not be reported as
  if it localized every causal mechanism.

## Local hardware boundary

The observed RTX 4070 Ti exposes 12,282 MiB through `nvidia-smi`; `nvcc` is not
currently visible. A 4B model alone is roughly 8 GB in BF16, before activations,
logits, allocator overhead, and the lens. With the default final layer as the
target, the 31 fitted source-layer matrices of shape `[2560, 2560]` occupy
387.5 MiB when saved in FP16 and 775 MiB after the loader converts them to
FP32 on CPU. These are arithmetic lower bounds, not a memory benchmark.

Applying a pre-fitted lens might fit only with careful measurement. Fitting a
lens adds repeated backward passes and should be treated as out of scope for
this 12 GB machine until a smaller rehearsal and an explicit memory budget
show otherwise. CPU-offloading a 122B model is not an accepted fallback.

## Safe commands now

From the repository root:

```powershell
uv sync --project interp --frozen
uv run --project interp python interp/check_environment.py --json
```

The checker uses the standard library, does not import heavy ML packages, does
not read API keys, and performs no network call. It reports absent packages as
`null`.

## Future opt-in sequence

Proceed only after the behavioral work yields a specific hypothesis and the
dependency/model downloads are explicitly approved.

1. Choose WSL2 or native Windows and record the exact Torch/CUDA-wheel choice.
   A separately installed CUDA toolkit is not automatically required by a
   prebuilt Torch wheel; the Torch GPU smoke test is the relevant gate.
2. Pin the Jacobian-lens source commit above and create a separate, reviewed
   dependency lock. Do not add those packages to the root behavioral project.
3. Run upstream's tiny synthetic adapter test before downloading a model.
4. If approved, download only `Qwen/Qwen3.5-4B` and the matching pre-fitted
   lens, record revisions and hashes, and first test `apply` on one short,
   non-research prompt while monitoring peak VRAM.
5. Connect J-lens to a pre-registered behavioral hypothesis. Do not treat a
   visually interesting slice as a causal result.
6. Lens fitting, larger models, cloud GPUs, and 122B remain separate decisions.

## Known scientific and implementation caveats

- The official walkthrough's pre-fitted lens is trained on a generic corpus,
  not on the donation-bet task. That is useful for rehearsal but is a domain
  assumption for any substantive analysis.
- The symbolic lens revision `qwen-n1000` resolved during this audit to commit
  `16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a`; the 406,332,644-byte artifact's
  LFS SHA-256 is recorded in `compatibility_snapshot.json`. Its fit config asks
  for 1,000 prompts but records 417 fitted prompts, and it does not pin the
  source model revision. The filename is therefore not enough provenance.
- The upstream fit skips the earliest source positions in its default setup;
  upstream issue #5 documents an input-copying confound and notes that early
  positions may be unfitted. Inspect position coverage before interpreting a
  token-level slice.
- `from_hf` mutates the supplied model by freezing parameters, and its
  `compile=True` path must not be combined with `device_map="auto"`.
- Qwen 3.5 is a hybrid architecture. The official walkthrough is encouraging,
  but exact model revision, Transformers revision, dtype, tokenizer behavior,
  and layer hooks must still be captured in the eventual experiment manifest.

Primary sources:

- <https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e>
- <https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/pyproject.toml>
- <https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/walkthrough.ipynb>
- <https://github.com/anthropics/jacobian-lens/issues/5>
- <https://transformer-circuits.pub/2026/workspace/index.html>
- <https://huggingface.co/Qwen/Qwen3.5-4B>
- <https://huggingface.co/neuronpedia/jacobian-lens/tree/16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a/qwen3.5-4b/jlens/Salesforce-wikitext>
