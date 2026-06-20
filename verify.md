# SongFormer Improvement Verification Checklist

This document defines a lightweight but repeatable validation flow for the
optimization branches (`imp1` ... `imp5`).

Use the same audio sample(s), same GPU, and same runtime path when comparing to
`main` so numbers are meaningful.

## Scope

- Baseline: `main`
- Improvement branches: `imp1`, `imp2`, `imp3`, `imp4`, `imp5`
- Runtime paths:
  - `app.py` flow (interactive)
  - `src/SongFormer/infer/infer.py` flow (batch/CLI)

## Pre-checks (once per test environment)

1. Confirm model checkpoints are available locally.
2. Confirm GPU info and driver/CUDA environment are stable.
3. Use identical input file(s) across all branches.
4. Ensure no unrelated workload is consuming significant VRAM.

## Per-branch Smoke Validation

Run this for each branch using the same input file(s).

### 1) Run Success

- Execute inference end-to-end.
- Verify:
  - no runtime exceptions
  - output file/table is generated
  - postprocessing completes

### 2) Output Sanity vs `main`

Compare to baseline output for the same audio:

- segment count is the same (or differences are explainable)
- label sequence is the same (or minor explainable drift)
- boundary timestamps are close (practical tolerance: <= 0.1 seconds)

### 3) Peak VRAM Measurement

Use the same measurement approach in all branches:

- call `torch.cuda.reset_peak_memory_stats()` before inference
- read `torch.cuda.max_memory_allocated()` after inference

Record both:

- baseline (`main`) peak VRAM
- current branch peak VRAM

### 4) Runtime Measurement (Recommended)

- Capture wall-clock runtime per file.
- Compare baseline vs branch runtime.
- Flag any unacceptable regressions.

This is especially important for chunked-attention fallback (`imp4`) where
memory wins can introduce small latency overhead.

## Branch-specific Expected Outcomes

### `imp1` (early-exit extraction)

- Expected: lower peak VRAM in SSL extraction path.
- Expected: same labels and near-identical boundaries.

### `imp2` (flash attention gating)

- On SM80+ GPU: lower VRAM during MusicFM attention; possible speedup.
- On unsupported GPU: fallback path should still run correctly.

### `imp3` (FP16 inference)

- Expected: lower memory footprint (~weights + activations).
- Expected: outputs remain stable for practical decision thresholds.

### `imp4` (chunked attention fallback)

- Expected: lower peak memory on non-flash path.
- Expected: output parity with possible minor runtime overhead.

### `imp5` (preallocated wrapped buffers)

- Expected: reduced transient VRAM spikes during wrapped embedding assembly.
- Expected: no output-format or inference-behavior change.

## Reporting Template (Copy Into MR)

```md
## Validation
Audio sample: <file or id>
GPU: <model>
Path tested: <app.py / infer.py>

- Run success: ✅ / ❌
- Labels parity vs main: ✅ / ⚠️
- Boundary tolerance (<=0.1s): ✅ / ⚠️
- Peak VRAM (main): <x.xx GB>
- Peak VRAM (this branch): <x.xx GB>
- Runtime (main): <x.xx s>
- Runtime (this branch): <x.xx s>
- Notes: <anything unusual>
```

## Decision Rule

A branch is merge-ready when:

1. Run succeeds without errors.
2. Output quality is equivalent for practical use.
3. VRAM trend matches the branch objective.
4. Runtime impact is acceptable for production use.
