# Improvement 3: FP16 Inference for SSL + SongFormer Models

## Why this change exists

The inference path previously used default FP32 tensors for:

- MuQ
- MusicFM
- SongFormer inference model
- Loaded waveform tensors

FP32 is memory-expensive for this workload and contributes directly to high GPU
memory pressure during embedding extraction and model inference.

## What was changed

- Introduced `INFERENCE_DTYPE = torch.float16` in:
  - `app.py`
  - `src/SongFormer/infer/infer.py`
- Updated model placement to use explicit dtype:
  - `.to(device=device, dtype=INFERENCE_DTYPE).eval()`
- Updated waveform tensor creation to use FP16 at source:
  - `torch.tensor(wav, dtype=INFERENCE_DTYPE).to(device)`
- Updated SongFormer inference model placement to match FP16 inference dtype.

## Design intent

The dtype constant centralizes precision control so operations teams can switch
between FP16/BF16/FP32 with a single configuration edit in each runtime path.

This change is intended to reduce memory footprint without changing inference
pipeline semantics or output schema.

## Correctness notes

FP16 is a lossy numeric representation relative to FP32. In practice for this
task, expected impact is minimal because final decisions are based on robust
peak/argmax logic rather than fragile exact-value comparisons.

Expected behavior:

- Same output structure and labels in normal operation.
- Potential tiny logit-level numeric drift that should not materially affect
  boundary and class decisions for typical confidence margins.

## Operational impact

- Roughly ~50% reduction in tensor memory for many activations/weights.
- Lower chance of OOM on memory-constrained GPUs.
- Potential throughput gains from tensor core acceleration on supported hardware.

## Reviewer checklist

- Run a representative audio batch before/after FP16 switch.
- Compare output labels and boundary timestamps against FP32 baseline.
- Validate no runtime dtype mismatch errors on target GPU fleet.
- Confirm peak memory reduction with runtime profiling tools.
