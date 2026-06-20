# Improvement 2: Enable Flash Attention for MusicFM (with Safe Fallback)

## Why this change exists

MusicFM was instantiated with `is_flash=False`, forcing the standard attention
path that materializes large attention score tensors in VRAM.

For long sequence inference, this can become a major memory bottleneck and
contribute to OOM risk.

## What was changed

- Added runtime capability checks:
  - `app.py`: `_flash_attention_available()`
  - `src/SongFormer/infer/infer.py`: `_flash_attention_available(device_id)`
- Switched MusicFM construction from fixed `is_flash=False` to:
  - `is_flash=use_flash` where `use_flash` is derived from GPU capability.
- Added MusicFM module-path bootstrap in both entrypoints so internal imports
  resolve correctly when flash mode is enabled.

## Compatibility design

Flash attention is enabled only when all required runtime conditions are met:

- CUDA is available
- Device compute capability is SM80 or newer (`major >= 8`)

Otherwise, the code falls back to the original non-flash path automatically.

This keeps behavior safe on mixed hardware fleets and developer machines.

## Correctness notes

This optimization is intended to preserve model function while improving memory
behavior and performance characteristics.

Expected output behavior: same segmentation quality and structure output format.
Minor floating-point differences from kernel implementation details are possible
but should not materially affect boundaries/labels in normal use.

## Operational impact

- Lower peak VRAM on supported GPUs during MusicFM attention blocks.
- Potential throughput/latency improvement from fused attention kernels.
- No required CLI/API contract change.

## Reviewer checklist

- Confirm fallback behavior on non-SM80 GPUs.
- Confirm flash path activates on SM80+ GPUs.
- Run representative inference and verify output parity expectations.
- Check memory profile for reduced peak usage on flash-capable hardware.
