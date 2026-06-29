# Improvement 4: Chunked Attention Fallback for Non-Flash GPUs

## Why this change exists

When Flash Attention is unavailable, standard self-attention can materialize
very large score tensors (`[batch, heads, N, N]`) for long audio sequences.

That memory pattern is often the dominant VRAM spike in conformer inference and
is a frequent source of OOM on older or lower-memory GPUs.

## What was changed

- Added new utility module:
  - `src/SongFormer/utils/chunked_attention.py`
- Implemented:
  - `chunked_scaled_dot_product_attention(...)`
  - `_make_chunked_forward(...)`
  - `apply_chunked_attention(...)`
- Integrated fallback into both runtime paths:
  - `app.py`
  - `src/SongFormer/infer/infer.py`

Integration behavior:

1. Detect whether Flash Attention is available.
2. If available: keep flash path.
3. If not available: monkey-patch conformer attention layers to chunked exact
   attention with `chunk_size=1024`.

## Correctness model

This is an exact-attention fallback, not sparse/approximate attention:

- Each query chunk still attends to the full key/value sequence.
- Only computation order and memory layout are changed.

Expected output behavior: equivalent segmentation behavior with substantially
lower peak memory in non-flash mode.

## Operational impact

- Major reduction in peak attention memory on non-SM80 hardware.
- Makes long-window inference feasible on more GPUs.
- Potential small latency overhead due to chunked loop execution.

## Reviewer checklist

- Confirm patch applies only when flash is unavailable.
- Validate representative audio output parity with baseline behavior.
- Measure peak VRAM before/after on non-flash hardware.
- Check latency impact and document acceptable production threshold.
