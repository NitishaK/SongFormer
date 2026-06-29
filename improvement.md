# Improvement 5: Pre-Allocate 30s Embedding Buffers (Avoid Concat Peak)

## Why this change exists

The previous implementation accumulated per-chunk embeddings in Python lists and
then called `torch.concatenate(...)` to build the final wrapped embedding.

That pattern creates a transient memory peak:

- all chunk tensors still exist
- plus a new contiguous destination tensor allocated by `concatenate`

During heavy inference windows, this temporary duplication can be large enough
to trigger OOM even when steady-state memory is otherwise acceptable.

## What was changed

In both runtime paths:

- `app.py`
- `src/SongFormer/infer/infer.py`

the 30s embedding construction logic was refactored to:

1. Collect chunk tensors and sum `total_30s_frames`.
2. Pre-allocate destination tensors once with `torch.empty(...)`.
3. Copy each chunk into destination slices using explicit offsets.
4. Free chunk lists immediately after copy.

This keeps peak memory lower than append-then-concatenate under equivalent load.

## Correctness notes

This is a memory-allocation strategy change, not a model/math change:

- Values written into destination slices are the same values previously passed to
  `torch.concatenate`.
- Tensor content and downstream inference semantics are intended to remain
  unchanged.

Expected output behavior: identical inference outputs, lower transient memory.

## Operational impact

- Reduced short-lived VRAM spikes during wrapped embedding assembly.
- Better stability margin on GPUs with constrained memory.
- No API/format change for output artifacts.

## Reviewer checklist

- Confirm wrapped embedding shapes match previous implementation.
- Run representative inference and compare output parity.
- Profile peak VRAM around the 30s wrapping section before/after change.
