# Improvement 1: Early-Exit SSL Embedding Extraction in `app.py`

## Why this change exists

The original inference path in `app.py` called MuQ and MusicFM with
`output_hidden_states=True` and then selected `hidden_states[10]`.

That approach causes two avoidable costs:

1. It runs all conformer encoder layers, even though SongFormer only uses the
   representation at layer index `10` (output after layer 9).
2. It materializes the full hidden-state tuple (13 tensors), where only one
   tensor is needed.

For long windows, this creates large transient VRAM pressure and increases the
risk of out-of-memory failures.

## What was changed

- Added `conformer_early_exit()` in `app.py`.
  - Temporarily truncates conformer layers to `[:target_layer]`.
  - Temporarily replaces final `layer_norm` with `Identity`.
  - Runs forward with `output_hidden_states=False`.
  - Restores original conformer state in a `finally` block.
- Added `extract_muq_embedding()` and `extract_musicfm_embedding()` helpers
  that replicate each model's preprocessing and then call early-exit.
- Replaced all naive hidden-state extraction calls in `process_audio()`:
  - 420s embedding extraction path.
  - 30s wrapped embedding extraction path.

## Correctness notes

This is an equivalence optimization, not an approximation:

- `hidden_states[10]` from the original code is the activation after layer 9,
  before later layers are executed.
- The new path computes exactly that activation directly.

Expected output behavior: identical segmentation labels and boundary structure,
with lower peak memory usage.

## Operational impact

- Lower peak VRAM during MuQ and MusicFM feature extraction.
- Reduced wasted compute from unused late conformer layers.
- No intended behavior change in postprocessing or output format.

## Reviewer checklist

- Confirm no API or CLI changes.
- Run one representative audio file before/after and compare:
  - predicted segment labels
  - boundary timestamps (allowing only normal floating-point noise)
- Observe reduced peak GPU memory during embedding extraction.
