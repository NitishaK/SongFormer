"""Early-exit SSL embedding extraction for MuQ and MusicFM."""

import torch
import torch.nn

# hidden_states[TARGET_LAYER] = output after conformer layer (TARGET_LAYER - 1).
TARGET_LAYER = 10


def conformer_early_exit(conformer, x, target_layer=TARGET_LAYER):
    """Return hidden_states[target_layer] without full hidden-state materialization."""
    original_layers = conformer.layers
    original_layer_norm = conformer.layer_norm
    conformer.layers = original_layers[:target_layer]
    conformer.layer_norm = torch.nn.Identity()
    try:
        out = conformer(x, output_hidden_states=False)
        return out.last_hidden_state
    finally:
        conformer.layers = original_layers
        conformer.layer_norm = original_layer_norm


def extract_muq_embedding(muq, audio_seg, target_layer=TARGET_LAYER):
    x = muq.model.preprocessing(audio_seg, features=["melspec_2048"])
    x = muq.model.normalize(x)
    x = muq.model.conv(x["melspec_2048"])
    return conformer_early_exit(muq.model.conformer, x, target_layer)


def extract_musicfm_embedding(musicfm, audio_seg, target_layer=TARGET_LAYER):
    x = musicfm.preprocessing(audio_seg, features=["melspec_2048"])
    x = musicfm.normalize(x)
    x = musicfm.conv(x["melspec_2048"])
    return conformer_early_exit(musicfm.conformer, x, target_layer)
