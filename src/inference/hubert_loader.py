"""
src/inference/hubert_loader.py - Standalone Official RVC HuBERT Feature Extractor

Loads official HuBERT base weights (hubert_base.pt) directly into torchaudio's
HuBERT architecture without requiring fairseq (which fails to build on Python 3.12).
Maps all 210 parameter tensors with exact shape and precision match.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional
import torch
import torchaudio


class _FairseqClassStub:
    """Stub to allow unpickling fairseq checkpoints without installing fairseq."""
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _FairseqSafePickle:
    class Unpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if "fairseq" in module:
                return _FairseqClassStub
            return super().find_class(module, name)

    load = pickle.load
    loads = pickle.loads


def _build_hubert_state_dict(fairseq_model_dict: dict) -> dict:
    """Maps fairseq HuBERT base state dict keys to torchaudio HuBERT model keys."""
    f = fairseq_model_dict
    m = {}

    # 1. Feature extractor Conv layers
    m["feature_extractor.conv_layers.0.conv.weight"] = f["feature_extractor.conv_layers.0.0.weight"].float()
    m["feature_extractor.conv_layers.0.layer_norm.weight"] = f["feature_extractor.conv_layers.0.2.weight"].float()
    m["feature_extractor.conv_layers.0.layer_norm.bias"] = f["feature_extractor.conv_layers.0.2.bias"].float()
    for i in range(1, 7):
        m[f"feature_extractor.conv_layers.{i}.conv.weight"] = f[f"feature_extractor.conv_layers.{i}.0.weight"].float()

    # 2. Feature projection
    if "layer_norm.weight" in f:
        m["encoder.feature_projection.layer_norm.weight"] = f["layer_norm.weight"].float()
        m["encoder.feature_projection.layer_norm.bias"] = f["layer_norm.bias"].float()

    m["encoder.feature_projection.projection.weight"] = f["post_extract_proj.weight"].float()
    m["encoder.feature_projection.projection.bias"] = f["post_extract_proj.bias"].float()

    # 3. Positional Convolution
    m["encoder.transformer.pos_conv_embed.conv.bias"] = f["encoder.pos_conv.0.bias"].float()
    m["encoder.transformer.pos_conv_embed.conv.parametrizations.weight.original0"] = f["encoder.pos_conv.0.weight_g"].float()
    m["encoder.transformer.pos_conv_embed.conv.parametrizations.weight.original1"] = f["encoder.pos_conv.0.weight_v"].float()

    # 4. Global Layer Norm
    if "encoder.layer_norm.weight" in f:
        m["encoder.transformer.layer_norm.weight"] = f["encoder.layer_norm.weight"].float()
        m["encoder.transformer.layer_norm.bias"] = f["encoder.layer_norm.bias"].float()

    # 5. Transformer Layers 0 to 11 (12 layers)
    for i in range(12):
        m[f"encoder.transformer.layers.{i}.attention.k_proj.weight"] = f[f"encoder.layers.{i}.self_attn.k_proj.weight"].float()
        m[f"encoder.transformer.layers.{i}.attention.k_proj.bias"] = f[f"encoder.layers.{i}.self_attn.k_proj.bias"].float()
        m[f"encoder.transformer.layers.{i}.attention.v_proj.weight"] = f[f"encoder.layers.{i}.self_attn.v_proj.weight"].float()
        m[f"encoder.transformer.layers.{i}.attention.v_proj.bias"] = f[f"encoder.layers.{i}.self_attn.v_proj.bias"].float()
        m[f"encoder.transformer.layers.{i}.attention.q_proj.weight"] = f[f"encoder.layers.{i}.self_attn.q_proj.weight"].float()
        m[f"encoder.transformer.layers.{i}.attention.q_proj.bias"] = f[f"encoder.layers.{i}.self_attn.q_proj.bias"].float()
        m[f"encoder.transformer.layers.{i}.attention.out_proj.weight"] = f[f"encoder.layers.{i}.self_attn.out_proj.weight"].float()
        m[f"encoder.transformer.layers.{i}.attention.out_proj.bias"] = f[f"encoder.layers.{i}.self_attn.out_proj.bias"].float()
        m[f"encoder.transformer.layers.{i}.layer_norm.weight"] = f[f"encoder.layers.{i}.self_attn_layer_norm.weight"].float()
        m[f"encoder.transformer.layers.{i}.layer_norm.bias"] = f[f"encoder.layers.{i}.self_attn_layer_norm.bias"].float()
        m[f"encoder.transformer.layers.{i}.feed_forward.intermediate_dense.weight"] = f[f"encoder.layers.{i}.fc1.weight"].float()
        m[f"encoder.transformer.layers.{i}.feed_forward.intermediate_dense.bias"] = f[f"encoder.layers.{i}.fc1.bias"].float()
        m[f"encoder.transformer.layers.{i}.feed_forward.output_dense.weight"] = f[f"encoder.layers.{i}.fc2.weight"].float()
        m[f"encoder.transformer.layers.{i}.feed_forward.output_dense.bias"] = f[f"encoder.layers.{i}.fc2.bias"].float()
        m[f"encoder.transformer.layers.{i}.final_layer_norm.weight"] = f[f"encoder.layers.{i}.final_layer_norm.weight"].float()
        m[f"encoder.transformer.layers.{i}.final_layer_norm.bias"] = f[f"encoder.layers.{i}.final_layer_norm.bias"].float()

    return m


def load_rvc_hubert(
    weights_path: Optional[str] = None,
    device: str = "cuda",
) -> torch.nn.Module:
    """
    Instantiates HuBERT Base and loads the exact official RVC pretrained weights.
    Falls back to torchaudio's default weights if weights_path does not exist.
    """
    model = torchaudio.pipelines.HUBERT_BASE.get_model()

    if weights_path and Path(weights_path).is_file():
        try:
            print(f"[HUBERT] Loading official RVC HuBERT weights from: {weights_path}...")
            cpt = torch.load(
                weights_path,
                map_location="cpu",
                pickle_module=_FairseqSafePickle,
                weights_only=False,
            )
            raw_sd = cpt["model"] if isinstance(cpt, dict) and "model" in cpt else cpt
            mapped_sd = _build_hubert_state_dict(raw_sd)
            model.load_state_dict(mapped_sd, strict=True)
            print("[HUBERT] Successfully loaded all 210 official RVC weights into PyTorch engine!")
        except Exception as exc:
            print(f"[WARN] Failed to load official weights ({exc}), using default torchaudio HuBERT.")
    else:
        print("[HUBERT] Using default torchaudio HuBERT Base.")

    return model.to(device).eval()
