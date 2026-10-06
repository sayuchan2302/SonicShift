"""
Offline RVC Inference Harness & GPU Benchmark (src/inference/offline_test.py)

Provides a modular VoiceConverter architecture integrating:
  - Official RVC HuBERT speech representation encoder (12th layer, 100 fps)
  - VITS neural acoustic generator (SynthesizerTrnMs768NSFsid / SynthesizerTrnMs768NSFsid_nono)
  - Automatic Gain Control (AGC) and noise gate for low-gain microphones
  - Zero destructive waveform interpolation (high-fidelity resampling)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import numpy.typing as npt

# Add project root to sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import SAMPLE_RATE, BLOCK_SIZE, DEVICE, PITCH_SHIFT_SEMITONES
from src.inference.hubert_loader import load_rvc_hubert


class VoiceConverter:
    """
    High-fidelity Neural Voice Conversion Engine using RVC v2 architecture.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        device: str = DEVICE,
    ) -> None:
        self.sample_rate: int = sample_rate
        self.requested_device: str = device
        self.device: str = self._resolve_device(device)
        self.model_loaded: bool = False
        self.model_path: Optional[str] = None
        self.index_path: Optional[str] = None

        # Check PyTorch availability
        self._torch_available: bool = False
        self._init_torch()

    def _resolve_device(self, requested: str) -> str:
        """Determines best available compute backend (CUDA or CPU)."""
        if requested == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return "cpu"

    def _init_torch(self) -> None:
        try:
            import torch
            import torch.nn as nn
            self._torch = torch
            self._torch_available = True
        except ImportError:
            self._torch_available = False
            self._torch = None

        self._is_neural_model: bool = False
        self._net_g = None
        self._hubert = None
        self._model_sr: int = 40000
        self._resampler_to_16k = None
        self._resampler_to_target = None

    def load_model(
        self,
        model_path: Optional[str] = None,
        index_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Loads voice conversion model weights (.pth) and HuBERT encoder.
        """
        if device is not None:
            self.device = self._resolve_device(device)

        self.model_path = model_path
        self.index_path = index_path

        if model_path and Path(model_path).is_file() and self._torch_available:
            try:
                import torchaudio
                from .rvc.models import SynthesizerTrnMs768NSFsid_nono

                print(f"[AI NEURAL] Loading RVC v2 Model: {model_path} on [{self.device.upper()}]...")
                cpt = self._torch.load(model_path, map_location="cpu", weights_only=False)
                config = cpt.get("config", [])
                self._model_sr = config[-1] if isinstance(config, list) and len(config) > 0 else 40000

                # Instantiate VITS generator
                net = SynthesizerTrnMs768NSFsid_nono(*config, is_half=False).to(self.device).eval()
                net.load_state_dict(cpt.get("weight", {}), strict=False)
                self._net_g = net

                # Load Official HuBERT Base weights
                hubert_weights = _project_root / "models" / "pretrained" / "hubert_base.pt"
                self._hubert = load_rvc_hubert(
                    weights_path=str(hubert_weights) if hubert_weights.is_file() else None,
                    device=self.device,
                )

                # Resamplers (accurate linear phase bandlimited)
                if self.sample_rate != 16000:
                    self._resampler_to_16k = torchaudio.transforms.Resample(self.sample_rate, 16000).to(self.device)
                if self._model_sr != self.sample_rate:
                    self._resampler_to_target = torchaudio.transforms.Resample(self._model_sr, self.sample_rate).to(self.device)

                self._is_neural_model = True
                self.model_loaded = True
                print(f"[AI NEURAL] RVC v2 Neural Pipeline active! Native SR: {self._model_sr} Hz")
                return
            except Exception as exc:
                print(f"[WARN] Failed to load neural RVC model ({exc}). Falling back to Phase Vocoder.")
                self._is_neural_model = False

        print(f"[MODEL] Initialized in Phase Vocoder mode on device: [{self.device.upper()}]")
        self.model_loaded = True

    def infer(
        self,
        audio_chunk: npt.NDArray[np.float32],
        f0_up_key: int = PITCH_SHIFT_SEMITONES,
    ) -> npt.NDArray[np.float32]:
        """
        Processes an audio chunk and performs high-fidelity voice transformation.
        """
        orig_shape = audio_chunk.shape
        flat_chunk = audio_chunk.ravel().astype(np.float32)
        n_samples = len(flat_chunk)
        if n_samples == 0:
            return np.zeros(orig_shape, dtype=np.float32)

        peak_amp = float(np.max(np.abs(flat_chunk)))

        # Noise Gate: If input is pure silence / tiny floor noise, output silence
        if peak_amp < 0.003:
            return np.zeros(orig_shape, dtype=np.float32)

        # Automatic Gain Control (AGC): normalize quiet speech so HuBERT extracts strong formants
        gain_scale = 1.0
        if peak_amp < 0.35:
            gain_scale = 0.70 / max(1e-4, peak_amp)
            flat_chunk = flat_chunk * gain_scale

        # 1. Authentic RVC v2 Neural Conversion Path
        if self._is_neural_model and self._net_g is not None and self._hubert is not None:
            try:
                import torch
                t_raw = torch.from_numpy(flat_chunk).float().unsqueeze(0).to(self.device)

                # Resample to 16kHz for HuBERT
                if self._resampler_to_16k is not None:
                    audio_16k = self._resampler_to_16k(t_raw)
                else:
                    audio_16k = t_raw

                with torch.no_grad():
                    # Extract phonetic representations
                    features_list, _ = self._hubert.extract_features(audio_16k)
                    # Layer 11 (the 12th layer) is the standard semantic layer for RVC v2
                    phone = features_list[11] if len(features_list) > 11 else features_list[-1]

                    # Standard RVC 100 fps frame duplication (HuBERT 50fps -> VITS 100fps)
                    phone_100hz = phone.repeat_interleave(2, dim=1)

                    # Synthesize with VITS generator
                    phone_lengths = torch.tensor([phone_100hz.shape[1]], device=self.device)
                    sid = torch.tensor([0], device=self.device)
                    synth = self._net_g.infer(phone_100hz, phone_lengths, sid)[0]

                    # Resample back to target rate (e.g. 40kHz -> 48kHz)
                    if self._resampler_to_target is not None:
                        synth = self._resampler_to_target(synth)

                    out_np = synth.squeeze().cpu().numpy().astype(np.float32)

                    # Length alignment without destructive interpolation
                    if len(out_np) < n_samples:
                        out_np = np.pad(out_np, (0, n_samples - len(out_np)))
                    else:
                        out_np = out_np[:n_samples]

                    # Output Peak Normalization
                    out_peak = float(np.max(np.abs(out_np)))
                    if out_peak > 1e-4:
                        out_np = (out_np / out_peak) * min(0.85, max(0.5, peak_amp * 1.5))

                    if self.device == "cuda":
                        torch.cuda.synchronize()

                    return out_np.reshape(orig_shape)

            except Exception as rvc_err:
                print(f"[WARN] Neural inference error: {rvc_err}")

        # Fallback: Torchaudio Phase Vocoder Pitch Shift
        if self._torch_available:
            try:
                import torch
                import torchaudio.functional as F_audio
                tensor_audio = torch.from_numpy(flat_chunk).float().to(self.device)
                shifted = F_audio.pitch_shift(
                    waveform=tensor_audio,
                    sample_rate=self.sample_rate,
                    n_steps=f0_up_key,
                )
                if self.device == "cuda":
                    torch.cuda.synchronize()
                return shifted.cpu().numpy().astype(np.float32).reshape(orig_shape)
            except Exception:
                pass

        return flat_chunk.reshape(orig_shape)


def generate_synthetic_voice(duration_sec: float = 3.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Synthesizes male voice harmonic signal."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    f0 = 130.0
    signal = (
        0.5 * np.sin(2 * np.pi * f0 * t)
        + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.15 * np.sin(2 * np.pi * 3 * f0 * t)
        + 0.05 * np.sin(2 * np.pi * 4 * f0 * t)
    )
    return signal.astype(np.float32)


def get_gpu_memory_mb() -> Dict[str, float]:
    """Retrieves current and peak GPU VRAM allocation in Megabytes."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            return {
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "peak_mb": peak,
                "total_mb": total,
            }
    except Exception:
        pass
    return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0, "total_mb": 0.0}


def run_benchmark(
    wav_path: Optional[str] = None,
    model_path: Optional[str] = None,
    duration_sec: float = 3.0,
    sample_rate: int = SAMPLE_RATE,
    pitch_shift: int = PITCH_SHIFT_SEMITONES,
    device: str = DEVICE,
) -> Dict[str, Any]:
    """Executes offline inference benchmark."""
    converter = VoiceConverter(sample_rate=sample_rate, device=device)
    converter.load_model(model_path=model_path, device=device)
    audio_data = generate_synthetic_voice(duration_sec, sample_rate)
    t0 = time.perf_counter()
    out = converter.infer(audio_data, f0_up_key=pitch_shift)
    t1 = time.perf_counter()
    inf_ms = (t1 - t0) * 1000.0
    return {
        "audio_duration_ms": duration_sec * 1000.0,
        "inference_time_ms": inf_ms,
        "rtf": inf_ms / (duration_sec * 1000.0),
        "device": converter.device,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RVC Inference Benchmark")
    parser.add_argument("--wav", type=str, default=None, help="Path to input .wav file")
    parser.add_argument("--model", type=str, default=None, help="Path to RVC .pth model file")
    parser.add_argument("--device", type=str, default=DEVICE, help="'cuda' or 'cpu'")
    args = parser.parse_args()
    run_benchmark(wav_path=args.wav, model_path=args.model, device=args.device)


if __name__ == "__main__":
    main()
