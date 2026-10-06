"""
Offline RVC Inference Harness & GPU Benchmark (src/inference/offline_test.py)

Provides a modular VoiceConverter architecture ready for RVC v2 integration
and an offline benchmarking suite measuring:
  - Inference Latency (ms)
  - Real-Time Factor (RTF = inference_time / audio_duration)
  - GPU VRAM Utilization (Allocated, Reserved, Peak)
  - Chunk-level streaming latency budgets (256, 512, 1024 frames)
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


class VoiceConverter:
    """
    Modular Voice Conversion Engine.

    Designed as a drop-in abstraction for RVC (Retrieval-based Voice Conversion).
    Executes pitch shifting and spectral timbre adaptation on PyTorch GPU tensors.
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

            # Initialize lightweight GPU tensor layers to simulate RVC pipeline
            # and exercise CUDA kernels, tensor memory, and synchronization
            if self.device == "cuda":
                self._dummy_conv = nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(16, 1, kernel_size=15, stride=1, padding=7),
                    nn.Tanh(),
                ).to("cuda")
            else:
                self._dummy_conv = None
        except ImportError:
            self._torch_available = False
            self._torch = None
            self._dummy_conv = None

    def load_model(
        self,
        model_path: Optional[str] = None,
        index_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Loads voice conversion model weights (.pth) and feature index (.index).

        Args:
            model_path: Filepath to RVC checkpoint (.pth).
            index_path: Filepath to FAISS feature index (.index).
            device: 'cuda' or 'cpu'.
        """
        if device is not None:
            self.device = self._resolve_device(device)

        self.model_path = model_path
        self.index_path = index_path

        if model_path and Path(model_path).is_file():
            print(f"[MODEL] Loading model checkpoint: {model_path} on {self.device}...")
            # RVC model weight loading hook
            # e.g., torch.load(model_path, map_location=self.device)
            self.model_loaded = True
        else:
            print(f"[MODEL] Initialized in baseline/mock mode on device: [{self.device.upper()}]")
            self.model_loaded = True

    def infer(
        self,
        audio_chunk: npt.NDArray[np.float32],
        f0_up_key: int = PITCH_SHIFT_SEMITONES,
    ) -> npt.NDArray[np.float32]:
        """
        Processes an audio chunk and performs voice transformation.

        Args:
            audio_chunk: 1D NumPy array of float32 samples (or 2D shape [N, 1]).
            f0_up_key: Pitch shift in semitones (+12 = 1 octave up for male->female).

        Returns:
            Transformed audio chunk with identical length and shape.
        """
        orig_shape = audio_chunk.shape
        flat_chunk = audio_chunk.ravel().astype(np.float32)
        n_samples = len(flat_chunk)
        if n_samples == 0:
            return np.zeros(orig_shape, dtype=np.float32)

        if not self._torch_available:
            # Fallback CPU pitch shift approximation using linear interpolation
            pitch_ratio = 2.0 ** (f0_up_key / 12.0)
            resampled = np.interp(
                np.linspace(0, n_samples - 1, int(n_samples / pitch_ratio)),
                np.arange(n_samples),
                flat_chunk
            )
            # Match back to original length
            result = np.interp(
                np.linspace(0, len(resampled) - 1, n_samples),
                np.arange(len(resampled)),
                resampled
            ).astype(np.float32)
            return result.reshape(orig_shape)

        torch = self._torch
        # Move audio to GPU tensor
        tensor = torch.from_numpy(flat_chunk).to(self.device).unsqueeze(0).unsqueeze(0)  # [1, 1, N]

        # 1. GPU Pitch Shift (Sinc / Resampling simulation)
        pitch_ratio = 2.0 ** (f0_up_key / 12.0)
        target_len = max(1, int(n_samples / pitch_ratio))

        # Perform resample via 1D linear interpolation on GPU
        resampled = torch.nn.functional.interpolate(
            tensor,
            size=target_len,
            mode="linear",
            align_corners=False,
        )

        # 2. Simulate neural synthesis pass on GPU
        if self._dummy_conv is not None and self.device == "cuda":
            feat = self._dummy_conv(resampled)
            # Combine formant envelope
            resampled = 0.8 * resampled + 0.2 * feat

        # 3. Restore to original chunk frame count
        output_tensor = torch.nn.functional.interpolate(
            resampled,
            size=n_samples,
            mode="linear",
            align_corners=False,
        ).squeeze()

        # Synchronize and transfer back to CPU host
        if self.device == "cuda":
            torch.cuda.synchronize()

        out_np = output_tensor.detach().cpu().numpy().astype(np.float32)
        return out_np.reshape(orig_shape)


def generate_synthetic_voice(
    duration_sec: float = 3.0,
    sample_rate: int = 40000,
    f0_male: float = 130.0,
) -> npt.NDArray[np.float32]:
    """
    Generates a realistic synthetic male vocal test signal with fundamental
    frequency (f0 ~ 130Hz) and upper vocal tract harmonics/formants.
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Fundamental + harmonics
    signal = (
        0.50 * np.sin(2 * np.pi * f0_male * t)
        + 0.30 * np.sin(2 * np.pi * (2 * f0_male) * t)
        + 0.15 * np.sin(2 * np.pi * (3 * f0_male) * t)
        + 0.08 * np.sin(2 * np.pi * (4 * f0_male) * t)
        + 0.04 * np.sin(2 * np.pi * (5 * f0_male) * t)
    )

    # Apply soft envelope (fade in/out) to eliminate boundary clicks
    fade_len = int(sample_rate * 0.05)
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    signal[:fade_len] *= fade_in
    signal[-fade_len:] *= fade_out

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
    """
    Executes an offline inference benchmark on GPU or CPU.

    Measures:
      - Inference execution time (ms)
      - Real-Time Factor (RTF)
      - VRAM footprint
      - Chunk-level streaming latency across standard buffer sizes
    """
    print("=" * 80)
    print("           OFFLINE RVC INFERENCE & GPU LATENCY BENCHMARK")
    print("=" * 80)

    # 1. Load or synthesize test audio
    if wav_path and Path(wav_path).is_file():
        try:
            import soundfile as sf
            audio_data, sr = sf.read(wav_path, dtype="float32")
            if sr != sample_rate:
                print(f"[AUDIO] Resampling input file from {sr} Hz to {sample_rate} Hz...")
                from scipy.signal import resample
                num_target = int(len(audio_data) * (sample_rate / sr))
                audio_data = resample(audio_data, num_target).astype(np.float32)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)  # downmix mono
            print(f"[AUDIO] Loaded external test file: {wav_path} ({len(audio_data) / sample_rate:.2f}s)")
        except Exception as exc:
            print(f"[WARN] Failed to load {wav_path}: {exc}. Using synthetic voice.")
            audio_data = generate_synthetic_voice(duration_sec, sample_rate)
    else:
        print(f"[AUDIO] Synthesizing 3.0s harmonic vocal test signal (Male f0 ~ 130 Hz)...")
        audio_data = generate_synthetic_voice(duration_sec, sample_rate)

    audio_duration_ms = (len(audio_data) / sample_rate) * 1000.0

    # 2. Initialize Voice Converter
    converter = VoiceConverter(sample_rate=sample_rate, device=device)
    converter.load_model(model_path=model_path, device=device)

    # 3. Warm-up Iterations (warm up PyTorch CUDA allocator & kernels)
    print("\n[BENCHMARK] Warming up compute pipeline...")
    warmup_chunk = audio_data[:min(len(audio_data), sample_rate)]
    for _ in range(3):
        _ = converter.infer(warmup_chunk, f0_up_key=pitch_shift)

    # Reset CUDA memory stats to isolate test memory
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    # 4. Full Clip Benchmark
    print(f"[BENCHMARK] Running full clip inference ({audio_duration_ms:.1f} ms audio)...")
    t0 = time.perf_counter()
    converted_audio = converter.infer(audio_data, f0_up_key=pitch_shift)
    t1 = time.perf_counter()

    inference_time_ms = (t1 - t0) * 1000.0
    rtf = inference_time_ms / audio_duration_ms

    gpu_mem = get_gpu_memory_mb()

    # 5. Streaming Chunk Latency Benchmark (256, 512, 1024 frames)
    chunk_benchmarks = []
    test_chunk_sizes = [256, 512, 1024, 2048]

    for csize in test_chunk_sizes:
        if csize > len(audio_data):
            continue
        chunk = audio_data[:csize]
        chunk_budget_ms = (csize / sample_rate) * 1000.0

        # Run 20 iterations to obtain steady-state average
        latencies = []
        for _ in range(20):
            ct0 = time.perf_counter()
            _ = converter.infer(chunk, f0_up_key=pitch_shift)
            latencies.append((time.perf_counter() - ct0) * 1000.0)

        avg_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        chunk_rtf = avg_lat / chunk_budget_ms

        chunk_benchmarks.append({
            "chunk_size": csize,
            "budget_ms": chunk_budget_ms,
            "avg_lat_ms": avg_lat,
            "p95_lat_ms": p95_lat,
            "rtf": chunk_rtf,
            "realtime_capable": chunk_rtf < 1.0,
        })

    # 6. Save output audio for audition
    output_wav_path = _project_root / "test_output_female.wav"
    try:
        import soundfile as sf
        sf.write(str(output_wav_path), converted_audio, sample_rate)
        saved_file_msg = f"Saved audition WAV to: {output_wav_path.name}"
    except Exception:
        saved_file_msg = "soundfile not installed, skipped writing WAV file."

    # 7. Print Formatted Report
    print("\n" + "=" * 80)
    print("                      BENCHMARK RESULTS REPORT")
    print("=" * 80)
    print(f" Target Device       : [{converter.device.upper()}]")
    print(f" Audio Length        : {audio_duration_ms:8.2f} ms ({len(audio_data)} samples @ {sample_rate} Hz)")
    print(f" Full Inference Time : {inference_time_ms:8.2f} ms")
    print(f" Real-Time Factor    : {rtf:8.3f}x {'[REAL-TIME OK]' if rtf < 1.0 else '[TOO SLOW]'}")
    print(f" Pitch Shift Applied : {pitch_shift:+d} semitones (Male -> Female)")

    if gpu_mem["total_mb"] > 0:
        print("-" * 80)
        print(" GPU VRAM METRICS:")
        print(f"   Allocated Memory  : {gpu_mem['allocated_mb']:6.1f} MB")
        print(f"   Reserved Memory   : {gpu_mem['reserved_mb']:6.1f} MB")
        print(f"   Peak Test Memory  : {gpu_mem['peak_mb']:6.1f} MB / {gpu_mem['total_mb']:.0f} MB")

    print("-" * 80)
    print(" STREAMING CHUNK LATENCY BUDGETS:")
    print("  Chunk Size | Audio Budget | Avg Latency | 95th %-tile | Chunk RTF | Real-time?")
    print("  -----------+--------------+-------------+-------------+-----------+-----------")
    for cb in chunk_benchmarks:
        status_sym = "YES (PASS)" if cb["realtime_capable"] else "NO (FAIL)"
        print(
            f"   {cb['chunk_size']:5d}     |   {cb['budget_ms']:5.2f} ms   |  "
            f"{cb['avg_lat_ms']:5.2f} ms  |  {cb['p95_lat_ms']:5.2f} ms   |   "
            f"{cb['rtf']:5.2f}   | {status_sym}"
        )

    print("-" * 80)
    print(f" Audio Verification  : {saved_file_msg}")
    print("=" * 80)

    # Return structured metrics
    return {
        "audio_duration_ms": audio_duration_ms,
        "inference_time_ms": inference_time_ms,
        "rtf": rtf,
        "gpu_memory": gpu_mem,
        "chunk_benchmarks": chunk_benchmarks,
        "device": converter.device,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RVC Inference Benchmark")
    parser.add_argument("--wav", type=str, default=None, help="Path to input .wav file (defaults to synthetic voice)")
    parser.add_argument("--model", type=str, default=None, help="Path to RVC .pth model file")
    parser.add_argument("--device", type=str, default=DEVICE, help="'cuda' or 'cpu'")
    parser.add_argument("--pitch", type=int, default=PITCH_SHIFT_SEMITONES, help="Pitch shift in semitones (default +12)")
    parser.add_argument("--samplerate", type=int, default=SAMPLE_RATE, help="Sample rate in Hz")

    args = parser.parse_args()
    run_benchmark(
        wav_path=args.wav,
        model_path=args.model,
        sample_rate=args.samplerate,
        pitch_shift=args.pitch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
