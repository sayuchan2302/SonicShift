"""
Audio I/O Passthrough Baseline (src/audio/passthrough.py)

Direct low-latency loopback test that captures audio frames from the physical
microphone and streams them directly into the output endpoint (VB-Audio Virtual Cable)
without any AI processing.

Key capabilities:
- Uses non-blocking stream callbacks with latency='low'.
- Measures roundtrip buffer latency and monitors driver underrun/overflow status.
- Renders an ASCII audio VU meter (dBFS) to confirm microphone sensitivity and signal health.
- Diagnoses driver stability before adding GPU inference workloads.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Optional, Any
import numpy as np

# Add project root to sys.path so config can be imported directly
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import (
    SAMPLE_RATE,
    BLOCK_SIZE,
    CHANNELS,
    INPUT_DEVICE_INDEX,
    OUTPUT_DEVICE_INDEX,
    INPUT_DEVICE_NAME,
    OUTPUT_DEVICE_NAME,
    LATENCY_PRESET,
    resolve_device_id,
)


def compute_rms_db(audio_chunk: np.ndarray) -> float:
    """Computes Root-Mean-Square level in decibels relative to full scale (dBFS)."""
    rms = np.sqrt(np.mean(np.square(audio_chunk)))
    if rms < 1e-7:
        return -96.0
    return max(-96.0, float(20.0 * math.log10(rms)))


def render_vu_bar(db: float, width: int = 24) -> str:
    """Renders a text VU meter bar for dBFS values ranging from -60 to 0 dBFS."""
    clamped = max(-60.0, min(0.0, db))
    ratio = (clamped + 60.0) / 60.0
    filled = int(round(ratio * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {db:5.1f} dBFS"


def run_passthrough(
    input_device: Optional[int] = None,
    output_device: Optional[int] = None,
    sample_rate: int = SAMPLE_RATE,
    block_size: int = BLOCK_SIZE,
    channels: int = CHANNELS,
    latency_preset: str = "low",
) -> None:
    try:
        import sounddevice as sd
    except ImportError:
        print("[ERROR] 'sounddevice' library is missing. Install via 'pip install sounddevice numpy'.")
        sys.exit(1)

    # Resolve devices
    try:
        in_id = (
            input_device
            if input_device is not None
            else resolve_device_id(INPUT_DEVICE_NAME, INPUT_DEVICE_INDEX, is_input=True)
        )
        out_id = (
            output_device
            if output_device is not None
            else resolve_device_id(OUTPUT_DEVICE_NAME, OUTPUT_DEVICE_INDEX, is_input=False)
        )
    except Exception as exc:
        print(f"[CONFIGURATION ERROR] {exc}")
        print("\nTip: Run 'python scripts/list_devices.py' to list all devices.")
        sys.exit(1)

    devices = sd.query_devices()
    in_name = devices[in_id]["name"]
    out_name = devices[out_id]["name"]

    print("=" * 78)
    print("           REALTIME AUDIO I/O PASSTHROUGH BASELINE TEST")
    print("=" * 78)
    print(f" Input Device  : [{in_id}] {in_name}")
    print(f" Output Device : [{out_id}] {out_name}")
    print(f" Sample Rate   : {sample_rate} Hz")
    print(f" Block Size    : {block_size} frames (~{(block_size / sample_rate) * 1000:.2f} ms/block)")
    print(f" Channels      : {channels} (Mono)")
    print(f" Latency Mode  : '{latency_preset}'")
    print("=" * 78)
    print("Starting loopback stream. Speak into your microphone!")
    print("Press Ctrl+C to terminate test.\n")

    # Diagnostic counters
    stats = {
        "frames_streamed": 0,
        "input_overflows": 0,
        "input_underflows": 0,
        "output_overflows": 0,
        "output_underflows": 0,
        "peak_db": -96.0,
        "latest_db": -96.0,
        "callback_count": 0,
    }

    start_time = time.perf_counter()

    # Callback for duplex / pass-through
    def passthrough_callback(
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        if status:
            if status.input_overflow:
                stats["input_overflows"] += 1
            if status.input_underflow:
                stats["input_underflows"] += 1
            if status.output_overflow:
                stats["output_overflows"] += 1
            if status.output_underflow:
                stats["output_underflows"] += 1

        # Direct copy (passthrough)
        outdata[:] = indata

        # Compute level stats
        current_db = compute_rms_db(indata)
        stats["latest_db"] = current_db
        if current_db > stats["peak_db"]:
            stats["peak_db"] = current_db

        stats["frames_streamed"] += frames
        stats["callback_count"] += 1

    try:
        # Open duplex Stream with low-latency settings
        with sd.Stream(
            device=(in_id, out_id),
            samplerate=sample_rate,
            blocksize=block_size,
            dtype="float32",
            channels=channels,
            latency=latency_preset,
            callback=passthrough_callback,
        ) as stream:
            # Estimate driver roundtrip latency
            reported_latency_ms = stream.latency * 1000.0 if isinstance(stream.latency, (int, float)) else sum(stream.latency) * 500.0
            print(f"[DRIVER] Hardware stream established. PortAudio reported roundtrip: {reported_latency_ms:.2f} ms\n")

            while True:
                time.sleep(0.08)  # ~12 FPS refresh rate for console VU
                db = stats["latest_db"]
                vu_str = render_vu_bar(db)
                elapsed = time.perf_counter() - start_time
                ovf = stats["input_overflows"] + stats["output_overflows"]
                unf = stats["input_underflows"] + stats["output_underflows"]

                sys.stdout.write(
                    f"\r[VU] {vu_str} | Latency: {reported_latency_ms:4.1f}ms | "
                    f"Drops (Ovf:{ovf}/Unf:{unf}) | Time: {elapsed:5.1f}s   "
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n" + "=" * 78)
        print("               PASSTHROUGH TEST SUMMARY REPORT")
        print("=" * 78)
        total_time = time.perf_counter() - start_time
        print(f" Total Duration   : {total_time:.2f} seconds")
        print(f" Total Callbacks  : {stats['callback_count']} iterations")
        print(f" Peak Level       : {stats['peak_db']:.1f} dBFS")
        print(f" Input Overflows  : {stats['input_overflows']}")
        print(f" Output Underflows: {stats['output_underflows']}")
        total_glitches = (
            stats["input_overflows"]
            + stats["input_underflows"]
            + stats["output_overflows"]
            + stats["output_underflows"]
        )
        if total_glitches == 0:
            print(" Stability Result : [EXCELLENT] Zero buffer underruns or overflows recorded.")
        else:
            print(f" Stability Result : [WARNING] {total_glitches} buffer dropouts recorded.")
            print("                   Consider increasing BLOCK_SIZE to 512 in config.py.")
        print("=" * 78)

    except Exception as exc:
        print(f"\n[STREAM ERROR] {exc}")
        print("\nIf PortAudio threw an error, check if both devices share the same sample rate")
        print("or try running the multi-threaded decoupled pipeline in src/audio/stream_manager.py.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-Latency Audio Passthrough Baseline Test")
    parser.add_argument("--input-device", type=int, default=None, help="Input device index (microphone)")
    parser.add_argument("--output-device", type=int, default=None, help="Output device index (VB-Cable / speakers)")
    parser.add_argument("--samplerate", type=int, default=SAMPLE_RATE, help="Sampling frequency in Hz")
    parser.add_argument("--blocksize", type=int, default=BLOCK_SIZE, help="Frames per callback block")
    parser.add_argument("--channels", type=int, default=CHANNELS, help="Number of channels (1=mono)")
    parser.add_argument("--latency", type=str, default=str(LATENCY_PRESET), help="Latency preset ('low', 'high')")

    args = parser.parse_args()
    run_passthrough(
        input_device=args.input_device,
        output_device=args.output_device,
        sample_rate=args.samplerate,
        block_size=args.blocksize,
        channels=args.channels,
        latency_preset=args.latency,
    )


if __name__ == "__main__":
    main()
