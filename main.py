"""
Realtime AI Voice Changer - Main Live Streaming Application

Hooks microphone input into the high-performance RingBuffer pipeline,
runs voice transformation (Male to Female pitch shifting & formant adaptation),
and outputs to Headphones (for monitoring) or VB-Audio Virtual Cable (for games/Discord).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional, Any
import numpy as np

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent
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
    PITCH_SHIFT_SEMITONES,
    DEVICE,
    LATENCY_PRESET,
    QUEUE_MAX_BLOCKS,
    resolve_device_id,
)
from src.audio.stream_manager import AudioStreamManager
from src.inference.offline_test import VoiceConverter


def compute_rms_db(audio_chunk: np.ndarray) -> float:
    """Computes Root-Mean-Square level in dBFS."""
    rms = np.sqrt(np.mean(np.square(audio_chunk)))
    if rms < 1e-7:
        return -96.0
    return max(-96.0, float(20.0 * math.log10(rms)))


def render_vu_bar(db: float, width: int = 20) -> str:
    """Renders ASCII audio VU meter bar."""
    clamped = max(-60.0, min(0.0, db))
    ratio = (clamped + 60.0) / 60.0
    filled = int(round(ratio * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {db:5.1f} dB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime AI Voice Changer (Male to Female)")
    parser.add_argument("--input", type=int, default=None, help="Input audio device index (Microphone)")
    parser.add_argument("--output", type=int, default=None, help="Output audio device index (Headphones or VB-Cable)")
    parser.add_argument("--pitch", type=int, default=PITCH_SHIFT_SEMITONES, help="Pitch shift in semitones (default +12 for M2F)")
    parser.add_argument("--device", type=str, default=DEVICE, help="Inference hardware backend ('cuda' or 'cpu')")
    parser.add_argument("--model", type=str, default=None, help="Path to RVC .pth model checkpoint")
    parser.add_argument("--index", type=str, default=None, help="Path to FAISS .index file")

    args = parser.parse_args()

    # 1. Resolve Audio Endpoints
    try:
        import sounddevice as sd
    except ImportError:
        print("[ERROR] 'sounddevice' is not installed. Run 'pip install sounddevice numpy'.")
        sys.exit(1)

    try:
        in_id = (
            args.input
            if args.input is not None
            else resolve_device_id(INPUT_DEVICE_NAME, INPUT_DEVICE_INDEX, is_input=True)
        )
        out_id = (
            args.output
            if args.output is not None
            else resolve_device_id(OUTPUT_DEVICE_NAME, OUTPUT_DEVICE_INDEX, is_input=False)
        )
    except Exception as exc:
        print(f"[CONFIGURATION ERROR] {exc}")
        print("Tip: Run 'python scripts/list_devices.py' to inspect your device indices.")
        sys.exit(1)

    devices = sd.query_devices()
    in_name = devices[in_id]["name"]
    out_name = devices[out_id]["name"]

    print("=" * 80)
    print("      REALTIME AI VOICE CHANGER (MALE TO FEMALE) - LIVE STREAMING")
    print("=" * 80)
    print(f" Input Device (Mic)   : [{in_id}] {in_name}")
    print(f" Output Device (Play) : [{out_id}] {out_name}")
    print(f" Sampling Rate        : {SAMPLE_RATE} Hz")
    print(f" Block Size (Chunk)   : {BLOCK_SIZE} frames (~{(BLOCK_SIZE / SAMPLE_RATE) * 1000:.2f} ms)")
    print(f" Pitch Transposition  : {args.pitch:+d} semitones (Male -> Female)")
    print(f" Hardware Backend     : [{args.device.upper()}]")
    print("=" * 80)

    # 2. Initialize Voice Converter Engine
    print("\n[AI ENGINE] Initializing Voice Converter...")
    converter = VoiceConverter(sample_rate=SAMPLE_RATE, device=args.device)
    converter.load_model(model_path=args.model, index_path=args.index, device=args.device)
    print(f"[AI ENGINE] Engine active on backend: [{converter.device.upper()}]")

    # State tracking
    latest_rms = {"db": -96.0}

    # 3. Audio transformation callback
    def voice_processor(chunk: np.ndarray) -> np.ndarray:
        latest_rms["db"] = compute_rms_db(chunk)
        # Apply AI / pitch transformation
        return converter.infer(chunk, f0_up_key=args.pitch)

    # 4. Initialize Stream Manager
    print("[STREAM] Starting low-latency RingBuffer audio pipeline...")
    stream_manager = AudioStreamManager(
        sample_rate=SAMPLE_RATE,
        block_size=BLOCK_SIZE,
        channels=CHANNELS,
        input_device=in_id,
        output_device=out_id,
        latency_preset=LATENCY_PRESET,
        buffer_blocks=QUEUE_MAX_BLOCKS,
        process_callback=voice_processor,
    )

    try:
        stream_manager.start()
        print("\n" + "=" * 80)
        print(" [LIVE] STREAMING ACTIVE! Speak into your microphone to hear transformed voice.")
        print(" Press Ctrl+C in terminal to stop.")
        print("=" * 80 + "\n")

        start_time = time.perf_counter()

        while True:
            time.sleep(0.1)  # 10 FPS console refresh
            m = stream_manager.get_metrics()
            vu_str = render_vu_bar(latest_rms["db"])
            elapsed = time.perf_counter() - start_time
            proc_lat = m.last_process_time_ms
            drops = m.input_overflows + m.output_underflows

            sys.stdout.write(
                f"\r[MIC] {vu_str} | Proc: {proc_lat:4.1f}ms | "
                f"Buff In:{m.input_buffer_fill_pct:3.0f}% Out:{m.output_buffer_fill_pct:3.0f}% | "
                f"Drops: {drops} | Time: {elapsed:5.1f}s  "
            )
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\nStopping audio stream...")
    finally:
        stream_manager.stop()
        print("[STREAM] Audio pipeline stopped cleanly.")


if __name__ == "__main__":
    main()
