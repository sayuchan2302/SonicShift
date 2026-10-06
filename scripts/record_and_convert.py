"""
scripts/record_and_convert.py - Record & Audition Voice Changer Tool

Records your voice from the microphone for a specified duration (e.g. 5 seconds),
converts it from Male to Female using neural RVC v2 on your NVIDIA RTX 3050 GPU,
saves both the original and converted audio files to recordings/, and automatically
opens the converted file so you can audition the result immediately!
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import (
    SAMPLE_RATE,
    INPUT_DEVICE_INDEX,
    INPUT_DEVICE_NAME,
    PITCH_SHIFT_SEMITONES,
    DEVICE,
    resolve_device_id,
)
from src.inference.offline_test import VoiceConverter


def record_voice(
    duration_sec: float = 5.0,
    sample_rate: int = SAMPLE_RATE,
    input_device: Optional[int] = None,
) -> np.ndarray:
    """Records audio from microphone for specified seconds with an active level bar."""
    import sounddevice as sd

    try:
        in_id = (
            input_device
            if input_device is not None
            else resolve_device_id(INPUT_DEVICE_NAME, INPUT_DEVICE_INDEX, is_input=True)
        )
    except Exception as exc:
        print(f"[ERROR] Khong tim thay microphone hop le: {exc}")
        sys.exit(1)

    dev_name = sd.query_devices(in_id)["name"]
    print("\n" + "=" * 70)
    print(f" Micro thu am    : [{in_id}] {dev_name}")
    print(f" Thoi luong      : {duration_sec:.1f} giay")
    print(f" Tan so lay mau  : {sample_rate} Hz (Mono)")
    print("=" * 70)

    # Countdown
    print("\nChuan bi noi (hay noi to, ro rang gan micro):")
    for count in [3, 2, 1]:
        print(f"  --> {count}...")
        time.sleep(1.0)
    print("  >>> BAT DAU NOI! <<<\n")

    frames_total = int(duration_sec * sample_rate)
    recorded_frames: list[np.ndarray] = []
    chunk_size = 1024
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        recorded_frames.append(indata.copy())

    with sd.InputStream(
        device=in_id,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=chunk_size,
        callback=callback,
    ):
        while time.time() - start_time < duration_sec:
            elapsed = time.time() - start_time
            pct = min(100.0, (elapsed / duration_sec) * 100)
            bar_len = 25
            filled = int(round((pct / 100.0) * bar_len))
            bar = "#" * filled + "-" * (bar_len - filled)

            # Measure latest frame RMS
            if recorded_frames:
                latest = recorded_frames[-1]
                rms = float(np.sqrt(np.mean(np.square(latest))))
                db = float(20 * np.log10(max(1e-5, rms)))
            else:
                db = -90.0

            sys.stdout.write(
                f"\r [Dang thu am]: [{bar}] {elapsed:4.1f}s / {duration_sec:4.1f}s | Muc am: {db:5.1f} dB  "
            )
            sys.stdout.flush()
            time.sleep(0.05)

    print("\n\nThu am hoan tat!")
    if not recorded_frames:
        return np.zeros((frames_total, 1), dtype=np.float32)

    audio_data = np.concatenate(recorded_frames, axis=0)[:frames_total]
    peak = float(np.max(np.abs(audio_data)))
    rms_total = float(np.sqrt(np.mean(audio_data**2)))
    db_total = float(20 * np.log10(max(1e-5, rms_total)))

    print(f"-> Thong so am goc: Peak = {peak:.4f} | RMS = {db_total:.1f} dB")
    if peak < 0.08:
        print(" [LƯU Ý]: Giọng nói từ micro khá nhỏ. Bộ xử lý Auto-Gain đã tự động khuếch đại tín hiệu để AI nhận dạng ngữ âm chính xác.")

    return audio_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Record and Audition Realtime AI Voice Changer")
    parser.add_argument("--duration", "-d", type=float, default=5.0, help="Thoi luong thu am (giay, mac dinh: 5.0)")
    parser.add_argument("--pitch", "-p", type=int, default=PITCH_SHIFT_SEMITONES, help="Do dich cao do (+12 cho Nam sang Nu)")
    parser.add_argument("--input", "-i", type=int, default=None, help="Device index cua micro (tuy chon)")
    parser.add_argument("--device", type=str, default=DEVICE, help="Thiet bi tinh toan ('cuda' hoac 'cpu')")
    parser.add_argument("--model", "-m", type=str, default=None, help="Duong dan file model RVC .pth (mac dinh: howatto_female.pth)")
    parser.add_argument("--play", action="store_true", default=True, help="Tu dong mo file nghe lai sau khi xu ly xong")

    args = parser.parse_args()

    try:
        import soundfile as sf
    except ImportError:
        print("[ERROR] Can cai thu vien soundfile: 'pip install soundfile'")
        sys.exit(1)

    # 1. Thu am giong goc
    raw_audio = record_voice(
        duration_sec=args.duration,
        sample_rate=SAMPLE_RATE,
        input_device=args.input,
    )

    # 2. Xu ly chuyen giong tren GPU bang Neural RVC
    print("\n[AI GPU] Dang chuyen doi giong tu Nam sang Nu bang Neural RVC v2 truyen HuBERT...")
    t0 = time.perf_counter()
    converter = VoiceConverter(sample_rate=SAMPLE_RATE, device=args.device)

    # Tu dong nap model howatto_female.pth neu co san
    model_to_use = args.model
    if model_to_use is None:
        default_ckpt = _project_root / "models" / "checkpoints" / "howatto_female.pth"
        if default_ckpt.is_file():
            model_to_use = str(default_ckpt)

    if model_to_use:
        converter.load_model(model_path=model_to_use, device=args.device)

    converted_audio = converter.infer(raw_audio, f0_up_key=args.pitch)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[AI GPU] Hoan thanh chuyen doi trong {elapsed_ms:.2f} ms tren [{converter.device.upper()}]!")

    # 3. Xuat file WAV
    out_dir = _project_root / "recordings"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = out_dir / f"giong_goc_{timestamp}.wav"
    converted_path = out_dir / f"giong_nu_{timestamp}.wav"

    sf.write(str(raw_path), raw_audio, SAMPLE_RATE)
    sf.write(str(converted_path), converted_audio, SAMPLE_RATE)

    print("\n" + "=" * 70)
    print(" KET QUA DA XUAT RA FILE THANH CONG:")
    print("=" * 70)
    print(f" 1. Giong goc cua ban       : {raw_path}")
    print(f" 2. Giong Nu chuyen doi AI  : {converted_path}")
    print("=" * 70)

    # 4. Tu dong mo file de nguoi dung nghe lai
    if args.play and os.name == "nt":
        print("\n--> Dang mo file giong nu de ban nghe thu...")
        try:
            os.startfile(str(converted_path))
        except Exception as play_err:
            print(f"[WARN] Khong the tu mo trinh phat nhac: {play_err}")
            print(f"Ban co the mo thu cong tai: {converted_path}")


if __name__ == "__main__":
    main()
