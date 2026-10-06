"""
scripts/record_and_convert.py - Record & Audition Voice Changer Tool

Records your voice from the microphone for a specified duration (e.g. 5 seconds)
or until you press Enter, converts it from Male to Female using PyTorch CUDA on
your NVIDIA RTX 3050, saves both the original and converted audio, and automatically
opens the converted file so you can listen to it immediately!
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np

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
        print(f"[ERROR] Failed to resolve microphone: {exc}")
        sys.exit(1)

    dev_name = sd.query_devices(in_id)["name"]
    print("\n" + "=" * 70)
    print(f" Micro thu âm : [{in_id}] {dev_name}")
    print(f" Thời lượng   : {duration_sec:.1f} giây")
    print(f" Tần số mẫu   : {sample_rate} Hz (Mono)")
    print("=" * 70)

    # Countdown
    print("\nChuẩn bị nói trong:")
    for count in [3, 2, 1]:
        print(f"  --> {count}...")
        time.sleep(1.0)
    print("  >>> BẮT ĐẦU NÓI! <<<\n")

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
            remaining = max(0.0, duration_sec - elapsed)
            pct = min(100.0, (elapsed / duration_sec) * 100)
            bar_len = 25
            filled = int(round((pct / 100.0) * bar_len))
            bar = "█" * filled + "░" * (bar_len - filled)

            # Measure latest frame RMS
            if recorded_frames:
                latest = recorded_frames[-1]
                rms = np.sqrt(np.mean(np.square(latest)))
                db = 20 * np.log10(max(1e-5, rms))
            else:
                db = -90.0

            sys.stdout.write(
                f"\r Đang thu âm: [{bar}] {elapsed:4.1f}s / {duration_sec:4.1f}s | Âm lượng: {db:5.1f} dB  "
            )
            sys.stdout.flush()
            time.sleep(0.05)

    print("\n\nThu âm hoàn tất!")
    if not recorded_frames:
        return np.zeros((frames_total, 1), dtype=np.float32)

    audio_data = np.concatenate(recorded_frames, axis=0)
    return audio_data[:frames_total]


def main() -> None:
    parser = argparse.ArgumentParser(description="Record and Audition Realtime AI Voice Changer")
    parser.add_argument("--duration", "-d", type=float, default=5.0, help="Thời lượng thu âm (giây, mặc định: 5.0)")
    parser.add_argument("--pitch", "-p", type=int, default=PITCH_SHIFT_SEMITONES, help="Độ dịch cao độ (+12 cho Nam sang Nữ)")
    parser.add_argument("--input", "-i", type=int, default=None, help="Device index của micro (tùy chọn)")
    parser.add_argument("--device", type=str, default=DEVICE, help="Thiết bị tính toán ('cuda' hoặc 'cpu')")
    parser.add_argument("--play", action="store_true", default=True, help="Tự động mở file nghe lại sau khi xử lý xong")

    args = parser.parse_args()

    try:
        import soundfile as sf
    except ImportError:
        print("[ERROR] Cần cài thư viện soundfile: 'pip install soundfile'")
        sys.exit(1)

    # 1. Thu âm giọng gốc
    raw_audio = record_voice(
        duration_sec=args.duration,
        sample_rate=SAMPLE_RATE,
        input_device=args.input,
    )

    # 2. Xử lý chuyển giọng trên GPU RTX 3050
    print("\n[AI GPU] Đang chuyển đổi giọng từ Nam sang Nữ bằng PyTorch CUDA...")
    t0 = time.perf_counter()
    converter = VoiceConverter(sample_rate=SAMPLE_RATE, device=args.device)
    converted_audio = converter.infer(raw_audio, f0_up_key=args.pitch)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[AI GPU] Hoàn thành chuyển đổi trong {elapsed_ms:.2f} ms trên [{converter.device.upper()}]!")

    # 3. Xuất file WAV
    out_dir = _project_root / "recordings"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = out_dir / f"giong_goc_{timestamp}.wav"
    converted_path = out_dir / f"giong_nu_{timestamp}.wav"

    sf.write(str(raw_path), raw_audio, SAMPLE_RATE)
    sf.write(str(converted_path), converted_audio, SAMPLE_RATE)

    print("\n" + "=" * 70)
    print(" KẾT QUẢ ĐÃ XUẤT RA FILE THÀNH CÔNG:")
    print("=" * 70)
    print(f" 1. Giọng gốc của bạn      : {raw_path}")
    print(f" 2. Giọng Nữ chuyển đổi    : {converted_path}")
    print("=" * 70)

    # 4. Tự động mở file để người dùng nghe lại
    if args.play and os.name == "nt":
        print("\n--> Đang mở file giọng nữ để bạn nghe lại...")
        try:
            os.startfile(str(converted_path))
        except Exception as play_err:
            print(f"[WARN] Không thể tự mở trình phát nhạc: {play_err}")
            print(f"Bạn có thể mở thủ công tại: {converted_path}")


if __name__ == "__main__":
    main()
