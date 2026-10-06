# SonicShift - Realtime AI Voice Changer (Male to Female) for Windows Gaming

High-performance, low-latency (<150ms) real-time AI voice conversion system designed for Windows gaming and streaming using NVIDIA CUDA acceleration and VB-Audio Virtual Cable.

---

## 1. System Architecture

```text
  [ Physical Microphone ]
             â”‚
             â–¼ (PortAudio InputStream Callback)
  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
  â”‚         Input Ring Buffer (Lock-Free)      â”‚
  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
             â”‚
             â–¼ (Producer-Consumer Worker Thread)
  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
  â”‚       GPU Inference Engine (PyTorch CUDA)  â”‚
  â”‚    â€¢ Pitch Shifter (+12 Semitones / M2F)  â”‚
  â”‚    â€¢ Formant / Spectral Synthesis         â”‚
  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
             â”‚
             â–¼
  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
  â”‚        Output Ring Buffer (Jitter Buffer)  â”‚
  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
             â”‚
             â–¼ (PortAudio OutputStream Callback)
  [ CABLE Input (VB-Audio Virtual Cable) ]
             â”‚
             â–¼ (Windows Virtual Audio Pipe)
  [ CABLE Output ] â”€â”€â”€â–º Discord / Steam / Valorant / OBS
```

### Latency Budget Breakdown (Target: < 150 ms)
- **Input Chunk (256 frames @ 40 kHz)**: `~6.4 ms`
- **PortAudio / WASAPI Hardware Buffer**: `~10.0 ms`
- **GPU Inference Pass (NVIDIA RTX 3050+)**: `~15.0 - 35.0 ms`
- **Output Jitter Buffer (256 frames @ 40 kHz)**: `~6.4 ms`
- **Output Driver Buffer**: `~10.0 ms`
- **Total Expected Roundtrip**: **~45 ms â€“ 70 ms** (Sub-100ms real-time gaming threshold).

---

## 2. Directory Structure

```text
realtime-voice-changer/
â”œâ”€â”€ config.py                 # Strongly-typed configuration & device resolvers
â”œâ”€â”€ .env.example              # Environment defaults template
â”œâ”€â”€ requirements.txt          # Python dependencies & CUDA PyTorch instructions
â”œâ”€â”€ README.md                 # Setup and operational guide
â”œâ”€â”€ scripts/
â”‚   â””â”€â”€ list_devices.py       # Audio device & VB-Cable discovery utility
â””â”€â”€ src/
    â”œâ”€â”€ audio/
    â”‚   â”œâ”€â”€ __init__.py       # Audio package exports
    â”‚   â”œâ”€â”€ ring_buffer.py    # Thread-safe circular audio buffer with telemetry
    â”‚   â”œâ”€â”€ stream_manager.py # Producer-Consumer multi-threaded audio pipeline
    â”‚   â””â”€â”€ passthrough.py    # Zero-AI low-latency loopback verification harness
    â””â”€â”€ inference/
        â”œâ”€â”€ __init__.py       # Inference package exports
        â””â”€â”€ offline_test.py   # RVC VoiceConverter & offline RTF/VRAM benchmark
```

---

## 3. Prerequisites

1. **Operating System**: Windows 10 or 11 (64-bit).
2. **NVIDIA GPU**: GeForce RTX series (e.g., RTX 3050 Laptop / Desktop or newer) with current NVIDIA Studio or Game Ready drivers.
3. **VB-Audio Virtual Cable**:
   - Download the free driver from [vb-audio.com/Cable/](https://vb-audio.com/Cable/).
   - Extract the ZIP, right-click `VBCABLE_Setup_x64.exe`, and select **Run as Administrator**.
   - Restart your computer if prompted.

---

## 4. Step-by-Step Installation Guide

### Step 1: Create Virtual Environment
Open PowerShell inside the project directory:

```powershell
# If script execution is restricted:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Create Python 3.10+ virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

### Step 2: Install PyTorch with CUDA Support
Standard `pip install torch` from PyPI defaults to CPU-only on Windows. Install the CUDA 12.1 build:

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

*(If you have CUDA 12.4 installed, replace `cu121` with `cu124`).*

### Step 3: Install Audio & DSP Dependencies
```powershell
pip install -r requirements.txt
```

---

## 5. Configuration & Verification Workflow

### Step 4: Discover Audio Devices
Scan your hardware and virtual audio endpoints:

```powershell
python scripts/list_devices.py
```

Look for:
- **Your Physical Microphone** in the `AUDIO INPUT DEVICES` table (prefer the `[WASAPI]` entry).
- **CABLE Input (VB-Audio Virtual Cable)** in the `AUDIO OUTPUT DEVICES` table.

Note their index numbers (`Idx`).

### Step 5: Configure `.env`
Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Open `.env` and set your indices or device names:

```ini
SAMPLE_RATE=40000
BLOCK_SIZE=256
INPUT_DEVICE_INDEX=1           # Replace with your mic index
OUTPUT_DEVICE_NAME="CABLE Input"
PITCH_SHIFT_SEMITONES=12       # +12 = 1 octave up (Male to Female)
DEVICE="cuda"
```

### Step 6: Test Low-Latency Passthrough
Verify that audio flows from your mic to the virtual cable without driver glitches, dropouts, or buffer underruns:

```powershell
python src/audio/passthrough.py
```

- Watch the real-time **VU meter** while speaking into your microphone.
- Check reported latency (typically `10ms - 20ms`).
- Press `Ctrl+C` to see the summary report. A result of `Zero buffer underruns or overflows` confirms your audio drivers are rock-solid.

### Step 7: Benchmark GPU Inference & Real-Time Factor (RTF)
Run the offline inference harness to measure PyTorch GPU speed and VRAM overhead:

```powershell
python src/inference/offline_test.py
```

The benchmark outputs:
- **Full Inference Time & RTF**: An RTF < 1.0 means the model runs faster than real-time. On an RTX GPU, baseline RTF is typically `0.05x - 0.20x` (5x to 20x faster than real-time!).
- **GPU VRAM Utilization**: Measures allocated and peak VRAM.
- **Streaming Chunk Latency Table**: Verifies that chunks of 256, 512, and 1024 frames complete well within their audio time budgets.
- Audition WAV: Generates `test_output_female.wav` so you can listen to the transformed audio.

---

## 6. Routing to Discord / In-Game Voice Chat

1. Open **Discord** -> **User Settings** -> **Voice & Video**.
2. Set **Input Device** to:
   - `CABLE Output (VB-Audio Virtual Cable)`
3. Set **Output Device** to your normal headphones/speakers.
4. Disable Discord's *Echo Cancellation* and *Noise Reduction* if you prefer pristine AI vocal fidelity.
5. In games (Valorant, CS2, Call of Duty, etc.), select `CABLE Output` as your default recording/microphone device.

---

## 7. Performance Tuning & Troubleshooting

- **Audio Crackling / Pops (Buffer Underruns)**:
  - Increase `BLOCK_SIZE` from `256` to `512` in `.env` or `config.py`.
  - Ensure sample rates match in Windows Sound Settings: Right-click Speaker icon -> **Sound Settings** -> **More Sound Settings** -> Right-click Microphone & CABLE Input -> **Properties** -> **Advanced** -> Set both to `24-bit, 48000 Hz` or `16-bit, 48000 Hz`.
- **High Latency**:
  - Make sure you select the **Windows WASAPI** device rather than MME or DirectSound.
  - Set `LATENCY_PRESET="low"`.

