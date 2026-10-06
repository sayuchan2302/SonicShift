"""
scripts/list_devices.py - Audio Device Discovery Utility

Scans and enumerates all available audio input and output devices on the Windows host.
Displays device index, host API (WASAPI, MME, DirectSound, WDM-KS), channel counts,
and default sample rates.

Highlights recommended devices:
- Physical Microphones (Input)
- VB-Audio Virtual Cable ('CABLE Input' for software routing into Discord/Games)
- WASAPI host API endpoints for sub-10ms buffer latency
"""

from __future__ import annotations

import sys
from typing import List, Dict, Any, Optional


def print_banner() -> None:
    print("=" * 80)
    print("      REALTIME AI VOICE CHANGER - AUDIO DEVICE DISCOVERY UTILITY")
    print("=" * 80)
    print("This tool enumerates all Windows audio hardware and virtual endpoints.")
    print("WASAPI devices are strongly recommended for low-latency live streaming.")
    print("=" * 80)
    print()


def check_dependencies() -> bool:
    try:
        import sounddevice as sd  # noqa: F401
        return True
    except ImportError:
        print("[ERROR] 'sounddevice' library is not installed in the current Python environment.")
        print()
        print("Please activate your virtual environment and install dependencies:")
        print("    .\\venv\\Scripts\\activate")
        print("    pip install sounddevice tabulate")
        print()
        return False


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Formats rows as an ASCII table without requiring external dependencies."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Formatting string
    format_template = " | ".join([f"{{:<{w}}}" for w in col_widths])
    separator = "-+-".join(["-" * w for w in col_widths])

    lines = [
        format_template.format(*headers),
        separator
    ]
    for row in rows:
        lines.append(format_template.format(*[str(c) for c in row]))

    return "\n".join(lines)


def list_devices() -> None:
    import sounddevice as sd

    print_banner()

    try:
        devices: List[Dict[str, Any]] = sd.query_devices()
        host_apis: List[Dict[str, Any]] = sd.query_hostapis()
        default_input, default_output = sd.default.device
    except Exception as exc:
        print(f"[ERROR] Failed to query host audio subsystem: {exc}")
        sys.exit(1)

    input_rows: List[List[str]] = []
    output_rows: List[List[str]] = []

    cable_inputs: List[int] = []
    cable_outputs: List[int] = []

    for idx, dev in enumerate(devices):
        name: str = dev["name"]
        api_id: int = dev["hostapi"]
        api_name: str = host_apis[api_id]["name"] if api_id < len(host_apis) else "Unknown"
        max_in: int = dev["max_input_channels"]
        max_out: int = dev["max_output_channels"]
        default_sr: float = dev["default_samplerate"]

        # Tag special device types
        is_cable = "cable" in name.lower() or "vb-audio" in name.lower()
        is_wasapi = "wasapi" in api_name.lower()

        # Indicator tags
        tags: List[str] = []
        if idx == default_input:
            tags.append("[DEFAULT IN]")
        if idx == default_output:
            tags.append("[DEFAULT OUT]")
        if is_cable:
            tags.append("[VB-CABLE]")
        if is_wasapi:
            tags.append("[WASAPI]")

        tag_str = " ".join(tags)

        # Truncate device name if overly long
        display_name = (name[:38] + "..") if len(name) > 40 else name

        if max_in > 0:
            input_rows.append([
                str(idx),
                display_name,
                api_name,
                str(max_in),
                f"{int(default_sr)} Hz",
                tag_str
            ])
            if is_cable:
                cable_inputs.append(idx)

        if max_out > 0:
            output_rows.append([
                str(idx),
                display_name,
                api_name,
                str(max_out),
                f"{int(default_sr)} Hz",
                tag_str
            ])
            if is_cable:
                cable_outputs.append(idx)

    headers = ["Idx", "Device Name", "Host API", "Ch", "Sample Rate", "Tags"]

    print(">>> AUDIO INPUT DEVICES (MICROPHONES / CAPTURE)")
    print("-" * 80)
    print(format_table(headers, input_rows))
    print()

    print(">>> AUDIO OUTPUT DEVICES (SPEAKERS / VIRTUAL CABLES / PLAYBACK)")
    print("-" * 80)
    print(format_table(headers, output_rows))
    print()

    # Helpful recommendations & next steps
    print("=" * 80)
    print("               DEVICE SELECTION & ROUTING GUIDE")
    print("=" * 80)
    print("1. PHYSICAL MICROPHONE (INPUT):")
    print("   Find your physical microphone in the 'AUDIO INPUT' table above.")
    print("   Prefer the device using the 'Windows WASAPI' Host API for lowest latency.")
    print("   Note the device index (Idx) or unique name string.")
    print()
    print("2. VB-AUDIO VIRTUAL CABLE (OUTPUT):")
    print("   Locate 'CABLE Input (VB-Audio Virtual Cable)' under the 'AUDIO OUTPUT' table.")
    print("   The voice changer sends transformed female audio TO this output.")
    if cable_outputs:
        print(f"   --> Found VB-Cable Output indices: {cable_outputs}")
    else:
        print("   --> WARNING: No VB-Audio Virtual Cable output found!")
        print("       Please download and install VB-CABLE from https://vb-audio.com/Cable/")
        print("       and restart the script.")
    print()
    print("3. CONFIGURE .env OR config.py:")
    print("   Set the values in your `.env` file:")
    print("     INPUT_DEVICE_INDEX=<your_mic_idx>")
    print("     OUTPUT_DEVICE_INDEX=<your_cable_input_idx>")
    print("   Or by name match:")
    print("     INPUT_DEVICE_NAME=\"<part_of_mic_name>\"")
    print("     OUTPUT_DEVICE_NAME=\"CABLE Input\"")
    print()
    print("4. DISCORD / GAME VOICE SETTINGS:")
    print("   In Discord/Game Audio Settings -> Input Device -> select 'CABLE Output (VB-Audio)'.")
    print("   This completes the pipeline: Physical Mic -> AI Engine -> CABLE Input -> Game.")
    print("=" * 80)


if __name__ == "__main__":
    if check_dependencies():
        list_devices()
    else:
        sys.exit(1)
