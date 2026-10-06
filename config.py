"""
Configuration module for the Realtime AI Voice Changer (SonicShift).

Provides strongly typed configuration settings with automatic environment
variable loading from a local .env file, fallback defaults, and smart device resolution
helpers for low-latency audio streaming with sounddevice.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Tuple, Any

# Simple built-in .env parser if python-dotenv is not yet installed in venv
def _load_dotenv_fallback(env_path: Path) -> None:
    if not env_path.is_file():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass


# Attempt to load .env
_env_file = Path(__file__).resolve().parent / ".env"
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=_env_file)
except ImportError:
    _load_dotenv_fallback(_env_file)


def _get_env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


def _get_env_optional_int(key: str) -> Optional[int]:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return None
    try:
        return int(val.strip())
    except ValueError:
        return None


@dataclass(frozen=True)
class VoiceChangerConfig:
    """Strongly-typed runtime configuration parameters."""

    # Audio Signal Processing
    sample_rate: int = field(default_factory=lambda: _get_env_int("SAMPLE_RATE", 48000))
    block_size: int = field(default_factory=lambda: _get_env_int("BLOCK_SIZE", 256))
    channels: int = field(default_factory=lambda: _get_env_int("CHANNELS", 1))

    # Hardware & Audio Routing
    input_device_name: Optional[str] = field(
        default_factory=lambda: os.getenv("INPUT_DEVICE_NAME", "Microphone").strip() or None
    )
    input_device_index: Optional[int] = field(
        default_factory=lambda: _get_env_optional_int("INPUT_DEVICE_INDEX")
    )
    output_device_name: Optional[str] = field(
        default_factory=lambda: os.getenv("OUTPUT_DEVICE_NAME", "CABLE Input").strip() or None
    )
    output_device_index: Optional[int] = field(
        default_factory=lambda: _get_env_optional_int("OUTPUT_DEVICE_INDEX")
    )

    # Conversion Parameters
    pitch_shift_semitones: int = field(
        default_factory=lambda: _get_env_int("PITCH_SHIFT_SEMITONES", 12)
    )
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "cuda").strip().lower())
    latency_preset: Union[str, float] = field(
        default_factory=lambda: os.getenv("LATENCY_PRESET", "low").strip()
    )

    # Ring Buffer / Streaming Queues
    queue_max_blocks: int = field(
        default_factory=lambda: _get_env_int("QUEUE_MAX_BLOCKS", 64)
    )

    # Optional model paths
    model_path: Optional[str] = field(
        default_factory=lambda: os.getenv("MODEL_PATH", "").strip() or None
    )
    index_path: Optional[str] = field(
        default_factory=lambda: os.getenv("INDEX_PATH", "").strip() or None
    )

    @property
    def block_duration_ms(self) -> float:
        """Returns the nominal duration of a single audio chunk in milliseconds."""
        return (self.block_size / self.sample_rate) * 1000.0

    @property
    def ring_buffer_capacity_frames(self) -> int:
        """Total frame capacity of ring buffers based on block count."""
        return self.block_size * self.queue_max_blocks

    @property
    def ring_buffer_capacity_ms(self) -> float:
        """Total buffer time cushion in milliseconds."""
        return (self.ring_buffer_capacity_frames / self.sample_rate) * 1000.0


# Default global instance
config = VoiceChangerConfig()

# Global module aliases for direct imports:
SAMPLE_RATE: int = config.sample_rate
BLOCK_SIZE: int = config.block_size
CHANNELS: int = config.channels
INPUT_DEVICE_NAME: Optional[str] = config.input_device_name
INPUT_DEVICE_INDEX: Optional[int] = config.input_device_index
OUTPUT_DEVICE_NAME: Optional[str] = config.output_device_name
OUTPUT_DEVICE_INDEX: Optional[int] = config.output_device_index
PITCH_SHIFT_SEMITONES: int = config.pitch_shift_semitones
DEVICE: str = config.device
LATENCY_PRESET: Union[str, float] = config.latency_preset
QUEUE_MAX_BLOCKS: int = config.queue_max_blocks
MODEL_PATH: Optional[str] = config.model_path
INDEX_PATH: Optional[str] = config.index_path


def resolve_device_id(
    name_query: Optional[str] = None,
    explicit_index: Optional[int] = None,
    is_input: bool = True,
) -> int:
    """
    Resolves the exact SoundDevice device ID based on explicit index or substring name match.
    Prefers Windows WASAPI devices for lowest latency on Windows.
    Smart scoring avoids picking empty 'External Microphone' jacks over built-in Microphones.

    Args:
        name_query: Substring to match in device name (e.g. 'CABLE Input', 'Microphone')
        explicit_index: Direct integer device index if already known.
        is_input: True for microphone/recording device, False for speaker/virtual cable input.

    Returns:
        Integer device index recognized by sounddevice.

    Raises:
        RuntimeError: If sounddevice cannot query devices or device is not found.
    """
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice library is not installed. Run 'pip install sounddevice' first."
        ) from exc

    devices = sd.query_devices()
    host_apis = sd.query_hostapis()

    # 1. If explicit index is provided, validate it
    if explicit_index is not None:
        if 0 <= explicit_index < len(devices):
            dev = devices[explicit_index]
            max_ch = dev["max_input_channels"] if is_input else dev["max_output_channels"]
            if max_ch > 0:
                return explicit_index
            raise ValueError(
                f"Device index {explicit_index} ('{dev['name']}') does not support "
                f"{'input' if is_input else 'output'} channels (max_channels={max_ch})."
            )
        raise ValueError(
            f"Device index {explicit_index} is out of range [0, {len(devices) - 1}]."
        )

    # 2. If name query is provided, search matching devices with smart scoring
    if name_query:
        query_lower = name_query.lower()
        candidates = []
        for idx, dev in enumerate(devices):
            max_ch = dev["max_input_channels"] if is_input else dev["max_output_channels"]
            if max_ch > 0 and query_lower in dev["name"].lower():
                api_name = host_apis[dev["hostapi"]]["name"]
                is_wasapi = "wasapi" in api_name.lower()
                name_l = dev["name"].lower()

                # Prioritize:
                # 1. WASAPI over MME / DirectSound
                # 2. If query does not mention "external", penalize "external microphone" (often an empty 3.5mm jack)
                # 3. Names starting with the query (e.g. "Microphone (Realtek)" over "External Microphone")
                wasapi_score = 1 if is_wasapi else 0
                is_external_jack = 1 if ("external" in name_l and "external" not in query_lower) else 0
                starts_with = 1 if name_l.startswith(query_lower) else 0

                score = (wasapi_score, -is_external_jack, starts_with)
                candidates.append((idx, dev, score))

        if candidates:
            # Sort candidates by highest score
            candidates.sort(key=lambda c: c[2], reverse=True)
            return candidates[0][0]

        raise RuntimeError(
            f"No matching {'input' if is_input else 'output'} device found with name substring '{name_query}'. "
            f"Run 'python scripts/list_devices.py' to list available devices."
        )

    # 3. Fallback to default device
    default_tuple = sd.default.device
    def_idx = default_tuple[0] if is_input else default_tuple[1]
    if def_idx is not None and def_idx >= 0:
        return def_idx

    raise RuntimeError(
        f"No default {'input' if is_input else 'output'} device detected on system."
    )
