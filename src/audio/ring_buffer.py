"""
High-performance, thread-safe Ring Buffer (Circular Buffer) for audio frames.

Designed specifically for low-latency Producer-Consumer audio streaming between
OS audio driver callbacks (PortAudio) and background AI inference workers.
"""

from __future__ import annotations

import threading
from typing import Tuple, Dict, Any, Optional
import numpy as np
import numpy.typing as npt


class RingBuffer:
    """
    Thread-safe circular audio buffer backed by a contiguous NumPy float32 array.

    Handles wraparound indexing via efficient NumPy slicing and tracks
    buffer underrun and overrun events for diagnostic telemetry.
    """

    def __init__(self, capacity_frames: int, channels: int = 1, dtype: type = np.float32) -> None:
        """
        Initializes the ring buffer.

        Args:
            capacity_frames: Maximum number of audio frames the buffer can hold.
            channels: Number of audio channels (1 for mono, 2 for stereo).
            dtype: NumPy data type, default is np.float32.
        """
        if capacity_frames <= 0:
            raise ValueError(f"capacity_frames must be positive, got {capacity_frames}")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        self.capacity: int = capacity_frames
        self.channels: int = channels
        self.dtype = dtype

        # Backing storage: shape (capacity, channels)
        self._buffer: npt.NDArray[np.float32] = np.zeros(
            (self.capacity, self.channels), dtype=self.dtype
        )

        self._read_idx: int = 0
        self._write_idx: int = 0
        self._size: int = 0  # Number of currently stored frames

        self._lock = threading.Lock()

        # Telemetry counters
        self._overrun_count: int = 0   # Producer wrote when buffer was full
        self._underrun_count: int = 0  # Consumer read when buffer was empty

    @property
    def overrun_count(self) -> int:
        """Total count of buffer overrun (overflow) incidents."""
        with self._lock:
            return self._overrun_count

    @property
    def underrun_count(self) -> int:
        """Total count of buffer underrun (starvation) incidents."""
        with self._lock:
            return self._underrun_count

    def available_read(self) -> int:
        """Returns number of available frames that can be read immediately."""
        with self._lock:
            return self._size

    def available_write(self) -> int:
        """Returns number of remaining free frames that can be written."""
        with self._lock:
            return self.capacity - self._size

    def fill_ratio(self) -> float:
        """Returns current buffer occupancy ratio in the range [0.0, 1.0]."""
        with self._lock:
            return self._size / self.capacity if self.capacity > 0 else 0.0

    def write(self, data: npt.NDArray[np.float32], drop_oldest_on_overflow: bool = True) -> int:
        """
        Writes audio frames into the ring buffer.

        Args:
            data: Audio array of shape (frames,) or (frames, channels).
            drop_oldest_on_overflow: If True, discards oldest frames when space is insufficient
                                     to prioritize real-time freshness. If False, rejects excess.

        Returns:
            Number of frames written into the buffer.
        """
        # Ensure 2D shape (frames, channels)
        if data.ndim == 1:
            data = data[:, np.newaxis]

        num_frames = data.shape[0]
        if num_frames == 0:
            return 0

        with self._lock:
            free_space = self.capacity - self._size

            if num_frames > free_space:
                self._overrun_count += 1
                if drop_oldest_on_overflow:
                    # Advance read pointer to discard oldest frames and make room
                    frames_to_drop = num_frames - free_space
                    # Clamp drop to current size
                    frames_to_drop = min(frames_to_drop, self._size)
                    self._read_idx = (self._read_idx + frames_to_drop) % self.capacity
                    self._size -= frames_to_drop
                else:
                    # Truncate incoming data to what fits
                    num_frames = free_space
                    data = data[:num_frames]
                    if num_frames == 0:
                        return 0

            # Slicing with wraparound
            end_idx = self._write_idx + num_frames
            if end_idx <= self.capacity:
                self._buffer[self._write_idx:end_idx] = data
            else:
                first_part = self.capacity - self._write_idx
                second_part = num_frames - first_part
                self._buffer[self._write_idx:] = data[:first_part]
                self._buffer[:second_part] = data[first_part:]

            self._write_idx = (self._write_idx + num_frames) % self.capacity
            self._size += num_frames

            return num_frames

    def read(
        self,
        num_frames: int,
        fill_zeros_on_underrun: bool = True
    ) -> npt.NDArray[np.float32]:
        """
        Reads audio frames from the ring buffer.

        Args:
            num_frames: Number of frames requested.
            fill_zeros_on_underrun: If True, pads missing frames with silence (0.0)
                                    when buffer has fewer frames than requested.

        Returns:
            NumPy array of shape (num_frames, channels) with read audio.
        """
        output = np.zeros((num_frames, self.channels), dtype=self.dtype)
        if num_frames <= 0:
            return output

        with self._lock:
            readable = min(num_frames, self._size)

            if readable < num_frames:
                self._underrun_count += 1

            if readable > 0:
                end_idx = self._read_idx + readable
                if end_idx <= self.capacity:
                    output[:readable] = self._buffer[self._read_idx:end_idx]
                else:
                    first_part = self.capacity - self._read_idx
                    second_part = readable - first_part
                    output[:first_part] = self._buffer[self._read_idx:]
                    output[first_part:readable] = self._buffer[:second_part]

                self._read_idx = (self._read_idx + readable) % self.capacity
                self._size -= readable

            if not fill_zeros_on_underrun and readable < num_frames:
                return output[:readable]

            return output

    def clear(self) -> None:
        """Resets the read/write pointers and empties the buffer."""
        with self._lock:
            self._buffer.fill(0)
            self._read_idx = 0
            self._write_idx = 0
            self._size = 0

    def reset_stats(self) -> None:
        """Resets overrun and underrun diagnostic counters."""
        with self._lock:
            self._overrun_count = 0
            self._underrun_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """Returns snapshot dictionary of buffer state."""
        with self._lock:
            return {
                "capacity": self.capacity,
                "current_size": self._size,
                "fill_ratio": self._size / self.capacity if self.capacity > 0 else 0.0,
                "overrun_count": self._overrun_count,
                "underrun_count": self._underrun_count,
            }
