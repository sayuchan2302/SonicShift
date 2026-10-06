"""
Multi-threaded Producer-Consumer Audio Stream Manager.

Decouples low-latency OS audio driver callbacks (PortAudio) from heavy AI inference
workloads using dual Ring Buffers (Input Ring Buffer & Output Ring Buffer) and a
dedicated processing worker thread.

Supports Dual-Output Monitoring:
Routes converted female voice simultaneously to:
  1. CABLE Input (VB-Audio Virtual Cable) -> feeds into Discord/Games.
  2. Headphones (Optional Monitor Stream) -> allows the user to hear their transformed voice live.

Architecture:
  [Physical Mic]
        │
  (Audio Callback - Producer)
        ▼
  [Input Ring Buffer]
        │
  (Worker Thread - Consumer & Producer) -> [AI Voice Converter]
        ├──► [Output Ring Buffer]  ──► (OutputStream Callback) ──► [CABLE Input (Discord/Game)]
        └──► [Monitor Ring Buffer] ──► (MonitorStream Callback) ──► [Headphones (Live Ears)]
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Union, Dict, Any
import numpy as np
import numpy.typing as npt

from .ring_buffer import RingBuffer


@dataclass
class AudioPipelineMetrics:
    """Real-time diagnostic metrics for audio pipeline health and latency."""
    input_latency_ms: float = 0.0
    output_latency_ms: float = 0.0
    total_est_latency_ms: float = 0.0
    input_buffer_fill_pct: float = 0.0
    output_buffer_fill_pct: float = 0.0
    input_overflows: int = 0
    input_underflows: int = 0
    output_overflows: int = 0
    output_underflows: int = 0
    last_process_time_ms: float = 0.0
    avg_process_time_ms: float = 0.0
    total_frames_streamed: int = 0


class AudioStreamManager:
    """
    Manages multi-stream audio capture, processing, and optional live headphone monitoring.
    """

    def __init__(
        self,
        sample_rate: int = 40000,
        block_size: int = 256,
        channels: int = 1,
        input_device: Optional[Union[int, str]] = None,
        output_device: Optional[Union[int, str]] = None,
        monitor_device: Optional[Union[int, str]] = None,
        monitor_volume: float = 1.0,
        latency_preset: Union[str, float] = "low",
        buffer_blocks: int = 64,
        process_callback: Optional[Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]] = None,
    ) -> None:
        """
        Initializes the audio streaming manager.

        Args:
            sample_rate: Audio sampling frequency in Hz (default 40000).
            block_size: Number of frames per audio callback chunk (default 256).
            channels: Number of audio channels (1 for mono).
            input_device: Sounddevice index or name for capture (Mic).
            output_device: Sounddevice index or name for playback (VB-Cable).
            monitor_device: Optional Sounddevice index for headphones monitor.
            monitor_volume: Volume multiplier for headphone monitoring [0.0 - 1.0].
            latency_preset: 'low', 'high', or duration in seconds.
            buffer_blocks: Ring buffer capacity in units of block_size.
            process_callback: Audio processing callback: f(chunk) -> transformed_chunk.
        """
        self.sample_rate: int = sample_rate
        self.block_size: int = block_size
        self.channels: int = channels
        self.input_device = input_device
        self.output_device = output_device
        self.monitor_device = monitor_device
        self.monitor_volume = float(np.clip(monitor_volume, 0.0, 1.0))
        self.latency_preset = latency_preset
        self.process_callback = process_callback or (lambda x: x)

        capacity_frames = self.block_size * buffer_blocks
        self.input_buffer = RingBuffer(capacity_frames=capacity_frames, channels=self.channels)
        self.output_buffer = RingBuffer(capacity_frames=capacity_frames, channels=self.channels)

        # Optional separate monitor buffer for headphones
        self.monitor_buffer: Optional[RingBuffer] = (
            RingBuffer(capacity_frames=capacity_frames, channels=self.channels)
            if self.monitor_device is not None
            else None
        )

        self._is_running = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._in_stream: Optional[Any] = None
        self._out_stream: Optional[Any] = None
        self._mon_stream: Optional[Any] = None

        # Telemetry metrics
        self.metrics = AudioPipelineMetrics()
        self._process_times: list[float] = []

    def _input_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """PortAudio input callback: immediately pushes captured mic audio to input buffer."""
        if status:
            if status.input_overflow:
                self.metrics.input_overflows += 1
            if status.input_underflow:
                self.metrics.input_underflows += 1

        self.input_buffer.write(indata.astype(np.float32), drop_oldest_on_overflow=True)
        self.metrics.total_frames_streamed += frames

    def _output_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """PortAudio output callback: reads processed frames from output buffer into DAC/VB-Cable."""
        if status:
            if status.output_overflow:
                self.metrics.output_overflows += 1
            if status.output_underflow:
                self.metrics.output_underflows += 1

        # Fetch processed audio; zero-pad if underrun
        chunk = self.output_buffer.read(frames, fill_zeros_on_underrun=True)
        outdata[:] = chunk

    def _monitor_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """PortAudio monitor callback: outputs transformed voice into headphones for self-listening."""
        if self.monitor_buffer is not None:
            chunk = self.monitor_buffer.read(frames, fill_zeros_on_underrun=True)
            outdata[:] = chunk * self.monitor_volume
        else:
            outdata.fill(0)

    def _worker_loop(self) -> None:
        """
        Background worker thread: Consumes audio chunks from input buffer,
        runs transformation/inference callback, and writes to output & monitor buffers.
        """
        block_duration_sec = self.block_size / self.sample_rate

        while self._is_running.is_set():
            # Check if at least one block is available to process
            available = self.input_buffer.available_read()
            if available < self.block_size:
                time.sleep(block_duration_sec * 0.25)
                continue

            # Read one block
            chunk = self.input_buffer.read(self.block_size, fill_zeros_on_underrun=False)
            if chunk.shape[0] < self.block_size:
                continue

            # Measure processing latency
            t0 = time.perf_counter()
            try:
                processed_chunk = self.process_callback(chunk)
            except Exception:
                processed_chunk = chunk

            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.metrics.last_process_time_ms = dt_ms

            # Maintain running average of worker processing time
            self._process_times.append(dt_ms)
            if len(self._process_times) > 50:
                self._process_times.pop(0)
            self.metrics.avg_process_time_ms = sum(self._process_times) / len(self._process_times)

            proc_f32 = processed_chunk.astype(np.float32)

            # Write transformed audio into output ring buffer (VB-Cable)
            self.output_buffer.write(proc_f32, drop_oldest_on_overflow=True)

            # Write into headphone monitor buffer if enabled
            if self.monitor_buffer is not None:
                self.monitor_buffer.write(proc_f32, drop_oldest_on_overflow=True)

    def start(self) -> None:
        """Starts input/output streams and the audio worker thread."""
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("sounddevice is required to start audio streams.") from exc

        if self._is_running.is_set():
            return

        self.input_buffer.clear()
        self.output_buffer.clear()
        if self.monitor_buffer is not None:
            self.monitor_buffer.clear()

        self._is_running.set()

        # Start worker thread
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="AudioWorkerThread",
            daemon=True,
        )
        self._worker_thread.start()

        # Open Input stream (Mic)
        self._in_stream = sd.InputStream(
            device=self.input_device,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=self.channels,
            dtype="float32",
            latency=self.latency_preset,
            callback=self._input_callback,
        )

        # Open Primary Output stream (VB-Cable)
        self._out_stream = sd.OutputStream(
            device=self.output_device,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=self.channels,
            dtype="float32",
            latency=self.latency_preset,
            callback=self._output_callback,
        )

        self._in_stream.start()
        self._out_stream.start()

        # Open Monitor Output stream (Headphones) if specified
        if self.monitor_device is not None:
            try:
                self._mon_stream = sd.OutputStream(
                    device=self.monitor_device,
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    channels=self.channels,
                    dtype="float32",
                    latency=self.latency_preset,
                    callback=self._monitor_callback,
                )
                self._mon_stream.start()
            except Exception as mon_err:
                print(f"[WARN] Failed to initialize monitor stream on device {self.monitor_device}: {mon_err}")
                self._mon_stream = None

        # Store measured driver latencies
        in_lat = getattr(self._in_stream, "latency", 0.0)
        out_lat = getattr(self._out_stream, "latency", 0.0)
        self.metrics.input_latency_ms = in_lat * 1000.0
        self.metrics.output_latency_ms = out_lat * 1000.0
        self.metrics.total_est_latency_ms = (
            self.metrics.input_latency_ms
            + self.metrics.output_latency_ms
            + (self.block_size / self.sample_rate * 1000.0)
        )

    def stop(self) -> None:
        """Gracefully halts audio streams and stops the worker thread."""
        self._is_running.clear()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

        if self._in_stream:
            try:
                self._in_stream.stop()
                self._in_stream.close()
            except Exception:
                pass
            self._in_stream = None

        if self._out_stream:
            try:
                self._out_stream.stop()
                self._out_stream.close()
            except Exception:
                pass
            self._out_stream = None

        if self._mon_stream:
            try:
                self._mon_stream.stop()
                self._mon_stream.close()
            except Exception:
                pass
            self._mon_stream = None

    def is_active(self) -> bool:
        """Returns True if streams and worker thread are active."""
        return (
            self._is_running.is_set()
            and (self._in_stream is not None and self._in_stream.active)
            and (self._out_stream is not None and self._out_stream.active)
        )

    def get_metrics(self) -> AudioPipelineMetrics:
        """Updates and returns pipeline diagnostics."""
        self.metrics.input_buffer_fill_pct = self.input_buffer.fill_ratio() * 100.0
        self.metrics.output_buffer_fill_pct = self.output_buffer.fill_ratio() * 100.0
        self.metrics.input_overflows = self.input_buffer.overrun_count
        self.metrics.input_underflows = self.input_buffer.underrun_count
        self.metrics.output_overflows = self.output_buffer.overrun_count
        self.metrics.output_underflows = self.output_buffer.underrun_count
        return self.metrics

    def __enter__(self) -> AudioStreamManager:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
