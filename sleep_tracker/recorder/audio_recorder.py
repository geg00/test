"""Audio recorder for sleep tracking."""

import threading
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    sd = None
    sf = None

from .config import RecorderConfig


class AudioRecorder:
    """Records audio during sleep sessions with smart activation."""

    def __init__(self, config: Optional[RecorderConfig] = None):
        self.config = config or RecorderConfig()
        self._is_recording = False
        self._is_paused = False
        self._audio_queue: queue.Queue = queue.Queue()
        self._recording_thread: Optional[threading.Thread] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._current_session_id: Optional[str] = None
        self._current_file: Optional[Path] = None
        self._chunk_index = 0
        self._on_audio_callback: Optional[Callable] = None
        self._silence_start: Optional[float] = None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    def set_audio_callback(self, callback: Callable[[np.ndarray, float], None]):
        """Set callback for real-time audio processing."""
        self._on_audio_callback = callback

    def start_session(self, session_name: Optional[str] = None) -> str:
        """Start a new recording session."""
        if self._is_recording:
            raise RuntimeError("Recording already in progress")

        if sd is None:
            raise ImportError("sounddevice not installed. Run: pip install sounddevice")

        self._current_session_id = session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.config.output_dir / self._current_session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        self._chunk_index = 0
        self._is_recording = True
        self._is_paused = False
        self._silence_start = None

        self._recording_thread = threading.Thread(target=self._record_audio, daemon=True)
        self._writer_thread = threading.Thread(target=self._write_audio, daemon=True)

        self._recording_thread.start()
        self._writer_thread.start()

        return self._current_session_id

    def stop_session(self) -> Path:
        """Stop the current recording session."""
        if not self._is_recording:
            raise RuntimeError("No recording in progress")

        self._is_recording = False
        self._audio_queue.put(None)  # Signal to stop writer thread

        if self._recording_thread:
            self._recording_thread.join(timeout=5.0)
        if self._writer_thread:
            self._writer_thread.join(timeout=5.0)

        session_dir = self.config.output_dir / self._current_session_id
        return session_dir

    def pause(self):
        """Pause recording (during silence)."""
        self._is_paused = True

    def resume(self):
        """Resume recording."""
        self._is_paused = False
        self._silence_start = None

    def _record_audio(self):
        """Main recording loop."""
        chunk_samples = int(self.config.sample_rate * self.config.chunk_duration)

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")
            if not self._is_paused:
                audio_data = indata.copy()
                timestamp = time.time()
                self._audio_queue.put((audio_data, timestamp))

                if self._on_audio_callback:
                    self._on_audio_callback(audio_data, timestamp)

                # Check for silence
                rms = np.sqrt(np.mean(audio_data ** 2))
                if rms < self.config.silence_threshold:
                    if self._silence_start is None:
                        self._silence_start = timestamp
                    elif timestamp - self._silence_start > self.config.max_silence_duration:
                        self.pause()
                else:
                    self._silence_start = None
                    if self._is_paused:
                        self.resume()

        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            callback=audio_callback,
            blocksize=chunk_samples
        ):
            while self._is_recording:
                time.sleep(0.1)

    def _write_audio(self):
        """Write audio chunks to files."""
        current_chunk = []
        chunk_start_time = None
        samples_per_chunk = int(self.config.sample_rate * self.config.chunk_duration)

        while True:
            try:
                item = self._audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                # End of recording - save remaining data
                if current_chunk:
                    self._save_chunk(current_chunk, chunk_start_time)
                break

            audio_data, timestamp = item

            if chunk_start_time is None:
                chunk_start_time = timestamp

            current_chunk.append(audio_data)

            # Check if we have enough samples for a chunk
            total_samples = sum(len(chunk) for chunk in current_chunk)
            if total_samples >= samples_per_chunk:
                self._save_chunk(current_chunk, chunk_start_time)
                current_chunk = []
                chunk_start_time = None

    def _save_chunk(self, audio_chunks: list, start_time: float):
        """Save audio chunk to file."""
        if not audio_chunks:
            return

        audio_data = np.concatenate(audio_chunks, axis=0)

        session_dir = self.config.output_dir / self._current_session_id
        timestamp_str = datetime.fromtimestamp(start_time).strftime("%H%M%S")
        filename = f"chunk_{self._chunk_index:04d}_{timestamp_str}.{self.config.file_format}"
        filepath = session_dir / filename

        sf.write(filepath, audio_data, self.config.sample_rate)
        self._chunk_index += 1

        return filepath

    def get_session_files(self, session_id: Optional[str] = None) -> list[Path]:
        """Get all audio files for a session."""
        session_id = session_id or self._current_session_id
        if not session_id:
            return []

        session_dir = self.config.output_dir / session_id
        if not session_dir.exists():
            return []

        return sorted(session_dir.glob(f"*.{self.config.file_format}"))
