"""Configuration for audio recording."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RecorderConfig:
    """Configuration settings for the audio recorder."""

    sample_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 30.0  # seconds per chunk
    output_dir: Path = Path("./recordings")
    file_format: str = "wav"

    # Voice activity detection settings
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive filtering
    min_speech_duration: float = 0.5  # minimum duration to consider as speech

    # Recording trigger settings
    silence_threshold: float = 0.01  # RMS threshold for silence detection
    max_silence_duration: float = 300.0  # max silence before pausing (5 min)

    # Storage settings
    max_storage_gb: Optional[float] = None  # None = unlimited
    compress_after_hours: float = 24.0  # compress recordings after this time

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
