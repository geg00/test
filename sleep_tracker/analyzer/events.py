"""Sleep event definitions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class SleepEventType(Enum):
    """Types of sleep events that can be detected."""

    SNORING = "snoring"
    BREATHING_REGULAR = "breathing_regular"
    BREATHING_IRREGULAR = "breathing_irregular"
    APNEA = "apnea"  # breathing pause
    MOVEMENT = "movement"
    TALKING = "talking"
    COUGHING = "coughing"
    SILENCE = "silence"
    RESTLESSNESS = "restlessness"
    DEEP_SLEEP = "deep_sleep"
    LIGHT_SLEEP = "light_sleep"
    REM_INDICATOR = "rem_indicator"  # possible REM based on breathing patterns


class SleepStage(Enum):
    """Sleep stages estimated from audio analysis."""

    AWAKE = "awake"
    LIGHT = "light"
    DEEP = "deep"
    REM = "rem"
    UNKNOWN = "unknown"


@dataclass
class SleepEvent:
    """Represents a detected sleep event."""

    event_type: SleepEventType
    start_time: datetime
    end_time: datetime
    confidence: float  # 0.0 to 1.0
    speaker_id: Optional[str] = None  # For multi-person tracking
    intensity: float = 0.0  # Event intensity (e.g., snoring loudness)
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "event_type": self.event_type.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration": self.duration,
            "confidence": self.confidence,
            "speaker_id": self.speaker_id,
            "intensity": self.intensity,
            "metadata": self.metadata,
        }


@dataclass
class SleepSegment:
    """A segment of sleep with estimated stage."""

    stage: SleepStage
    start_time: datetime
    end_time: datetime
    confidence: float
    speaker_id: Optional[str] = None
    events: list[SleepEvent] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class SleepSummary:
    """Summary of a sleep session."""

    session_id: str
    speaker_id: Optional[str]
    start_time: datetime
    end_time: datetime
    total_duration: float  # seconds
    time_in_bed: float  # seconds
    estimated_sleep_time: float  # seconds
    sleep_efficiency: float  # percentage

    # Time in each stage (seconds)
    time_awake: float = 0.0
    time_light: float = 0.0
    time_deep: float = 0.0
    time_rem: float = 0.0

    # Event counts
    snoring_episodes: int = 0
    snoring_duration: float = 0.0
    apnea_episodes: int = 0
    movement_episodes: int = 0

    # Quality metrics
    sleep_quality_score: float = 0.0  # 0-100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "speaker_id": self.speaker_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_duration": self.total_duration,
            "time_in_bed": self.time_in_bed,
            "estimated_sleep_time": self.estimated_sleep_time,
            "sleep_efficiency": self.sleep_efficiency,
            "time_awake": self.time_awake,
            "time_light": self.time_light,
            "time_deep": self.time_deep,
            "time_rem": self.time_rem,
            "snoring_episodes": self.snoring_episodes,
            "snoring_duration": self.snoring_duration,
            "apnea_episodes": self.apnea_episodes,
            "movement_episodes": self.movement_episodes,
            "sleep_quality_score": self.sleep_quality_score,
        }
