"""Data models for sleep session storage."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json


@dataclass
class SleepSession:
    """Represents a sleep recording session."""

    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "recording"  # recording, completed, analyzing, analyzed
    recording_path: Optional[str] = None
    speakers_detected: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "recording_path": self.recording_path,
            "speakers_detected": self.speakers_detected,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SleepSession":
        return cls(
            session_id=data["session_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            status=data.get("status", "completed"),
            recording_path=data.get("recording_path"),
            speakers_detected=data.get("speakers_detected", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SleepSegment:
    """A segment of analyzed sleep data."""

    segment_id: str
    session_id: str
    speaker_id: Optional[str]
    start_time: datetime
    end_time: datetime
    sleep_stage: str
    confidence: float
    events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "session_id": self.session_id,
            "speaker_id": self.speaker_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "sleep_stage": self.sleep_stage,
            "confidence": self.confidence,
            "events": self.events,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SleepSegment":
        return cls(
            segment_id=data["segment_id"],
            session_id=data["session_id"],
            speaker_id=data.get("speaker_id"),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            sleep_stage=data["sleep_stage"],
            confidence=data.get("confidence", 0.0),
            events=data.get("events", []),
        )
