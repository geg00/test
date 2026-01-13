"""Speaker profile for multi-person tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json
from pathlib import Path
import numpy as np


@dataclass
class SpeakerProfile:
    """Profile for a detected speaker/sleeper."""

    speaker_id: str
    name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    # Voice embedding (averaged from samples)
    embedding: Optional[np.ndarray] = None
    embedding_samples: int = 0

    # Acoustic characteristics
    avg_fundamental_freq: Optional[float] = None  # F0 in Hz
    freq_std: Optional[float] = None
    avg_snoring_freq: Optional[float] = None

    # Spatial characteristics (if stereo)
    typical_position: Optional[str] = None  # "left", "right", "center"

    def update_embedding(self, new_embedding: np.ndarray, weight: float = 0.1):
        """Update embedding with new sample using exponential moving average."""
        if self.embedding is None:
            self.embedding = new_embedding
            self.embedding_samples = 1
        else:
            # Weighted average favoring existing profile
            alpha = weight
            self.embedding = (1 - alpha) * self.embedding + alpha * new_embedding
            self.embedding_samples += 1

    def similarity(self, embedding: np.ndarray) -> float:
        """Calculate cosine similarity with another embedding."""
        if self.embedding is None:
            return 0.0

        # Cosine similarity
        dot = np.dot(self.embedding, embedding)
        norm_self = np.linalg.norm(self.embedding)
        norm_other = np.linalg.norm(embedding)

        if norm_self == 0 or norm_other == 0:
            return 0.0

        return float(dot / (norm_self * norm_other))

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "speaker_id": self.speaker_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "embedding_samples": self.embedding_samples,
            "avg_fundamental_freq": self.avg_fundamental_freq,
            "freq_std": self.freq_std,
            "avg_snoring_freq": self.avg_snoring_freq,
            "typical_position": self.typical_position,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpeakerProfile":
        """Create from dictionary."""
        embedding = None
        if data.get("embedding") is not None:
            embedding = np.array(data["embedding"])

        return cls(
            speaker_id=data["speaker_id"],
            name=data.get("name"),
            created_at=datetime.fromisoformat(data["created_at"]),
            embedding=embedding,
            embedding_samples=data.get("embedding_samples", 0),
            avg_fundamental_freq=data.get("avg_fundamental_freq"),
            freq_std=data.get("freq_std"),
            avg_snoring_freq=data.get("avg_snoring_freq"),
            typical_position=data.get("typical_position"),
        )

    def save(self, path: Path):
        """Save profile to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SpeakerProfile":
        """Load profile from file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
