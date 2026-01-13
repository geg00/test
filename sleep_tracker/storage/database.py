"""SQLite database for sleep session storage."""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import SleepSession, SleepSegment


class SleepDatabase:
    """SQLite database for storing sleep data."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("./sleep_data.db")
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT DEFAULT 'recording',
                    recording_path TEXT,
                    speakers_detected TEXT,
                    metadata TEXT
                )
            """)

            # Segments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS segments (
                    segment_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    speaker_id TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    sleep_stage TEXT,
                    confidence REAL,
                    events TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    speaker_id TEXT,
                    event_type TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    confidence REAL,
                    intensity REAL,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Summaries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    speaker_id TEXT,
                    total_duration REAL,
                    sleep_efficiency REAL,
                    time_awake REAL,
                    time_light REAL,
                    time_deep REAL,
                    time_rem REAL,
                    snoring_episodes INTEGER,
                    snoring_duration REAL,
                    apnea_episodes INTEGER,
                    sleep_quality_score REAL,
                    created_at TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            conn.commit()

    def save_session(self, session: SleepSession):
        """Save or update a session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO sessions
                (session_id, start_time, end_time, status, recording_path, speakers_detected, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.start_time.isoformat(),
                session.end_time.isoformat() if session.end_time else None,
                session.status,
                session.recording_path,
                json.dumps(session.speakers_detected),
                json.dumps(session.metadata),
            ))
            conn.commit()

    def get_session(self, session_id: str) -> Optional[SleepSession]:
        """Get a session by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()

            if row:
                return SleepSession(
                    session_id=row[0],
                    start_time=datetime.fromisoformat(row[1]),
                    end_time=datetime.fromisoformat(row[2]) if row[2] else None,
                    status=row[3],
                    recording_path=row[4],
                    speakers_detected=json.loads(row[5]) if row[5] else [],
                    metadata=json.loads(row[6]) if row[6] else {},
                )
            return None

    def list_sessions(self, limit: int = 10) -> list[SleepSession]:
        """List recent sessions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM sessions ORDER BY start_time DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()

            sessions = []
            for row in rows:
                sessions.append(SleepSession(
                    session_id=row[0],
                    start_time=datetime.fromisoformat(row[1]),
                    end_time=datetime.fromisoformat(row[2]) if row[2] else None,
                    status=row[3],
                    recording_path=row[4],
                    speakers_detected=json.loads(row[5]) if row[5] else [],
                    metadata=json.loads(row[6]) if row[6] else {},
                ))
            return sessions

    def save_event(
        self,
        session_id: str,
        event_type: str,
        start_time: datetime,
        end_time: datetime,
        speaker_id: Optional[str] = None,
        confidence: float = 0.0,
        intensity: float = 0.0,
        metadata: Optional[dict] = None,
    ):
        """Save a sleep event."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO events
                (session_id, speaker_id, event_type, start_time, end_time, confidence, intensity, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                speaker_id,
                event_type,
                start_time.isoformat(),
                end_time.isoformat(),
                confidence,
                intensity,
                json.dumps(metadata) if metadata else None,
            ))
            conn.commit()

    def get_events(
        self,
        session_id: str,
        speaker_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> list[dict]:
        """Get events for a session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM events WHERE session_id = ?"
            params = [session_id]

            if speaker_id:
                query += " AND speaker_id = ?"
                params.append(speaker_id)

            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)

            query += " ORDER BY start_time"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                {
                    "event_id": row[0],
                    "session_id": row[1],
                    "speaker_id": row[2],
                    "event_type": row[3],
                    "start_time": row[4],
                    "end_time": row[5],
                    "confidence": row[6],
                    "intensity": row[7],
                    "metadata": json.loads(row[8]) if row[8] else {},
                }
                for row in rows
            ]

    def save_summary(self, summary: dict):
        """Save a sleep summary."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO summaries
                (session_id, speaker_id, total_duration, sleep_efficiency,
                 time_awake, time_light, time_deep, time_rem,
                 snoring_episodes, snoring_duration, apnea_episodes,
                 sleep_quality_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                summary.get("session_id"),
                summary.get("speaker_id"),
                summary.get("total_duration"),
                summary.get("sleep_efficiency"),
                summary.get("time_awake"),
                summary.get("time_light"),
                summary.get("time_deep"),
                summary.get("time_rem"),
                summary.get("snoring_episodes"),
                summary.get("snoring_duration"),
                summary.get("apnea_episodes"),
                summary.get("sleep_quality_score"),
                datetime.now().isoformat(),
            ))
            conn.commit()

    def get_summaries(
        self,
        session_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """Get sleep summaries."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM summaries WHERE 1=1"
            params = []

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)

            if speaker_id:
                query += " AND speaker_id = ?"
                params.append(speaker_id)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                {
                    "summary_id": row[0],
                    "session_id": row[1],
                    "speaker_id": row[2],
                    "total_duration": row[3],
                    "sleep_efficiency": row[4],
                    "time_awake": row[5],
                    "time_light": row[6],
                    "time_deep": row[7],
                    "time_rem": row[8],
                    "snoring_episodes": row[9],
                    "snoring_duration": row[10],
                    "apnea_episodes": row[11],
                    "sleep_quality_score": row[12],
                    "created_at": row[13],
                }
                for row in rows
            ]
