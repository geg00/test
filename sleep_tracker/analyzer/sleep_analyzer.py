"""Sleep pattern analyzer using audio analysis."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

try:
    from scipy import signal
    from scipy.ndimage import uniform_filter1d
except ImportError:
    signal = None

from .events import SleepEvent, SleepEventType, SleepStage, SleepSegment, SleepSummary


class SleepAnalyzer:
    """Analyzes audio recordings for sleep patterns."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: float = 0.025,  # 25ms frames
        hop_length: float = 0.010,  # 10ms hop
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self._frame_samples = int(sample_rate * frame_length)
        self._hop_samples = int(sample_rate * hop_length)

        # Detection thresholds (can be calibrated)
        self.snoring_freq_range = (30, 300)  # Hz
        self.breathing_freq_range = (0.1, 0.5)  # Hz (breathing rate)
        self.snoring_energy_threshold = 0.3
        self.movement_threshold = 0.5
        self.silence_threshold = 0.01

    def analyze_file(
        self,
        audio_path: Path,
        start_time: Optional[datetime] = None,
        speaker_id: Optional[str] = None,
    ) -> list[SleepEvent]:
        """Analyze a single audio file for sleep events."""
        if librosa is None:
            raise ImportError("librosa not installed. Run: pip install librosa")

        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        if start_time is None:
            start_time = datetime.now()

        events = []

        # Detect various sleep events
        events.extend(self._detect_snoring(audio, start_time, speaker_id))
        events.extend(self._detect_breathing_patterns(audio, start_time, speaker_id))
        events.extend(self._detect_movement(audio, start_time, speaker_id))
        events.extend(self._detect_talking(audio, start_time, speaker_id))
        events.extend(self._detect_apnea(audio, start_time, speaker_id))

        # Sort by start time
        events.sort(key=lambda e: e.start_time)

        return events

    def analyze_session(
        self,
        audio_files: list[Path],
        session_id: str,
        speaker_id: Optional[str] = None,
    ) -> tuple[list[SleepEvent], list[SleepSegment], SleepSummary]:
        """Analyze a complete sleep session."""
        all_events = []

        for audio_path in sorted(audio_files):
            # Extract timestamp from filename if possible
            start_time = self._extract_timestamp_from_filename(audio_path)
            events = self.analyze_file(audio_path, start_time, speaker_id)
            all_events.extend(events)

        # Estimate sleep stages based on events
        segments = self._estimate_sleep_stages(all_events)

        # Generate summary
        summary = self._generate_summary(session_id, speaker_id, all_events, segments)

        return all_events, segments, summary

    def _detect_snoring(
        self,
        audio: np.ndarray,
        start_time: datetime,
        speaker_id: Optional[str],
    ) -> list[SleepEvent]:
        """Detect snoring episodes in audio."""
        events = []

        # Compute spectrogram
        hop = self._hop_samples
        n_fft = self._frame_samples * 2

        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=n_fft)

        # Focus on snoring frequency range
        snore_mask = (freqs >= self.snoring_freq_range[0]) & (freqs <= self.snoring_freq_range[1])
        snore_energy = magnitude[snore_mask].mean(axis=0)

        # Normalize
        if snore_energy.max() > 0:
            snore_energy = snore_energy / snore_energy.max()

        # Find snoring segments
        is_snoring = snore_energy > self.snoring_energy_threshold

        # Group consecutive frames
        snoring_regions = self._find_contiguous_regions(is_snoring, min_duration=1.0)

        for start_frame, end_frame in snoring_regions:
            start_sec = start_frame * hop / self.sample_rate
            end_sec = end_frame * hop / self.sample_rate
            intensity = float(snore_energy[start_frame:end_frame].mean())

            event = SleepEvent(
                event_type=SleepEventType.SNORING,
                start_time=start_time + timedelta(seconds=start_sec),
                end_time=start_time + timedelta(seconds=end_sec),
                confidence=min(0.9, intensity + 0.3),
                speaker_id=speaker_id,
                intensity=intensity,
                metadata={"avg_energy": intensity},
            )
            events.append(event)

        return events

    def _detect_breathing_patterns(
        self,
        audio: np.ndarray,
        start_time: datetime,
        speaker_id: Optional[str],
    ) -> list[SleepEvent]:
        """Detect breathing patterns in audio."""
        events = []

        # Compute envelope of the signal
        envelope = np.abs(audio)

        # Smooth with moving average (breathing is slow ~0.2-0.3 Hz)
        window_size = int(self.sample_rate * 2)  # 2 second window
        if len(envelope) > window_size:
            smoothed = uniform_filter1d(envelope, size=window_size)

            # Find breathing rate using autocorrelation
            autocorr = np.correlate(smoothed, smoothed, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # Find peaks in autocorrelation (breathing period)
            min_period = int(self.sample_rate / 0.5)  # max 0.5 Hz (30 breaths/min)
            max_period = int(self.sample_rate / 0.1)  # min 0.1 Hz (6 breaths/min)

            if len(autocorr) > max_period:
                segment = autocorr[min_period:max_period]
                if len(segment) > 0:
                    peak_idx = np.argmax(segment) + min_period
                    breathing_rate = self.sample_rate / peak_idx  # Hz

                    # Determine if breathing is regular or irregular
                    regularity = autocorr[peak_idx] / autocorr[0] if autocorr[0] > 0 else 0

                    duration = len(audio) / self.sample_rate
                    event_type = (
                        SleepEventType.BREATHING_REGULAR
                        if regularity > 0.3
                        else SleepEventType.BREATHING_IRREGULAR
                    )

                    event = SleepEvent(
                        event_type=event_type,
                        start_time=start_time,
                        end_time=start_time + timedelta(seconds=duration),
                        confidence=min(0.8, regularity + 0.2),
                        speaker_id=speaker_id,
                        metadata={
                            "breathing_rate_hz": breathing_rate,
                            "breaths_per_minute": breathing_rate * 60,
                            "regularity": regularity,
                        },
                    )
                    events.append(event)

        return events

    def _detect_movement(
        self,
        audio: np.ndarray,
        start_time: datetime,
        speaker_id: Optional[str],
    ) -> list[SleepEvent]:
        """Detect movement/rustling sounds."""
        events = []

        # Movement typically shows as broadband noise with sudden onset
        # Compute short-term energy
        frame_size = self._frame_samples
        hop = self._hop_samples

        energy = []
        for i in range(0, len(audio) - frame_size, hop):
            frame = audio[i:i + frame_size]
            energy.append(np.sqrt(np.mean(frame ** 2)))

        energy = np.array(energy)

        if len(energy) > 1:
            # Compute energy derivative (sudden changes indicate movement)
            energy_diff = np.abs(np.diff(energy))

            if energy_diff.max() > 0:
                energy_diff = energy_diff / energy_diff.max()

            # Find movement regions
            is_movement = energy_diff > self.movement_threshold
            movement_regions = self._find_contiguous_regions(is_movement, min_duration=0.2)

            for start_frame, end_frame in movement_regions:
                start_sec = start_frame * hop / self.sample_rate
                end_sec = end_frame * hop / self.sample_rate

                event = SleepEvent(
                    event_type=SleepEventType.MOVEMENT,
                    start_time=start_time + timedelta(seconds=start_sec),
                    end_time=start_time + timedelta(seconds=end_sec),
                    confidence=0.7,
                    speaker_id=speaker_id,
                    intensity=float(energy_diff[start_frame:end_frame].mean()),
                )
                events.append(event)

        return events

    def _detect_talking(
        self,
        audio: np.ndarray,
        start_time: datetime,
        speaker_id: Optional[str],
    ) -> list[SleepEvent]:
        """Detect sleep talking."""
        events = []

        # Use voice activity detection based on spectral features
        # Speech has characteristic formant structure

        hop = self._hop_samples
        n_fft = self._frame_samples * 2

        # Compute mel spectrogram (better for speech)
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_fft=n_fft, hop_length=hop, n_mels=40
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Speech typically has energy in 300-3000 Hz range with formant structure
        # Compute spectral centroid as a simple speech indicator
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, n_fft=n_fft, hop_length=hop
        )[0]

        # Speech centroid typically 500-2000 Hz
        is_speech_like = (centroid > 500) & (centroid < 2000)

        # Also need sufficient energy
        rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop)[0]
        has_energy = rms > 0.02

        is_talking = is_speech_like & has_energy

        talking_regions = self._find_contiguous_regions(is_talking, min_duration=0.5)

        for start_frame, end_frame in talking_regions:
            start_sec = start_frame * hop / self.sample_rate
            end_sec = end_frame * hop / self.sample_rate

            event = SleepEvent(
                event_type=SleepEventType.TALKING,
                start_time=start_time + timedelta(seconds=start_sec),
                end_time=start_time + timedelta(seconds=end_sec),
                confidence=0.6,  # Sleep talking is hard to detect accurately
                speaker_id=speaker_id,
            )
            events.append(event)

        return events

    def _detect_apnea(
        self,
        audio: np.ndarray,
        start_time: datetime,
        speaker_id: Optional[str],
    ) -> list[SleepEvent]:
        """Detect potential apnea events (breathing pauses)."""
        events = []

        # Compute RMS energy
        hop = self._hop_samples
        frame_size = self._frame_samples * 4  # Larger frames for breathing

        rms = librosa.feature.rms(y=audio, frame_length=frame_size, hop_length=hop)[0]

        # Find extended silence periods (potential apnea)
        is_silent = rms < self.silence_threshold

        # Apnea is typically 10+ seconds of breathing pause
        apnea_regions = self._find_contiguous_regions(is_silent, min_duration=10.0)

        for start_frame, end_frame in apnea_regions:
            start_sec = start_frame * hop / self.sample_rate
            end_sec = end_frame * hop / self.sample_rate
            duration = end_sec - start_sec

            # Longer pauses are more concerning
            severity = min(1.0, duration / 30.0)

            event = SleepEvent(
                event_type=SleepEventType.APNEA,
                start_time=start_time + timedelta(seconds=start_sec),
                end_time=start_time + timedelta(seconds=end_sec),
                confidence=0.5,  # Audio alone can't definitively detect apnea
                speaker_id=speaker_id,
                intensity=severity,
                metadata={
                    "pause_duration": duration,
                    "warning": "Consult a doctor for proper sleep apnea diagnosis",
                },
            )
            events.append(event)

        return events

    def _find_contiguous_regions(
        self,
        mask: np.ndarray,
        min_duration: float = 0.5,
    ) -> list[tuple[int, int]]:
        """Find contiguous True regions in a boolean array."""
        regions = []

        # Pad with False to handle edge cases
        padded = np.concatenate([[False], mask, [False]])
        diff = np.diff(padded.astype(int))

        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        min_frames = int(min_duration * self.sample_rate / self._hop_samples)

        for start, end in zip(starts, ends):
            if end - start >= min_frames:
                regions.append((start, end))

        return regions

    def _estimate_sleep_stages(
        self,
        events: list[SleepEvent],
    ) -> list[SleepSegment]:
        """Estimate sleep stages from detected events."""
        if not events:
            return []

        segments = []
        segment_duration = timedelta(minutes=5)  # 5-minute epochs

        # Group events by time
        start = min(e.start_time for e in events)
        end = max(e.end_time for e in events)

        current_time = start
        while current_time < end:
            segment_end = current_time + segment_duration

            # Get events in this segment
            segment_events = [
                e for e in events
                if e.start_time < segment_end and e.end_time > current_time
            ]

            # Estimate stage based on events
            stage = self._classify_sleep_stage(segment_events)

            segment = SleepSegment(
                stage=stage,
                start_time=current_time,
                end_time=segment_end,
                confidence=0.6,
                events=segment_events,
            )
            segments.append(segment)

            current_time = segment_end

        return segments

    def _classify_sleep_stage(self, events: list[SleepEvent]) -> SleepStage:
        """Classify sleep stage based on events in a time window."""
        if not events:
            return SleepStage.UNKNOWN

        event_types = [e.event_type for e in events]

        # Simple heuristic classification
        if SleepEventType.TALKING in event_types or SleepEventType.MOVEMENT in event_types:
            movement_count = sum(1 for e in events if e.event_type == SleepEventType.MOVEMENT)
            if movement_count > 3:
                return SleepStage.AWAKE
            return SleepStage.LIGHT

        if SleepEventType.SNORING in event_types:
            return SleepStage.DEEP

        # Check breathing regularity
        breathing_events = [
            e for e in events
            if e.event_type in [SleepEventType.BREATHING_REGULAR, SleepEventType.BREATHING_IRREGULAR]
        ]

        if breathing_events:
            irregular_count = sum(
                1 for e in breathing_events
                if e.event_type == SleepEventType.BREATHING_IRREGULAR
            )
            if irregular_count > len(breathing_events) / 2:
                return SleepStage.REM

        return SleepStage.LIGHT

    def _generate_summary(
        self,
        session_id: str,
        speaker_id: Optional[str],
        events: list[SleepEvent],
        segments: list[SleepSegment],
    ) -> SleepSummary:
        """Generate a summary of the sleep session."""
        if not events:
            now = datetime.now()
            return SleepSummary(
                session_id=session_id,
                speaker_id=speaker_id,
                start_time=now,
                end_time=now,
                total_duration=0,
                time_in_bed=0,
                estimated_sleep_time=0,
                sleep_efficiency=0,
            )

        start_time = min(e.start_time for e in events)
        end_time = max(e.end_time for e in events)
        total_duration = (end_time - start_time).total_seconds()

        # Calculate time in each stage
        stage_times = {stage: 0.0 for stage in SleepStage}
        for segment in segments:
            stage_times[segment.stage] += segment.duration

        # Count events
        snoring_events = [e for e in events if e.event_type == SleepEventType.SNORING]
        apnea_events = [e for e in events if e.event_type == SleepEventType.APNEA]
        movement_events = [e for e in events if e.event_type == SleepEventType.MOVEMENT]

        snoring_duration = sum(e.duration for e in snoring_events)

        # Calculate sleep efficiency
        time_asleep = stage_times[SleepStage.LIGHT] + stage_times[SleepStage.DEEP] + stage_times[SleepStage.REM]
        sleep_efficiency = (time_asleep / total_duration * 100) if total_duration > 0 else 0

        # Calculate sleep quality score (simple heuristic)
        quality_score = self._calculate_sleep_quality(
            sleep_efficiency,
            stage_times,
            len(apnea_events),
            snoring_duration / total_duration if total_duration > 0 else 0,
        )

        return SleepSummary(
            session_id=session_id,
            speaker_id=speaker_id,
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            time_in_bed=total_duration,
            estimated_sleep_time=time_asleep,
            sleep_efficiency=sleep_efficiency,
            time_awake=stage_times[SleepStage.AWAKE],
            time_light=stage_times[SleepStage.LIGHT],
            time_deep=stage_times[SleepStage.DEEP],
            time_rem=stage_times[SleepStage.REM],
            snoring_episodes=len(snoring_events),
            snoring_duration=snoring_duration,
            apnea_episodes=len(apnea_events),
            movement_episodes=len(movement_events),
            sleep_quality_score=quality_score,
        )

    def _calculate_sleep_quality(
        self,
        efficiency: float,
        stage_times: dict,
        apnea_count: int,
        snoring_ratio: float,
    ) -> float:
        """Calculate an overall sleep quality score (0-100)."""
        score = 0.0

        # Efficiency component (up to 40 points)
        score += min(40, efficiency * 0.4)

        # Deep sleep component (up to 25 points) - ideally 15-20% of sleep
        total_sleep = sum(stage_times.values())
        if total_sleep > 0:
            deep_ratio = stage_times[SleepStage.DEEP] / total_sleep
            if deep_ratio >= 0.15:
                score += 25
            else:
                score += deep_ratio / 0.15 * 25

        # REM component (up to 20 points) - ideally 20-25% of sleep
        if total_sleep > 0:
            rem_ratio = stage_times[SleepStage.REM] / total_sleep
            if rem_ratio >= 0.20:
                score += 20
            else:
                score += rem_ratio / 0.20 * 20

        # Penalties
        score -= apnea_count * 5  # Penalty for apnea events
        score -= snoring_ratio * 15  # Penalty for snoring

        return max(0, min(100, score))

    def _extract_timestamp_from_filename(self, filepath: Path) -> datetime:
        """Extract timestamp from audio chunk filename."""
        # Expected format: chunk_0001_HHMMSS.wav
        try:
            parts = filepath.stem.split("_")
            if len(parts) >= 3:
                time_str = parts[2]
                # Get date from parent directory (session_id format: YYYYMMDD_HHMMSS)
                session_id = filepath.parent.name
                date_str = session_id.split("_")[0]

                dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                return dt
        except (ValueError, IndexError):
            pass

        return datetime.now()
