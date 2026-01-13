"""Speaker detection and diarization for multi-person sleep tracking."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
except ImportError:
    AgglomerativeClustering = None
    StandardScaler = None

from .speaker_profile import SpeakerProfile


class SpeakerDetector:
    """Detects and separates multiple speakers/sleepers in audio."""

    def __init__(
        self,
        sample_rate: int = 16000,
        embedding_dim: int = 256,
        similarity_threshold: float = 0.7,
        min_segment_duration: float = 2.0,
    ):
        self.sample_rate = sample_rate
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.min_segment_duration = min_segment_duration

        # Known speaker profiles
        self.profiles: dict[str, SpeakerProfile] = {}
        self._next_speaker_id = 1

    def add_profile(self, profile: SpeakerProfile):
        """Add a known speaker profile."""
        self.profiles[profile.speaker_id] = profile

    def get_profile(self, speaker_id: str) -> Optional[SpeakerProfile]:
        """Get a speaker profile by ID."""
        return self.profiles.get(speaker_id)

    def detect_speakers(
        self,
        audio: np.ndarray,
        max_speakers: int = 2,
    ) -> list[tuple[str, float, float]]:
        """
        Detect speakers in audio and return segments.

        Returns:
            List of (speaker_id, start_time, end_time) tuples
        """
        if librosa is None:
            raise ImportError("librosa not installed. Run: pip install librosa")

        # Extract features for diarization
        segments = self._segment_audio(audio)
        embeddings = [self._extract_embedding(seg) for seg in segments]

        if not embeddings:
            return []

        embeddings = np.array(embeddings)

        # Cluster embeddings to find speakers
        speaker_labels = self._cluster_speakers(embeddings, max_speakers)

        # Map clusters to speaker IDs
        results = []
        segment_duration = len(audio) / self.sample_rate / len(segments)

        for i, (segment, label) in enumerate(zip(segments, speaker_labels)):
            speaker_id = self._get_or_create_speaker(embeddings[i], label)
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration
            results.append((speaker_id, start_time, end_time))

        # Merge consecutive segments with same speaker
        results = self._merge_consecutive_segments(results)

        return results

    def identify_speaker(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Identify which known speaker this audio belongs to.

        Returns:
            (speaker_id, confidence) tuple
        """
        embedding = self._extract_embedding(audio)

        best_match = None
        best_similarity = -1

        for speaker_id, profile in self.profiles.items():
            similarity = profile.similarity(embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_id

        if best_match is None or best_similarity < self.similarity_threshold:
            # Create new speaker
            speaker_id = f"speaker_{self._next_speaker_id}"
            self._next_speaker_id += 1

            profile = SpeakerProfile(speaker_id=speaker_id)
            profile.update_embedding(embedding)
            self.profiles[speaker_id] = profile

            return speaker_id, 0.5

        # Update existing profile
        self.profiles[best_match].update_embedding(embedding)

        return best_match, best_similarity

    def diarize_session(
        self,
        audio_files: list[Path],
        max_speakers: int = 2,
    ) -> dict[str, list[tuple[Path, float, float]]]:
        """
        Perform diarization on an entire session.

        Returns:
            Dictionary mapping speaker_id to list of (file, start, end) tuples
        """
        speaker_segments: dict[str, list] = {}

        for audio_path in sorted(audio_files):
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            segments = self.detect_speakers(audio, max_speakers)

            for speaker_id, start, end in segments:
                if speaker_id not in speaker_segments:
                    speaker_segments[speaker_id] = []
                speaker_segments[speaker_id].append((audio_path, start, end))

        return speaker_segments

    def extract_speaker_audio(
        self,
        audio: np.ndarray,
        segments: list[tuple[str, float, float]],
        target_speaker: str,
    ) -> np.ndarray:
        """Extract audio for a specific speaker."""
        speaker_audio = []

        for speaker_id, start, end in segments:
            if speaker_id == target_speaker:
                start_sample = int(start * self.sample_rate)
                end_sample = int(end * self.sample_rate)
                speaker_audio.append(audio[start_sample:end_sample])

        if speaker_audio:
            return np.concatenate(speaker_audio)
        return np.array([])

    def _segment_audio(
        self,
        audio: np.ndarray,
        segment_duration: float = 3.0,
    ) -> list[np.ndarray]:
        """Split audio into fixed-length segments."""
        segment_samples = int(segment_duration * self.sample_rate)
        segments = []

        for i in range(0, len(audio) - segment_samples, segment_samples):
            segment = audio[i:i + segment_samples]

            # Only include segments with sufficient energy
            rms = np.sqrt(np.mean(segment ** 2))
            if rms > 0.01:
                segments.append(segment)

        return segments

    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract audio embedding for speaker identification."""
        # Use mel-frequency features as a simple embedding
        n_mfcc = 20
        n_fft = 512
        hop_length = 256

        # Compute MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        # Compute statistics over time
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_delta = np.mean(librosa.feature.delta(mfccs), axis=1)

        # Additional spectral features
        spectral_centroid = np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        )
        spectral_bandwidth = np.mean(
            librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
        )
        spectral_rolloff = np.mean(
            librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        )
        zero_crossing = np.mean(librosa.feature.zero_crossing_rate(audio))

        # Fundamental frequency (F0) estimation
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=50,
            fmax=500,
            sr=self.sample_rate,
        )
        f0_mean = np.nanmean(f0) if np.any(~np.isnan(f0)) else 0
        f0_std = np.nanstd(f0) if np.any(~np.isnan(f0)) else 0

        # Combine into embedding
        embedding = np.concatenate([
            mfcc_mean,
            mfcc_std,
            mfcc_delta,
            [spectral_centroid / 10000],  # Normalize
            [spectral_bandwidth / 10000],
            [spectral_rolloff / 10000],
            [zero_crossing],
            [f0_mean / 500],  # Normalize
            [f0_std / 100],
        ])

        # Pad or truncate to fixed size
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dim]

        return embedding

    def _cluster_speakers(
        self,
        embeddings: np.ndarray,
        max_speakers: int,
    ) -> np.ndarray:
        """Cluster embeddings to identify speakers."""
        if AgglomerativeClustering is None:
            # Fallback: simple threshold-based clustering
            return self._simple_clustering(embeddings, max_speakers)

        if len(embeddings) < 2:
            return np.zeros(len(embeddings), dtype=int)

        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_norm = scaler.fit_transform(embeddings)

        # Try different numbers of clusters
        best_labels = None
        best_score = -1

        for n_clusters in range(1, min(max_speakers + 1, len(embeddings) + 1)):
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(embeddings_norm)

            # Score based on cluster cohesion
            score = self._cluster_score(embeddings_norm, labels)

            if score > best_score:
                best_score = score
                best_labels = labels

        return best_labels if best_labels is not None else np.zeros(len(embeddings), dtype=int)

    def _simple_clustering(
        self,
        embeddings: np.ndarray,
        max_speakers: int,
    ) -> np.ndarray:
        """Simple threshold-based clustering fallback."""
        labels = np.zeros(len(embeddings), dtype=int)
        centroids = [embeddings[0]]

        for i, emb in enumerate(embeddings[1:], 1):
            # Find closest centroid
            similarities = [
                np.dot(emb, c) / (np.linalg.norm(emb) * np.linalg.norm(c) + 1e-8)
                for c in centroids
            ]
            best_idx = np.argmax(similarities)
            best_sim = similarities[best_idx]

            if best_sim > self.similarity_threshold or len(centroids) >= max_speakers:
                labels[i] = best_idx
                # Update centroid
                cluster_mask = labels[:i+1] == best_idx
                centroids[best_idx] = embeddings[:i+1][cluster_mask].mean(axis=0)
            else:
                # New cluster
                labels[i] = len(centroids)
                centroids.append(emb)

        return labels

    def _cluster_score(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Calculate clustering quality score."""
        unique_labels = np.unique(labels)

        if len(unique_labels) == 1:
            return 0.5

        # Calculate average intra-cluster similarity
        intra_sim = 0
        for label in unique_labels:
            mask = labels == label
            cluster_emb = embeddings[mask]
            if len(cluster_emb) > 1:
                centroid = cluster_emb.mean(axis=0)
                sims = [
                    np.dot(e, centroid) / (np.linalg.norm(e) * np.linalg.norm(centroid) + 1e-8)
                    for e in cluster_emb
                ]
                intra_sim += np.mean(sims)

        intra_sim /= len(unique_labels)

        # Calculate average inter-cluster distance
        centroids = [embeddings[labels == l].mean(axis=0) for l in unique_labels]
        inter_dist = 0
        count = 0
        for i, c1 in enumerate(centroids):
            for j, c2 in enumerate(centroids[i+1:], i+1):
                dist = 1 - np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8)
                inter_dist += dist
                count += 1

        inter_dist = inter_dist / count if count > 0 else 0

        return intra_sim * 0.5 + inter_dist * 0.5

    def _get_or_create_speaker(
        self,
        embedding: np.ndarray,
        cluster_label: int,
    ) -> str:
        """Get existing speaker ID or create new one."""
        # Check against known profiles
        for speaker_id, profile in self.profiles.items():
            if profile.similarity(embedding) > self.similarity_threshold:
                profile.update_embedding(embedding)
                return speaker_id

        # Create new speaker
        speaker_id = f"person_{cluster_label + 1}"

        if speaker_id not in self.profiles:
            profile = SpeakerProfile(speaker_id=speaker_id)
            profile.update_embedding(embedding)
            self.profiles[speaker_id] = profile

        return speaker_id

    def _merge_consecutive_segments(
        self,
        segments: list[tuple[str, float, float]],
    ) -> list[tuple[str, float, float]]:
        """Merge consecutive segments with the same speaker."""
        if not segments:
            return []

        merged = []
        current_speaker, current_start, current_end = segments[0]

        for speaker_id, start, end in segments[1:]:
            if speaker_id == current_speaker:
                current_end = end
            else:
                merged.append((current_speaker, current_start, current_end))
                current_speaker, current_start, current_end = speaker_id, start, end

        merged.append((current_speaker, current_start, current_end))

        return merged

    def save_profiles(self, directory: Path):
        """Save all speaker profiles to a directory."""
        directory.mkdir(parents=True, exist_ok=True)
        for speaker_id, profile in self.profiles.items():
            profile.save(directory / f"{speaker_id}.json")

    def load_profiles(self, directory: Path):
        """Load speaker profiles from a directory."""
        if not directory.exists():
            return

        for profile_path in directory.glob("*.json"):
            profile = SpeakerProfile.load(profile_path)
            self.profiles[profile.speaker_id] = profile
