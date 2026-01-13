"""Command-line interface for Sleep Tracker."""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import click
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
except ImportError:
    click = None
    Console = None

from .recorder import AudioRecorder, RecorderConfig
from .analyzer import SleepAnalyzer, SleepEvent
from .diarization import SpeakerDetector
from .storage import SleepDatabase, SleepSession
from .reports import ReportGenerator


console = Console() if Console else None


def print_error(msg: str):
    if console:
        console.print(f"[red]Error:[/red] {msg}")
    else:
        print(f"Error: {msg}")


def print_success(msg: str):
    if console:
        console.print(f"[green]{msg}[/green]")
    else:
        print(msg)


def print_info(msg: str):
    if console:
        console.print(f"[blue]{msg}[/blue]")
    else:
        print(msg)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Sleep Tracker - Record and analyze your sleep patterns."""
    pass


@cli.command()
@click.option("--output-dir", "-o", type=click.Path(), default="./recordings",
              help="Directory to save recordings")
@click.option("--duration", "-d", type=float, default=None,
              help="Recording duration in hours (default: until stopped)")
@click.option("--session-name", "-n", type=str, default=None,
              help="Custom session name")
def record(output_dir: str, duration: Optional[float], session_name: Optional[str]):
    """Start a sleep recording session."""
    config = RecorderConfig(output_dir=Path(output_dir))
    recorder = AudioRecorder(config)
    db = SleepDatabase()

    try:
        session_id = recorder.start_session(session_name)
        print_success(f"Recording started: {session_id}")
        print_info("Press Ctrl+C to stop recording...")

        # Save session to database
        session = SleepSession(
            session_id=session_id,
            start_time=datetime.now(),
            status="recording",
            recording_path=str(config.output_dir / session_id),
        )
        db.save_session(session)

        start_time = time.time()
        duration_seconds = duration * 3600 if duration else None

        while True:
            time.sleep(1)

            # Check duration limit
            if duration_seconds and (time.time() - start_time) >= duration_seconds:
                print_info("\nDuration limit reached.")
                break

            # Show status
            elapsed = time.time() - start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)

            status = "PAUSED (silence)" if recorder.is_paused else "RECORDING"
            sys.stdout.write(f"\r[{status}] {hours:02d}:{minutes:02d}:{seconds:02d}")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print_info("\nStopping recording...")

    finally:
        session_dir = recorder.stop_session()
        print_success(f"\nRecording saved to: {session_dir}")

        # Update session
        session.end_time = datetime.now()
        session.status = "completed"
        db.save_session(session)


@cli.command()
@click.argument("session_path", type=click.Path(exists=True))
@click.option("--max-speakers", "-s", type=int, default=2,
              help="Maximum number of speakers to detect")
@click.option("--output-dir", "-o", type=click.Path(), default="./reports",
              help="Directory to save reports")
def analyze(session_path: str, max_speakers: int, output_dir: str):
    """Analyze a recorded sleep session."""
    session_dir = Path(session_path)
    output_path = Path(output_dir)

    # Find audio files
    audio_files = sorted(session_dir.glob("*.wav"))
    if not audio_files:
        print_error(f"No audio files found in {session_path}")
        return

    print_info(f"Found {len(audio_files)} audio files")

    # Initialize components
    analyzer = SleepAnalyzer()
    detector = SpeakerDetector()
    reporter = ReportGenerator(output_path)
    db = SleepDatabase()

    session_id = session_dir.name

    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Step 1: Speaker diarization
            task = progress.add_task("Detecting speakers...", total=None)
            speaker_segments = detector.diarize_session(audio_files, max_speakers)
            speakers = list(speaker_segments.keys())
            progress.update(task, completed=True)
            print_success(f"Detected {len(speakers)} speaker(s): {', '.join(speakers)}")

            # Step 2: Analyze each speaker separately
            all_summaries = []

            for speaker_id in speakers:
                task = progress.add_task(f"Analyzing {speaker_id}...", total=None)

                events, segments, summary = analyzer.analyze_session(
                    audio_files,
                    session_id,
                    speaker_id,
                )

                # Save events to database
                for event in events:
                    db.save_event(
                        session_id=session_id,
                        event_type=event.event_type.value,
                        start_time=event.start_time,
                        end_time=event.end_time,
                        speaker_id=speaker_id,
                        confidence=event.confidence,
                        intensity=event.intensity,
                        metadata=event.metadata,
                    )

                # Save summary
                db.save_summary(summary.to_dict())
                all_summaries.append((speaker_id, summary))

                progress.update(task, completed=True)

            # Step 3: Generate reports
            task = progress.add_task("Generating reports...", total=None)

            for speaker_id, summary in all_summaries:
                events = db.get_events(session_id, speaker_id)

                # Text report
                text_report = reporter.generate_text_report(summary, events, speaker_id)
                report_path = output_path / f"{session_id}_{speaker_id}_report.txt"
                with open(report_path, "w") as f:
                    f.write(text_report)

                # Chart
                chart_path = reporter.generate_chart(summary, events, speaker_label=speaker_id)

                # JSON report
                reporter.save_json_report(summary, events)

                print_success(f"\nReport for {speaker_id}:")
                console.print(Panel(text_report, title=speaker_id))

            # Comparison report if multiple speakers
            if len(all_summaries) > 1:
                comparison = reporter.generate_comparison_report(all_summaries)
                comparison_path = output_path / f"{session_id}_comparison.txt"
                with open(comparison_path, "w") as f:
                    f.write(comparison)
                print_success("\nComparison Report:")
                console.print(Panel(comparison, title="Multi-Person Comparison"))

            progress.update(task, completed=True)

    else:
        # Fallback without rich
        print("Detecting speakers...")
        speaker_segments = detector.diarize_session(audio_files, max_speakers)
        speakers = list(speaker_segments.keys())
        print(f"Detected {len(speakers)} speaker(s)")

        for speaker_id in speakers:
            print(f"Analyzing {speaker_id}...")
            events, segments, summary = analyzer.analyze_session(
                audio_files, session_id, speaker_id
            )
            print(f"\n{speaker_id} Sleep Summary:")
            print(f"  Sleep Quality: {summary.sleep_quality_score:.0f}/100")
            print(f"  Sleep Efficiency: {summary.sleep_efficiency:.1f}%")
            print(f"  Snoring Episodes: {summary.snoring_episodes}")

    print_success(f"\nReports saved to: {output_path}")


@cli.command()
@click.option("--limit", "-l", type=int, default=10, help="Number of sessions to show")
def history(limit: int):
    """Show recording history."""
    db = SleepDatabase()
    sessions = db.list_sessions(limit)

    if not sessions:
        print_info("No sessions found.")
        return

    if console:
        table = Table(title="Sleep Sessions")
        table.add_column("Session ID", style="cyan")
        table.add_column("Date", style="green")
        table.add_column("Duration")
        table.add_column("Status")
        table.add_column("Speakers")

        for session in sessions:
            duration = ""
            if session.end_time and session.start_time:
                dur = (session.end_time - session.start_time).total_seconds()
                hours = int(dur // 3600)
                minutes = int((dur % 3600) // 60)
                duration = f"{hours}h {minutes}m"

            table.add_row(
                session.session_id,
                session.start_time.strftime("%Y-%m-%d %H:%M"),
                duration,
                session.status,
                ", ".join(session.speakers_detected) or "-",
            )

        console.print(table)
    else:
        print("Sleep Sessions:")
        for session in sessions:
            print(f"  {session.session_id} - {session.start_time} - {session.status}")


@cli.command()
@click.argument("session_id")
@click.option("--speaker", "-s", type=str, default=None, help="Filter by speaker")
def report(session_id: str, speaker: Optional[str]):
    """View report for a session."""
    db = SleepDatabase()
    summaries = db.get_summaries(session_id, speaker)

    if not summaries:
        print_error(f"No analysis found for session: {session_id}")
        return

    reporter = ReportGenerator()

    for summary_data in summaries:
        speaker_id = summary_data.get("speaker_id")
        events = db.get_events(session_id, speaker_id)

        # Reconstruct summary object
        from .analyzer.events import SleepSummary
        summary = SleepSummary(
            session_id=session_id,
            speaker_id=speaker_id,
            start_time=datetime.fromisoformat(summary_data.get("created_at", datetime.now().isoformat())),
            end_time=datetime.now(),  # Placeholder
            total_duration=summary_data.get("total_duration", 0),
            time_in_bed=summary_data.get("total_duration", 0),
            estimated_sleep_time=summary_data.get("total_duration", 0) * (summary_data.get("sleep_efficiency", 0) / 100),
            sleep_efficiency=summary_data.get("sleep_efficiency", 0),
            time_awake=summary_data.get("time_awake", 0),
            time_light=summary_data.get("time_light", 0),
            time_deep=summary_data.get("time_deep", 0),
            time_rem=summary_data.get("time_rem", 0),
            snoring_episodes=summary_data.get("snoring_episodes", 0),
            snoring_duration=summary_data.get("snoring_duration", 0),
            apnea_episodes=summary_data.get("apnea_episodes", 0),
            sleep_quality_score=summary_data.get("sleep_quality_score", 0),
        )

        text = reporter.generate_text_report(summary, events, speaker_id)
        if console:
            console.print(Panel(text, title=f"Report: {speaker_id or session_id}"))
        else:
            print(text)


@cli.command()
@click.argument("name")
@click.option("--samples", "-s", type=int, default=5,
              help="Number of audio samples to collect")
def enroll(name: str, samples: int):
    """Enroll a new person for speaker recognition."""
    print_info(f"Enrolling '{name}' for speaker recognition")
    print_info(f"Will collect {samples} voice samples")
    print_info("Make some sounds (snoring, breathing, talking) when prompted...")

    detector = SpeakerDetector()
    config = RecorderConfig()
    recorder = AudioRecorder(config)

    # Load existing profiles
    profiles_dir = Path("./profiles")
    detector.load_profiles(profiles_dir)

    from .diarization import SpeakerProfile
    import numpy as np

    profile = SpeakerProfile(speaker_id=name, name=name)

    try:
        for i in range(samples):
            print_info(f"\nSample {i+1}/{samples}: Press Enter to start recording (5 seconds)...")
            input()

            # Record sample
            import sounddevice as sd
            sample_duration = 5.0
            audio = sd.rec(
                int(sample_duration * config.sample_rate),
                samplerate=config.sample_rate,
                channels=1,
                dtype=np.float32,
            )
            sd.wait()
            audio = audio.flatten()

            # Extract embedding
            embedding = detector._extract_embedding(audio)
            profile.update_embedding(embedding)

            print_success(f"Sample {i+1} captured")

    except KeyboardInterrupt:
        print_info("\nEnrollment cancelled")
        return

    # Save profile
    detector.profiles[name] = profile
    detector.save_profiles(profiles_dir)

    print_success(f"\n'{name}' enrolled successfully!")
    print_info(f"Profile saved to: {profiles_dir / f'{name}.json'}")


def main():
    """Main entry point."""
    if click is None:
        print("Error: Required packages not installed.")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    cli()


if __name__ == "__main__":
    main()
