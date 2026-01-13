"""Report generation for sleep analysis."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import json

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    plt = None
    mdates = None

from ..analyzer.events import SleepSummary, SleepStage, SleepEventType


class ReportGenerator:
    """Generates sleep analysis reports."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("./reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_text_report(
        self,
        summary: SleepSummary,
        events: list[dict],
        speaker_label: Optional[str] = None,
    ) -> str:
        """Generate a text report for a sleep session."""
        speaker_name = speaker_label or summary.speaker_id or "Unknown"

        lines = [
            "=" * 60,
            f"SLEEP ANALYSIS REPORT - {speaker_name}",
            "=" * 60,
            "",
            f"Session ID: {summary.session_id}",
            f"Date: {summary.start_time.strftime('%Y-%m-%d')}",
            f"Time: {summary.start_time.strftime('%H:%M')} - {summary.end_time.strftime('%H:%M')}",
            "",
            "-" * 40,
            "SLEEP OVERVIEW",
            "-" * 40,
            f"Time in Bed:        {self._format_duration(summary.time_in_bed)}",
            f"Total Sleep Time:   {self._format_duration(summary.estimated_sleep_time)}",
            f"Sleep Efficiency:   {summary.sleep_efficiency:.1f}%",
            f"Sleep Quality:      {summary.sleep_quality_score:.0f}/100",
            "",
            "-" * 40,
            "SLEEP STAGES",
            "-" * 40,
            f"Awake:              {self._format_duration(summary.time_awake)} ({self._percent(summary.time_awake, summary.total_duration)}%)",
            f"Light Sleep:        {self._format_duration(summary.time_light)} ({self._percent(summary.time_light, summary.total_duration)}%)",
            f"Deep Sleep:         {self._format_duration(summary.time_deep)} ({self._percent(summary.time_deep, summary.total_duration)}%)",
            f"REM Sleep:          {self._format_duration(summary.time_rem)} ({self._percent(summary.time_rem, summary.total_duration)}%)",
            "",
            "-" * 40,
            "SLEEP EVENTS",
            "-" * 40,
            f"Snoring Episodes:   {summary.snoring_episodes}",
            f"Total Snoring:      {self._format_duration(summary.snoring_duration)}",
            f"Apnea Events:       {summary.apnea_episodes}",
            f"Movement Episodes:  {summary.movement_episodes}",
            "",
        ]

        # Add warnings if needed
        warnings = self._generate_warnings(summary)
        if warnings:
            lines.extend([
                "-" * 40,
                "HEALTH INSIGHTS",
                "-" * 40,
            ])
            lines.extend(warnings)
            lines.append("")

        # Add recommendations
        recommendations = self._generate_recommendations(summary)
        if recommendations:
            lines.extend([
                "-" * 40,
                "RECOMMENDATIONS",
                "-" * 40,
            ])
            lines.extend(recommendations)
            lines.append("")

        lines.extend([
            "=" * 60,
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
        ])

        return "\n".join(lines)

    def generate_comparison_report(
        self,
        summaries: list[tuple[str, SleepSummary]],
    ) -> str:
        """Generate a comparison report for multiple people."""
        if not summaries:
            return "No data available for comparison."

        lines = [
            "=" * 70,
            "MULTI-PERSON SLEEP COMPARISON REPORT",
            "=" * 70,
            "",
        ]

        # Header
        header = f"{'Metric':<25}"
        for name, _ in summaries:
            header += f"{name:<20}"
        lines.append(header)
        lines.append("-" * 70)

        # Metrics
        metrics = [
            ("Total Sleep", lambda s: self._format_duration(s.estimated_sleep_time)),
            ("Sleep Efficiency", lambda s: f"{s.sleep_efficiency:.1f}%"),
            ("Sleep Quality", lambda s: f"{s.sleep_quality_score:.0f}/100"),
            ("Deep Sleep", lambda s: self._format_duration(s.time_deep)),
            ("REM Sleep", lambda s: self._format_duration(s.time_rem)),
            ("Snoring Episodes", lambda s: str(s.snoring_episodes)),
            ("Snoring Duration", lambda s: self._format_duration(s.snoring_duration)),
            ("Apnea Events", lambda s: str(s.apnea_episodes)),
        ]

        for metric_name, metric_fn in metrics:
            row = f"{metric_name:<25}"
            for _, summary in summaries:
                row += f"{metric_fn(summary):<20}"
            lines.append(row)

        lines.extend([
            "",
            "=" * 70,
        ])

        return "\n".join(lines)

    def generate_chart(
        self,
        summary: SleepSummary,
        events: list[dict],
        output_path: Optional[Path] = None,
        speaker_label: Optional[str] = None,
    ) -> Optional[Path]:
        """Generate sleep chart visualization."""
        if plt is None:
            return None

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        speaker_name = speaker_label or summary.speaker_id or "Sleep"

        # 1. Sleep stages over time
        ax1 = axes[0]
        self._plot_sleep_stages(ax1, events, summary)
        ax1.set_title(f"{speaker_name} - Sleep Stages")

        # 2. Sleep stage distribution (pie chart)
        ax2 = axes[1]
        self._plot_stage_distribution(ax2, summary)
        ax2.set_title("Sleep Stage Distribution")

        # 3. Events timeline
        ax3 = axes[2]
        self._plot_events_timeline(ax3, events, summary)
        ax3.set_title("Sleep Events")

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / f"{summary.session_id}_{summary.speaker_id or 'report'}.png"

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_sleep_stages(self, ax, events: list[dict], summary: SleepSummary):
        """Plot sleep stages over time."""
        stage_mapping = {
            "awake": 4,
            "light": 3,
            "rem": 2,
            "deep": 1,
            "unknown": 0,
        }

        # Group events by time to estimate stages
        times = []
        stages = []

        current_time = summary.start_time
        while current_time < summary.end_time:
            times.append(current_time)

            # Determine stage at this time
            stage = self._get_stage_at_time(events, current_time)
            stages.append(stage_mapping.get(stage, 0))

            current_time += timedelta(minutes=5)

        if times:
            ax.fill_between(times, stages, alpha=0.3)
            ax.plot(times, stages, linewidth=2)

        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(['Deep', 'REM', 'Light', 'Awake'])
        ax.set_ylim(0.5, 4.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_xlabel('Time')

    def _plot_stage_distribution(self, ax, summary: SleepSummary):
        """Plot pie chart of sleep stage distribution."""
        labels = []
        sizes = []
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']

        if summary.time_awake > 0:
            labels.append(f'Awake\n{self._format_duration(summary.time_awake)}')
            sizes.append(summary.time_awake)

        if summary.time_light > 0:
            labels.append(f'Light\n{self._format_duration(summary.time_light)}')
            sizes.append(summary.time_light)

        if summary.time_rem > 0:
            labels.append(f'REM\n{self._format_duration(summary.time_rem)}')
            sizes.append(summary.time_rem)

        if summary.time_deep > 0:
            labels.append(f'Deep\n{self._format_duration(summary.time_deep)}')
            sizes.append(summary.time_deep)

        if sizes:
            ax.pie(sizes, labels=labels, colors=colors[:len(sizes)],
                   autopct='%1.1f%%', startangle=90)
        ax.axis('equal')

    def _plot_events_timeline(self, ax, events: list[dict], summary: SleepSummary):
        """Plot events on a timeline."""
        event_colors = {
            'snoring': '#ff6b6b',
            'apnea': '#e74c3c',
            'movement': '#3498db',
            'talking': '#9b59b6',
        }

        y_positions = {
            'snoring': 3,
            'apnea': 2,
            'movement': 1,
            'talking': 0,
        }

        for event in events:
            event_type = event.get('event_type', '').lower()
            if event_type in y_positions:
                start = datetime.fromisoformat(event['start_time'])
                end = datetime.fromisoformat(event['end_time'])
                duration = (end - start).total_seconds() / 60  # minutes

                y = y_positions[event_type]
                color = event_colors.get(event_type, '#95a5a6')

                ax.barh(y, duration, left=mdates.date2num(start),
                        height=0.6, color=color, alpha=0.7)

        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels([k.capitalize() for k in y_positions.keys()])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_xlabel('Time')
        ax.set_xlim(summary.start_time, summary.end_time)

    def _get_stage_at_time(self, events: list[dict], time: datetime) -> str:
        """Determine sleep stage at a given time based on events."""
        for event in events:
            start = datetime.fromisoformat(event['start_time'])
            end = datetime.fromisoformat(event['end_time'])

            if start <= time <= end:
                event_type = event.get('event_type', '')
                if event_type in ['movement', 'talking']:
                    return 'light'
                elif event_type == 'snoring':
                    return 'deep'

        return 'light'  # Default

    def _format_duration(self, seconds: float) -> str:
        """Format duration in hours and minutes."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    def _percent(self, value: float, total: float) -> str:
        """Calculate percentage."""
        if total == 0:
            return "0"
        return f"{value / total * 100:.0f}"

    def _generate_warnings(self, summary: SleepSummary) -> list[str]:
        """Generate health warnings based on analysis."""
        warnings = []

        if summary.apnea_episodes > 5:
            warnings.append(
                "! IMPORTANT: Multiple potential apnea events detected. "
                "Please consult a sleep specialist for proper evaluation."
            )

        if summary.snoring_duration > summary.total_duration * 0.3:
            warnings.append(
                "! Heavy snoring detected (>30% of sleep time). "
                "This may indicate sleep-disordered breathing."
            )

        if summary.sleep_efficiency < 75:
            warnings.append(
                "! Low sleep efficiency. Consider evaluating your sleep environment "
                "and bedtime habits."
            )

        if summary.time_deep < summary.total_duration * 0.10:
            warnings.append(
                "! Below average deep sleep. Deep sleep is crucial for "
                "physical recovery and immune function."
            )

        return warnings

    def _generate_recommendations(self, summary: SleepSummary) -> list[str]:
        """Generate sleep recommendations."""
        recommendations = []

        if summary.sleep_quality_score < 70:
            recommendations.append(
                "- Consider maintaining a consistent sleep schedule"
            )
            recommendations.append(
                "- Avoid screens 1 hour before bedtime"
            )

        if summary.snoring_episodes > 10:
            recommendations.append(
                "- Try sleeping on your side to reduce snoring"
            )
            recommendations.append(
                "- Ensure bedroom is well-ventilated"
            )

        if summary.movement_episodes > 20:
            recommendations.append(
                "- Check mattress comfort and room temperature"
            )

        if summary.time_deep < summary.total_duration * 0.15:
            recommendations.append(
                "- Regular exercise (not too close to bedtime) can increase deep sleep"
            )
            recommendations.append(
                "- Avoid alcohol before bed as it reduces deep sleep"
            )

        return recommendations

    def save_json_report(
        self,
        summary: SleepSummary,
        events: list[dict],
        output_path: Optional[Path] = None,
    ) -> Path:
        """Save report data as JSON."""
        if output_path is None:
            output_path = self.output_dir / f"{summary.session_id}_{summary.speaker_id or 'report'}.json"

        data = {
            "summary": summary.to_dict(),
            "events": events,
            "generated_at": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path
