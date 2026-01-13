# Sleep Tracking Analyzer

A comprehensive sleep monitoring application that records audio during sleep, analyzes sleep patterns, and supports multi-person detection with separate analysis for each sleeper.

## Features

- **Audio Recording**: Records throughout the night with smart silence detection to save storage
- **Sleep Pattern Analysis**: Detects and analyzes:
  - Snoring episodes
  - Breathing patterns (regular/irregular)
  - Potential apnea events
  - Movement and restlessness
  - Sleep talking
- **Multi-Person Support**: Automatically detects and separates multiple sleepers, providing individual analysis for each person
- **Sleep Stage Estimation**: Estimates sleep stages (Awake, Light, Deep, REM) based on audio patterns
- **Detailed Reports**: Generates comprehensive reports including:
  - Sleep efficiency
  - Sleep quality score
  - Time in each sleep stage
  - Event summaries
  - Health insights and recommendations
- **Visualization**: Creates charts showing sleep stages and events over time

## Installation

```bash
# Clone the repository
git clone https://github.com/geg00/test.git
cd test

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Usage

### Start Recording

```bash
# Start a sleep recording session
sleep-tracker record

# With options
sleep-tracker record --duration 8 --output-dir ./my_recordings --session-name "night_jan_13"
```

Press `Ctrl+C` to stop recording.

### Analyze a Session

```bash
# Analyze a recorded session
sleep-tracker analyze ./recordings/20260113_220000

# With options (detect up to 2 people)
sleep-tracker analyze ./recordings/20260113_220000 --max-speakers 2
```

### View History

```bash
# Show recent sessions
sleep-tracker history

# Show more sessions
sleep-tracker history --limit 20
```

### View Reports

```bash
# View report for a specific session
sleep-tracker report 20260113_220000

# View report for a specific person
sleep-tracker report 20260113_220000 --speaker person_1
```

### Enroll a Person (for better recognition)

```bash
# Enroll a new person for speaker recognition
sleep-tracker enroll "John"
```

## Multi-Person Detection

When two or more people are sleeping, the app will:

1. Automatically detect multiple sound sources
2. Create separate profiles for each person (person_1, person_2, etc.)
3. Analyze sleep patterns independently for each person
4. Generate individual reports for each sleeper
5. Provide a comparison report showing side-by-side metrics

## Output Files

After analysis, the following files are generated in the `./reports` directory:

- `{session_id}_{speaker}_report.txt` - Text report for each person
- `{session_id}_{speaker}.png` - Sleep visualization chart
- `{session_id}_{speaker}.json` - JSON data for integration
- `{session_id}_comparison.txt` - Multi-person comparison report

## Sleep Quality Score

The sleep quality score (0-100) is calculated based on:

- Sleep efficiency (time asleep / time in bed)
- Percentage of deep sleep (ideal: 15-20%)
- Percentage of REM sleep (ideal: 20-25%)
- Snoring frequency (penalty)
- Apnea events (penalty)

## Health Warnings

The app will alert you about:

- Potential sleep apnea (multiple breathing pauses)
- Excessive snoring (>30% of sleep time)
- Low sleep efficiency (<75%)
- Insufficient deep sleep

**Note**: This app is not a medical device. Consult a healthcare professional for proper sleep disorder diagnosis.

## Project Structure

```
sleep_tracker/
├── __init__.py
├── __main__.py
├── cli.py                  # Command-line interface
├── recorder/
│   ├── __init__.py
│   ├── audio_recorder.py   # Audio recording logic
│   └── config.py           # Recording configuration
├── analyzer/
│   ├── __init__.py
│   ├── events.py           # Sleep event definitions
│   └── sleep_analyzer.py   # Sleep pattern analysis
├── diarization/
│   ├── __init__.py
│   ├── speaker_detector.py # Multi-person detection
│   └── speaker_profile.py  # Speaker profiles
├── storage/
│   ├── __init__.py
│   ├── database.py         # SQLite storage
│   └── models.py           # Data models
└── reports/
    ├── __init__.py
    └── report_generator.py # Report generation
```

## Requirements

- Python 3.10+
- Working microphone
- See `requirements.txt` for full dependencies

## License

MIT License
