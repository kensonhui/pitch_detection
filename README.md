# Audio Analysis

This repository is a playground for exploring audio analysis techniques.

## Getting started

```bash
# Optional venv
python -m venv venv
source venv/bin/activate
# Dependancies
pip install librosa, numpy, soundfile, matplotlib
# Run
python yin_pitch_detection.py
```

yin_pitch_detection.py
- Smaller window size seems to get better results until you reach 128, where it gets significantly worse.
- Future: Convert to MIDI