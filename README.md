# AI Martial Arts Motion Tracker

Local, camera-based training assistant. It estimates a single athlete's pose, measures joint angles, tracks hand/foot motion, and applies **rule-based** punch/kick heuristics. Sessions are stored in SQLite on your machine.

This is a training aid. It is **not** a force meter, medical device, or official scoring system.

## Features

- Real-time webcam pose skeleton (MediaPipe Pose)
- Joint angles (elbows, shoulders, hips, knees)
- Hand, foot, knee, hip, and shoulder-rotation tracking
- Rule-based detection: jab, cross, hook, straight punch, front kick, roundhouse-style kick, side kick
- Stance / form heuristics with a 0–100 training score
- Debounced live coaching cues (green / yellow / red)
- Video upload analysis (MP4, AVI, MOV)
- Session history, Plotly progress charts, athlete profiles
- Optional camera calibration for **estimated** physical speed
- Optional estimated 3D skeleton (monocular, not a body scan)
- Local session reports (download as text)
- Dataset collector + ML feature scaffold (no pretrained classifier)

## Architecture

```
app.py                 Streamlit entry
config/                Settings and .env
core/                  Pose, angles, motion, techniques, scoring, reports
martial_arts/          Style plugins (boxing, kickboxing, karate, …)
models/                Dataclasses
database/              SQLite
ui/                    Dashboard pages
utils/                 Geometry, video, logging
sparring/              Future two-person interfaces only
ml/                    Labeled-sequence collection + feature extraction
tests/                 pytest
data/                  Local DB, logs, uploads (not sent to a server)
```

Vision, persistence, and UI are separate. Adding a new style means enabling technique IDs in `martial_arts/` — detection rules live in `core/technique_detector.py`.

## Requirements

- **Python 3.10–3.12** (3.12 recommended). MediaPipe Pose is most reliable here.
- A webcam for Live Tracker (optional if you only analyze videos)
- macOS / Windows / Linux with camera permissions granted to the terminal or Cursor

## Installation

```bash
cd martial_arts_tracker
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # optional
```

The first Live Tracker / Video Analysis run downloads a ~5.5 MB MediaPipe Pose Landmarker into `data/models/` if it is not already present (needed for MediaPipe 0.10.30+, which uses the Tasks API).

```bash
streamlit run app.py
```

The app opens in your browser. Use the sidebar:

1. Dashboard  
2. Live Tracker  
3. Video Analysis  
4. Technique Analysis  
5. Performance  
6. Session History  
7. Athlete Profile  
8. Settings  

## Camera setup

1. Grant camera access to Terminal, iTerm, VS Code, or Cursor (macOS: System Settings → Privacy & Security → Camera).
2. On Live Tracker, set **Camera** to `0` (built-in) or `1` (USB).
3. Stand far enough that **head to feet** are visible, with contrast against the background.
4. If the camera fails, the app shows a friendly error — use **Video Analysis** instead.

A red **CAMERA ACTIVE** banner is shown while the webcam is running. Frames are processed locally.

## Calibration

Ordinary webcams do not know real-world scale. By default the app reports:

| Quantity | Meaning |
| --- | --- |
| Pixel speed | pixels / second |
| Normalized speed | image-fraction / second |
| Estimated m/s | only if height or pixels-per-meter is set |

In **Settings** (and optional athlete height on the profile), you can enter a reference distance, camera distance, or pixels/meter.

Accuracy depends on camera position, lighting, perspective, frame rate, and calibration. Uncalibrated values are **never** shown as exact m/s. Impact force is not measured.

## Supported techniques (MVP)

Rule-based geometry — not a trained fight model. Low confidence shows **Technique uncertain**.

**Punches:** jab, cross, hook, straight punch  
**Kicks:** front kick, roundhouse-style kick, side kick  

Styles (plugins): Boxing (punches only), Kickboxing, Karate, Taekwondo, Muay Thai, MMA, Kung Fu. Elbows, knees, and clinch weapons are not classified yet.

## Scoring

Overall = Accuracy 30% + Alignment 20% + Extension 20% + Balance 15% + Consistency 15%.

These are training heuristics, not competition scores.

## Privacy and safety

- No face recognition and no biometric identity profiles
- No upload of video to external servers by default
- Delete athletes or sessions from the UI
- Coaching cues are not injury diagnoses

## Tests

```bash
pytest
```

Coverage includes angles, distances, velocities, technique rules, scoring, SQLite, missing landmarks, empty video, and unavailable cameras.

## Troubleshooting

| Problem | What to try |
| --- | --- |
| `streamlit: command not found` | Activate `.venv` and reinstall requirements |
| MediaPipe import error on Python 3.13 | Recreate the venv with Python 3.12 |
| Camera flickers, freezes, or drops | Close Zoom/Meet/Photo Booth. Use camera index `0`. Start the session once and leave style/resolution alone until you stop. Grant Camera permission to Terminal or Cursor. |
| Pose not detected clearly | Better lighting, full body in frame, fewer people |
| App feels slow | Lower resolution width (320–480); the pipeline skips frames if FPS drops |
| `Could not open video` | Re-export as H.264 MP4 |

## Project screenshots

Run the app and capture:

- Dashboard metric cards and score trend
- Live Tracker with skeleton overlay and live metrics
- Video Analysis timeline
- Performance charts
- Session report download

## Limitations

- Single-person tracking only
- Monocular 3D is an estimate
- Technique labels are geometric heuristics
- Speed without calibration is not physical velocity
- Multiple people in view are not separated (sparring mode is an interface stub)
- No ML classifier is trained in this repository

## Future improvements

- Labeled dataset → sequence classifier (LSTM / TCN / transformer)
- Dedicated multi-person pose for sparring (distance, attack rate, reaction to a cue)
- Style-specific kata / poomsae templates
- Wearable IMU fusion
- Calibrated stereo or depth cameras for metric speed

## License

For personal training use. MediaPipe and OpenCV are used under their respective licenses.
