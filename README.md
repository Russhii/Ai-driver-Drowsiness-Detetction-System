# Driver Drowsiness Detection System

A real-time multimodal drowsiness detection system built in Python using
OpenCV. Implements vision-based facial analysis fused with steering wheel
pressure sensing, exactly as proposed in our literature review.

---

## Project Structure

```
drowsiness_detector/
│
├── main.py                    ← Run this to start the system
├── detector.py                ← Core vision detection engine
├── alert_system.py            ← Multi-level alert + CSV logging
├── pressure_sensor.py         ← Arduino interface + simulator
├── hud.py                     ← On-screen HUD renderer
├── arduino_pressure_sensor.ino← Upload this to Arduino
├── evaluate_dataset.py        ← Offline dataset evaluation
├── requirements.txt           ← Python dependencies
├── logs/
│   ├── alert_log.csv          ← Auto-generated event log
│   └── screenshots/           ← Saved frames (press S)
```

---

## What is Implemented (~35% of full project)

| Module | Status | What it does |
|--------|--------|-------------|
| Face detection | ✅ Done | MediaPipe Face Mesh, with OpenCV DNN fallback |
| Eye state (EAR) | ✅ Done | Open/closed detection, consecutive frame counter |
| Yawn detection (MOR) | ✅ Done | Mouth open ratio via smile cascade |
| Head pose proxy | ✅ Done | Vertical face position tracking |
| Drowsiness score | ✅ Done | Weighted fusion of eye+yawn+head (0→1) |
| Alert state machine | ✅ Done | ALERT → WARNING → DROWSY with cooldown |
| Pressure sensor | ✅ Done | Arduino serial reader + simulator |
| Fusion logic | ✅ Done | Two-modality confirmation before DROWSY alert |
| HUD overlay | ✅ Done | Live display with all indicators |
| CSV event logging | ✅ Done | Timestamped alert log |
| Arduino firmware | ✅ Done | Upload-ready .ino sketch |
| CNN eye classifier | 🔲 Next | Replace Haar with trained CNN model |
| Dataset training | 🔲 Next | Train on MRL Eye / NTHU-DDD dataset |
| EEG integration | 🔲 Future | Add physiological signal channel |

---

## Installation

Use Python 3.9–3.12. From the project folder, create and activate a virtual
environment, then install the compatible dependencies:

```bash
py -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For real Arduino hardware, also install:

```bash
python -m pip install pyserial
```

---

## Running the System

### With webcam (no hardware sensor):
```bash
python main.py
```

### Simulate drowsy pressure (demo mode):
```bash
python main.py --simulate-drowsy
```

### With Arduino pressure sensor:
```bash
python main.py --serial-port COM3        # Windows
python main.py --serial-port /dev/ttyUSB0  # Linux/Mac
```

### From a video file:
```bash
python main.py --source test_video.mp4
```

### Headless (no window, logs only):
```bash
python main.py --source test_video.mp4 --no-window
```

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| Q   | Quit |
| R   | Reset drowsiness counters |
| S   | Save screenshot to logs/ |
| D   | Toggle simulated drowsy pressure |

---

## How the Detection Works

```
Camera Frame
     │
     ▼
Face Mesh (MediaPipe; OpenCV DNN fallback)
     │
    ├──► Eye landmarks/texture → EAR score
    ├──► Mouth landmarks/texture → MAR score
    └──► Head pose/face position → Head score
                │
                ▼
         Weighted Fusion
         (Eye×0.60 + Yawn×0.25 + Head×0.15)
                │
                ▼
         Drowsiness Score (0→1)
                │
         ┌──── + ────┐
         │           │
    Pressure     Score > 0.55?
    Sensor           │
    Grip Lost?       │
         │           │
         └───── AND ─┘
                │
                ▼
         DROWSY ALERT 🚨
```

**Fusion rule:** DROWSY alert requires BOTH the vision score to exceed
0.55 AND the pressure sensor to detect grip loss. This prevents false
alarms from brief blinks or lighting changes — addressing the key
limitation found in single-modal systems in our literature review.

---

## Alert Log Format

Saved to `logs/alert_log.csv`:

```
timestamp,elapsed_sec,alert_level,drowsiness_score
2025-03-15 14:23:01,45.2,DROWSY,0.712
2025-03-15 14:23:19,63.7,DROWSY,0.689
```

---

## Thresholds (Tunable)

In `detector.py`:
```python
EYE_CLOSED_CONSEC_FRAMES = 18
YAWN_CONSEC_FRAMES       = 20
SCORE_DROWSY_THRESH      = 0.50
SCORE_ALERT_THRESH       = 0.25
```

In `pressure_sensor.py`:
```python
PRESSURE_THRESHOLD = 200.0   # Pa (from Guardian Alert paper)
```

In `alert_system.py`:
```python
warning_hold_sec = 2.0   # seconds WARNING must be held before upgrading
cooldown_sec     = 8.0   # seconds between repeated DROWSY alerts
```

---

## Citing This Work

Based on the methodology from:
- Guardian Alert (Thomas et al., IEEE RASSE 2023)
- Real Time Driver Fatigue Detection (Dey et al., RUET 2023)
- Spiking Vision AlertNet (Santhiya et al., ICESC 2025)
- Smart Driving with AI (Rabbi et al., Array Elsevier 2026)
