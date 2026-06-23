# Project Nexus ⚽

> **A full-stack football intelligence system** spanning real-time officiating AI, player biometrics, crowd safety, and an open-hardware haptic accessibility device for blind fans.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![Arduino](https://img.shields.io/badge/Arduino-Mega%202560-teal?logo=arduino)](https://arduino.cc)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Author](https://img.shields.io/badge/Author-Unwilling--mcu-black?logo=github)](https://github.com/Unwilling-mcu)

---

## Overview

Project Nexus is a five-module football technology stack that addresses gaps left by existing solutions like FIFA's Semi-Automated Offside Technology (SAOT). Built as a final-year B.Tech project at KIIT University, it combines computer vision, domain-adaptive machine learning, embedded firmware, and accessible hardware design into a single cohesive system.

```
Stadium cameras + Ball IMU + Jersey sensors
            │
     ┌──────▼──────┐
     │  RefLens AI  │  ← Foul / Dive / Handball / Offside detection
     └──────┬───────┘
            │ alerts
   ┌─────────┼──────────┐
   │         │          │
   ▼         ▼          ▼
Earpiece   VAR Booth  TouchField
(referee)  (3D clip)  (haptic device
                       for blind fans)
```

---

## The Five Modules

| # | Module | Status | Novelty |
|---|--------|--------|---------|
| 01 | **BioMesh** — Smart jersey sensor layer | Concept + spec | Fabric-integrated (not vest) for live officiating |
| 02 | **RefLens** — AI referee assistant | **Built** ✅ | DANN domain adaptation across stadium cameras |
| 03 | **NeuroGuard** — Head impact safety | Concept + spec | No FIFA heading-load protocol exists yet |
| 04 | **PitchSense** — Crowd intelligence | Concept + spec | Dual-signal (vision + audio) crowd safety |
| 05 | **TouchField** — Haptic accessibility device | **Built** ✅ | Open-hardware; no open version exists globally |

---

## Module 02 — RefLens

### What it does

RefLens detects five types of football incidents from video in under 100ms:

- **Offside** (extends SAOT with full-body pose)
- **Foul classification** (push / trip / stamp / elbow)
- **Dive / simulation detection**
- **Handball** (arm-position + ball-IMU corroboration)
- **Clean play** (no alert)

Alerts go to the referee's earpiece with a confidence score. Only predictions above a per-class threshold trigger alerts. The VAR booth receives a 3D wireframe clip for review.

### Architecture

```
Video frames (50fps)
      │
  PoseExtractor (YOLOv8-pose)
      │  17 keypoints × 3 (x, y, conf) per player
  PoseWindowBuffer
      │  rolling 2-second window (100 frames)
  TemporalEncoder (Dilated TCN)
      │  (batch, 51, 100) → (batch, 256) embedding
  ┌───┴────────────────────┐
  EventClassifier          DomainClassifier
  (5-class softmax)        (adversarial — DANN)
      │
  AlertDispatcher
  ├── earpiece alert (if conf ≥ threshold)
  ├── VAR queue (if conf ≥ 0.85 × threshold)
  └── broadcast WebSocket → TouchField
```

### Domain Adaptation (the research contribution)

Models trained on one stadium's camera angles fail to generalise to another due to viewpoint, lighting, and resolution differences. RefLens uses **Domain-Adversarial Neural Networks (DANN)** to solve this:

1. A `GradientReversal` layer forces the `TemporalEncoder` to produce features that fool the `DomainClassifier`
2. The encoder learns to ignore stadium-specific artefacts
3. Event classification performance is preserved across domains

This is directly connected to the **Transfer Learning / Domain Adaptation** research track at KIIT. A conference paper on cross-stadium foul detection is planned as a deliverable.

### Quick start

```bash
git clone https://github.com/Unwilling-mcu/ProjectNexus
cd ProjectNexus/ml

# Install dependencies
pip install torch torchvision ultralytics fastapi uvicorn websockets numpy

# Run demo (simulated 300 frames, no GPU needed)
python reflens_pipeline.py

# Output: demo_match_log.json with all flagged events
```

### Training (with your own data)

```python
from reflens_pipeline import RefLensModel, train_epoch
import torch

model     = RefLensModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(50):
    loss = train_epoch(model, optimizer,
                       source_loader,   # labelled home stadium data
                       target_loader,   # unlabelled target stadium data
                       epoch, total_epochs=50)

torch.save(model.state_dict(), "reflens_weights.pt")
```

---

## Module 05 — TouchField

### What it does

TouchField is an open-hardware haptic device that lets blind and visually impaired fans follow a live football match through touch:

- **20 × 12 pin grid** (240 solenoid pins) represents the pitch
- Pins rise/vibrate to show ball, home players, away players, and referee
- **Ball pin vibrates at 20Hz** — instantly distinguishable from static players
- **DFPlayer Mini audio module** narrates goals, fouls, cards in real time
- **BLE companion app** for smartphones (React Native, planned v1.1)

### Hardware BOM (v0 prototype)

| Component | Qty | Approx. cost |
|-----------|-----|-------------|
| Arduino Mega 2560 | 1 | ₹800 |
| TLC5940 16-ch PWM driver | 15 | ₹1,200 |
| 5V mini push-pull solenoids | 240 | ₹4,800 |
| IRF540N N-MOSFET (solenoid drive) | 15 | ₹150 |
| DFPlayer Mini + microSD | 1 | ₹200 |
| HC-05 BLE module | 1 | ₹120 |
| 12V 5A switching PSU | 1 | ₹350 |
| **Total (prototype)** | | **≈ ₹7,620** |

### System diagram

```
Stadium SAOT feed (WebSocket)
        │
  touchfield_bridge.py
  ├── GridRenderer  (105m×68m → 20×12)
  ├── SerialTransmitter  (115200 baud)
  └── WS server (companion app, port 8766)
        │ serial JSON @ 30fps
  Arduino Mega 2560
  ├── TLC5940 chain (15 ICs, 240 channels)
  ├── 240 solenoid pins
  ├── DFPlayer Mini (audio narration)
  └── HC-05 (BLE companion app)
```

### Quick start (demo — no Arduino needed)

```bash
cd ProjectNexus/arduino

# Install bridge dependencies
pip install pyserial websockets numpy

# Run in demo mode (synthetic match simulation, prints to stdout)
python touchfield_bridge.py --demo

# Run with real Arduino on Linux
python touchfield_bridge.py --port /dev/ttyUSB0 --demo

# Run with real Arduino + live SAOT feed
python touchfield_bridge.py --port /dev/ttyUSB0 --ws ws://192.168.1.100:8765
```

### Arduino firmware

Flash `touchfield_firmware.ino` to an Arduino Mega 2560 using the Arduino IDE.

Required libraries (install via Library Manager):
- `ArduinoJson` v6
- `SoftwareSerial` (built-in)

---

## Repository Structure

```
ProjectNexus/
├── ml/
│   ├── reflens_pipeline.py      # Full RefLens ML pipeline + DANN
│   ├── demo_match_log.json      # Output from demo run
│   └── requirements.txt
├── arduino/
│   ├── touchfield_firmware.ino  # Arduino Mega firmware
│   ├── touchfield_bridge.py     # Python data bridge
│   └── requirements.txt
├── docs/
│   └── nexus_system_design.docx # Full architecture document
└── README.md
```

---

## Research Contributions

1. **Cross-stadium foul detection via DANN** — applying domain-adversarial training to football pose sequences. Target venue: CVPR 2026 workshop on sports AI.
2. **Open-hardware haptic sports device** — reproducible design for football accessibility. Target venue: CHI 2026 accessibility track.

---

## Roadmap

- [x] RefLens core pipeline (TCN + DANN architecture)
- [x] TouchField firmware (Arduino Mega + TLC5940)
- [x] TouchField data bridge (Python WebSocket → Serial)
- [ ] RefLens training dataset (collecting match footage)
- [ ] TouchField PCB v1 (KiCad design, solenoid array)
- [ ] NeuroGuard mouthguard firmware (ESP32 + ICM-42688)
- [ ] PitchSense crowd dashboard (React + YOLOv8 crowd detector)
- [ ] Companion mobile app (React Native, BLE)
- [ ] System integration test at KIIT campus stadium

---

## Author

**Sanchayan** · B.Tech Information Technology · KIIT University · Batch 2023–2027

GitHub: [@Unwilling-mcu](https://github.com/Unwilling-mcu)

---

## License

MIT License — see [LICENSE](LICENSE). Hardware design files released under CERN-OHL-P v2.
