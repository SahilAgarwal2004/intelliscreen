# IntelliScreen — Context-Aware AI Interview & Proctoring Platform

IntelliScreen is an explainable, multimodal AI system engineered for technical interview assessment and proctoring. Unlike legacy proctoring tools that flag every brief glance away as cheating, IntelliScreen reasons across:

$$\mathbf{Behavior} + \mathbf{Time} + \mathbf{Context} + \mathbf{Multiple\ Signals}$$

---

## Key Highlights

- **Perception–Context Architecture:** Decouples raw perception (faces, gaze, objects) from higher-level behavioral and temporal reasoning.
- **Explainable Suspicion Scoring:** Produces human-auditable evidence timelines with confidence ratings and duration metrics rather than opaque binary flags.
- **Context-Aware Evaluation:** Integrates question complexity, candidate response latency, and technical concept recall with behavioral observation.
- **Strict Data Privacy:** Ephemeral frame processing; structured events and numerical metrics are recorded without storing raw video by default.

---

## High-Level Architecture

```
VIDEO STREAM ──► Preprocessing ──► Vision Perception (Face, Gaze, Objects)
                                               │
                                               ▼
                                    Structured Observations
                                               │
                                               ▼
                                    Behavior Event Extraction
                                               │
                                               ▼
                                    Temporal Context Engine
                                               │
AUDIO STREAM ──► Speech-to-Text ──► Semantic Answer Evaluation
                                               │
                                               ▼
                                   Multimodal Context Fusion
                                               │
                                               ▼
                                    Suspicion Scoring Engine
                                               │
                                               ▼
                                     Explainable Audit Log
```

---

## Repository Structure

```
IntelliScreen/
├── README.md                 # Project documentation
└── services/                 # AI service, perception pipelines, tests & tools
    ├── .env.example          # Configurable thresholds template
    ├── .gitignore            # Git exclusion rules
    ├── pyproject.toml        # Python packaging and tool configuration
    ├── models/               # Cached model weights (YuNet, MediaPipe, YOLO)
    ├── scripts/              # Verification & smoke test scripts
    ├── src/
    │   └── intelliscreen/
    │       ├── config/       # Pydantic Settings & threshold management
    │       ├── core/         # Base schemas, exceptions, and structured logger
    │       ├── vision/       # Computer vision pipelines (Face, Landmarks, Gaze, Objects)
    │       └── temporal/     # Temporal event engine & sliding window buffer
    └── tests/
        ├── conftest.py       # Shared fixtures
        ├── fixtures/         # Sample media fixtures & annotated demo artifacts
        └── unit/             # Isolated module tests
```

---

## Quickstart

### 1. Environment Setup
```bash
cd services
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configuration
```bash
cd services
cp .env.example .env
```

### 3. Run Test Suite
```bash
cd services
pytest
```
