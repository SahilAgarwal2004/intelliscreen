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
├── .env.example              # Configurable thresholds template
├── pyproject.toml            # Python packaging and tool configuration
├── README.md                 # Project documentation
├── src/
│   └── intelliscreen/
│       ├── config/           # Pydantic Settings & threshold management
│       ├── core/             # Base schemas, exceptions, and structured logger
│       ├── vision/           # Computer vision pipelines (Face, Landmarks, Gaze, Objects)
│       ├── behavior/         # Discrete state machine & temporal context engine
│       ├── audio/            # Speech-to-text integration
│       ├── nlp/              # Sentence embeddings & technical concept evaluator
│       ├── fusion/           # Multimodal feature fusion & suspicion scorer
│       └── api/              # FastAPI service & session management
└── tests/
    ├── conftest.py           # Shared fixtures
    ├── unit/                 # Isolated module tests
    └── integration/          # End-to-end pipeline tests
```

---

## Quickstart

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configuration
```bash
cp .env.example .env
```

### 3. Run Test Suite
```bash
pytest
```
