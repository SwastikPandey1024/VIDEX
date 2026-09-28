# VIDEX — Video Intelligence

> Perception · Tracking · OCR · ASR · Semantic Reasoning · Event Detection

VIDEX is a modular video intelligence pipeline designed for structured analysis of video content. It ingests raw video, applies perception and tracking layers, extracts text and audio, and produces structured evidence records that feed a semantic event engine and reporting UI.

---

## Status

**Phase 0 — Scaffolding complete.**  
Domain contracts, provider interfaces, and API shell are in place.  
No ML model inference is implemented yet.

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Foundation, schemas, provider interfaces, API shell | ✅ Complete |
| 1 | Video ingestion, frame extraction, object detection | 🔜 Planned |
| 2 | Tracking, trajectory analysis | 🔜 Planned |
| 3 | OCR (Hindi), ASR (Hindi) | 🔜 Planned |
| 4 | Semantic reasoning / VLM | 🔜 Planned |
| 5 | Event engine, evidence store, reporting UI | 🔜 Planned |

---

## Architecture

```
Video Input
    │
    ▼
Ingestion ──────► Frame Store
    │
    ▼
Perception (DetectionProvider)
    │
    ▼
Tracking (TrackingProvider)
    │
    ▼
Temporal Analysis
    │
    ├──► OCR (OCRProvider)
    ├──► ASR (ASRProvider)
    └──► VLM (VideoReasoningProvider)
    │
    ▼
Evidence Store
    │
    ▼
Event Engine (EventDetector)
    │
    ▼
Intelligence UI / Reports
```

See [`docs/architecture/system-design.md`](docs/architecture/system-design.md) for the full design.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Git

### Setup

```bash
# Clone
git clone <repo-url>
cd VIDEX

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install (dev mode)
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env as needed

# Run API
uvicorn videx.api.main:app --reload

# Check health
curl http://localhost:8000/health
```

### Tests

```bash
pytest                       # all tests
pytest tests/unit/           # unit only
pytest tests/integration/    # integration only
```

### Lint / Type Check

```bash
ruff check src/ tests/       # lint
ruff format src/ tests/      # format
mypy src/                    # type check
```

---

## Project Structure

```
VIDEX/
├── src/
│   └── videx/
│       ├── config.py          ← Pydantic settings (single source of truth)
│       ├── domain/
│       │   └── schemas.py     ← All domain models (Video, Frame, Detection, Evidence…)
│       ├── providers/
│       │   └── base.py        ← Protocol interfaces for all AI providers
│       └── api/
│           ├── main.py        ← FastAPI application factory
│           └── routers/
│               └── health.py  ← GET /health
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
│   ├── architecture/
│   └── decisions/             ← ADRs (Architecture Decision Records)
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Architecture Decision Records

| ADR | Decision | Status |
|-----|----------|--------|
| [ADR-001](docs/decisions/ADR-001-detector.md) | Object detector selection | Proposed |
| [ADR-002](docs/decisions/ADR-002-tracker.md) | Multi-object tracker selection | Proposed |
| [ADR-003](docs/decisions/ADR-003-ocr.md) | Hindi OCR engine selection | Proposed |
| [ADR-004](docs/decisions/ADR-004-asr.md) | Hindi ASR engine selection | Proposed |

---

## Design Principles

1. **Provider-agnostic** — every AI component is accessed through a Protocol interface. Swap models without touching business logic.
2. **Schema-first** — the `Evidence` schema is the contract between all pipeline stages.
3. **No hardcoded secrets** — all configuration through `.env` / environment variables.
4. **Typed throughout** — Pydantic for validation, mypy strict mode.
5. **Testable without ML** — all tests in Phase 0 run without model weights or GPU.

---

## License

MIT
