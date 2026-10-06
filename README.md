# VIDEX — Video Intelligence

> Perception · Tracking · OCR · ASR · Semantic Reasoning · Event Detection

VIDEX is a modular video intelligence pipeline designed for structured analysis of video content. It ingests raw video, applies perception and tracking layers, extracts text and audio, and produces structured evidence records that feed a semantic event engine and reporting UI.

---

## Status

**Phase 8 — Agentic Investigator complete.**  
Evidence-grounded spatiotemporal query reasoning engine, 16 typed read-only tools, and claim validation.

| Phase | Description | Status |
| --- | --- | --- |
| Phase 0 | Foundation | ✅ |
| Phase 1 | Ingestion | ✅ |
| Phase 2 | Perception | ✅ |
| Phase 3 | OCR | ✅ |
| Phase 4 | Audio / ASR | ✅ |
| Phase 5 | Temporal Events | ✅ |
| Phase 6 | Semantic Intelligence | ✅ |
| Phase 7 | Evidence Graph | ✅ |
| Phase 8 | Agentic Investigator | ✅ |
| Phase 9 | Investigation UI | 🔜 |

---

## Architecture

```text
Video
 ↓
Ingestion
 ↓
Perception / Tracking / OCR / ASR
 ↓
Temporal Events
 ↓
Semantic Router / VLM
 ↓
Canonical Evidence
 ↓
Evidence Graph
 ↓
VIDEX Investigator
 ↓
FastAPI
 ↓
React Investigation Workstation
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
| --- | --- | --- |
| [ADR-001](docs/decisions/ADR-001-detector.md) | Object detector selection | Accepted |
| [ADR-002](docs/decisions/ADR-002-tracker.md) | Multi-object tracker selection | Accepted |
| [ADR-003](docs/decisions/ADR-003-ocr.md) | Hindi OCR engine selection | Accepted |
| [ADR-004](docs/decisions/ADR-004-asr.md) | Hindi ASR engine selection | Accepted |
| [ADR-005](docs/decisions/ADR-005-semantic-boundary-and-vlm-routing.md) | Semantic boundary and VLM routing | Accepted |
| [ADR-006](docs/decisions/ADR-006-evidence-graph-and-spatiotemporal-memory.md) | Evidence graph and spatiotemporal memory | Accepted |

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
