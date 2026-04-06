---
title: WEGH (OpenEnv)
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---
# 🔥 WEGH: Macro-Architectural CPU Designer

> An RL environment where AI agents learn to design CPU microarchitectures — from IoT microcontrollers to Apple M-series superscalar processors.

## What Is This? (Real-World Utility: 30% Evaluation Weight)

WEGH is a production-grade Electronic Design Automation (EDA) reinforcement learning environment built for the **Meta PyTorch OpenEnv Hackathon**.

**Real-Life Manufacturing & Research Application:**
In actual semiconductor fabrication and CPU research (at companies like Apple, Intel, and TSMC), architects battle highly complex, multi-objective constraints. Minor IPC improvements often explode thermal density limiters, rendering a chip unmanufacturable.
WEGH completely replaces traditional "toy" RL problems with actual industrial modeling. It allows frontier AI agents to tackle real-world physical boundaries by making genuine architectural decisions:
- *"Should I add more execution units or a bigger cache?"*
- *"How many P-cores vs E-cores balance performance and thermal?"*
- *"Will this ROB size cause an uncontrollable thermal hotspot over 65°C?"*

The environment models realistic CPU physiological limits — cache hit rates scale logarithmically against die area size, multi-component heat aggregation triggers thermal throttling down to 0 MHz, and pipeline logic accurately correlates Instruction-Per-Cycle yield metrics.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Docker Container (2 vCPU, 8GB RAM)          │
│                                                          │
│  Python/OpenEnv Server (port 7860)                      │
│  ├── FastAPI + Uvicorn (WebSocket + REST)                │
│  ├── WEGHEnvironment(Environment)                │
│  ├── Dense Reward Shaping [0.0, 1.0]                    │
│  └── Engineering Feedback for LLM in-context learning   │
│           │                                              │
│           │ httpx (localhost JSON)                       │
│           ▼                                              │
│  Go Simulation Engine (port 8080, internal only)        │
│  ├── DAG-based CPU component graph                      │
│  ├── Topological sort + cycle detection                 │
│  ├── Analytical PPA models (IPC, Power, Area, Thermal)  │
│  └── Per-episode state isolation                        │
└─────────────────────────────────────────────────────────┘
```

## Three Progressive Tasks

| Task | Difficulty | Focus | Constraint |
|---|---|---|---|
| **8-Bit IoT Microcontroller** | Easy | Power efficiency | < 50mW total power |
| **RV32IM 5-Stage Pipeline** | Medium | IPC vs Area | < 10mm² die area |
| **M-Series Heterogeneous Superscalar** | Hard | Thermal management | < 1.5 W/mm² power density |

## Quick Start

### Local Development

Our architecture uses purely native, dynamic child-processes. You do not need to boot the Go-engine manually.

```bash
# 1. Build Go engine natively (Optional: Python will fallback to path executing)
cd engine && go build -o ../go-engine ./cmd/server && cd ..

# 2. Install Python deps (UV is hackathon recommended)
uv pip install -r requirements.txt # Or via poetry
pip install openenv-core fastapi uvicorn httpx openai

# 3. Start environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t wegh .
docker run -p 7860:7860 wegh
```

### Run Inference

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your-token"
python inference.py
```

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Environment Server | Python + FastAPI | OpenEnv-compliant API |
| Simulation Engine | Go 1.22 | DAG validation + PPA heuristics |
| IPC Protocol | HTTP/JSON (localhost) | Zero-latency Python↔Go bridge |
| Packaging | Docker multi-stage | 2 vCPU / 8GB RAM compliant |
| RL Framework | OpenEnv (Meta) | Gymnasium-style step/reset/state |

## License

MIT
