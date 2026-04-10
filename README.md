---
title: WEGH Architecture
emoji: 🖥️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# 🔬 WEGH — Workload Evaluation for Generative Hardware

[![OpenEnv](https://img.shields.io/badge/OpenEnv-0.2.3-blue.svg)](https://github.com/openenv)
[![Go](https://img.shields.io/badge/Go-1.22-00ADD8.svg)](https://go.dev)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**WEGH** is a reinforcement learning environment where AI agents learn to design CPU microarchitectures — from IoT microcontrollers to Apple M-series superscalar processors. Built for the **Meta PyTorch OpenEnv Hackathon**.

> **Live Space**: `https://<your-space>.hf.space`

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Container                      │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │          Go Simulation Engine (port 7860)         │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │   │
│  │  │ Pipeline  │ │  Cache   │ │  PPA Simulator   │ │   │
│  │  │ Simulator │ │ Hierarch │ │  (Power/Area/    │ │   │
│  │  │ (Cycle-   │ │ (MESI    │ │   Thermal)       │ │   │
│  │  │ Accurate) │ │ Protocol)│ │                  │ │   │
│  │  └──────────┘ └──────────┘ └──────────────────┘ │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │   │
│  │  │  Episode  │ │   DAG    │ │  Grading + Sub-  │ │   │
│  │  │  Manager  │ │ Topology │ │  Scores + Exploit│ │   │
│  │  └──────────┘ └──────────┘ └──────────────────┘ │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │         Python Inference (inference.py)            │   │
│  │  LLM Agent + Heuristic Fallback + Reward Norm     │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

The Go engine is a **self-contained OpenEnv server** — it serves all standard HTTP endpoints directly without any Python framework dependency. The inference script connects to it via HTTP using direct `httpx` calls.

## API Endpoints

| Method | Path       | Description                                     |
|--------|------------|------------------------------------------------|
| `GET`  | `/health`  | Health check: `{"status": "ok"}`               |
| `GET`  | `/ping`    | Liveness probe                                  |
| `POST` | `/reset`   | Start new episode: `{"task": "rv32im", "seed": 42}` |
| `POST` | `/step`    | Execute action: `{"action": {...}}`             |
| `GET`  | `/state`   | Current environment state                       |
| `GET`  | `/grade`   | Episode grade with sub-scores + exploit detection |
| `GET`  | `/tasks`   | Available tasks (3 progressive difficulties)    |
| `GET`  | `/metrics` | Prometheus-format performance metrics           |
| `GET`  | `/replay`  | Full episode replay data                        |

## Tasks

### Task 0: 8-Bit IoT Microcontroller (Easy, 20 steps)
Design a minimal 8-bit MCU for IoT sensors. **Power budget: 50mW.**

| Sub-score | Weight | Description |
|-----------|--------|-------------|
| Power     | 60%    | Power consumption vs 50mW budget |
| Throughput| 20%    | Performance (IPC) |
| Area      | 15%    | Die area efficiency |
| Thermal   | 5%     | Junction temperature |

### Task 1: RV32IM 5-Stage Pipeline (Medium, 30 steps)
Design a RISC-V RV32IM core. **Area budget: 10mm² at 7nm.**

| Sub-score | Weight | Description |
|-----------|--------|-------------|
| IPC       | 50%    | Instructions Per Clock |
| Area      | 30%    | Die area vs 10mm² budget |
| Power     | 15%    | Power consumption |
| Thermal   | 5%     | Junction temperature |

### Task 2: M-Series Heterogeneous Superscalar (Hard, 40 steps)
Design an Apple M-series inspired CPU. **Power density limit: 1.5 W/mm².**

| Sub-score  | Weight | Description |
|------------|--------|-------------|
| Throughput | 35%    | Multi-core throughput (GIPS) |
| Efficiency | 25%    | Performance per watt |
| Thermal    | 25%    | Power density + throttle factor |
| Area       | 15%    | Die area efficiency |

## Baseline Scores

| Task | Heuristic Policy | Score |
|------|-----------------|-------|
| IoT 8-bit | Low-voltage, minimal SRAM | *varies* |
| RV32IM | Cache optimization + 2-wide decode | *varies* |
| M-Series | 4P+4E, TAGE BP, 16MB L3 | *varies* |

Run `python inference.py --fast-mode --output baseline_scores.json` to generate current baselines.

## Quick Start

### Docker (recommended)
```bash
docker build -t wegh .
docker run -p 7860:7860 wegh
# Server ready at http://localhost:7860
```

### Local Development
```bash
# Build Go engine
cd engine && go build -o ../go-engine ./cmd/server && cd ..

# Start server
./go-engine

# Run inference (in another terminal)
python inference.py --env-url http://localhost:7860 --fast-mode
```

### Test with curl
```bash
# Health check
curl http://localhost:7860/health

# Reset an episode
curl -X POST http://localhost:7860/reset -H 'Content-Type: application/json' \
  -d '{"task": "rv32im", "seed": 42}'

# Take an action
curl -X POST http://localhost:7860/step -H 'Content-Type: application/json' \
  -d '{"action": {"type": "resize", "component": "l1d", "param_name": "size_kb", "value": 32}}'

# Get grade
curl http://localhost:7860/grade

# Get tasks
curl http://localhost:7860/tasks
```

## Inference Script

The agent supports three execution modes:

```bash
# Full LLM mode (requires API key)
export HF_TOKEN=your_key
python inference.py --env-url http://localhost:7860

# Hybrid mode: LLM every 3 steps, heuristic between
python inference.py --llm-every 3

# Fast heuristic-only mode (no API key needed)
python inference.py --fast-mode --output scores.json
```

## Project Structure

```
WEGH/
├── engine/                    # Go simulation engine
│   ├── cmd/server/main.go     #   Self-contained OpenEnv HTTP server
│   ├── api/
│   │   ├── handlers.go        #   All API endpoint handlers + grading
│   │   └── models.go          #   Request/response models
│   └── internal/
│       ├── cache/             #   Cache hierarchy simulator (MESI)
│       ├── episodes/          #   Episode state management
│       ├── graph/             #   DAG topology engine
│       ├── mathutil/          #   Numerical utilities
│       ├── pipeline/          #   Cycle-accurate pipeline simulator
│       └── simulator/         #   Hybrid PPA evaluation
├── server/
│   └── app.py                 #   Python entry point (Go binary launcher)
├── tests/
│   ├── test_graders.py        #   pytest: 20+ grading tests
│   └── validate.py            #   Pre-submission validator (50+ checks)
├── inference.py               #   LLM agent with heuristic fallback
├── wegh_graders.py            #   OpenEnv grader hooks
├── models.py                  #   Pydantic observation/action models
├── client.py                  #   OpenEnv client API
├── openenv.yaml               #   Full OpenEnv spec (~200 lines)
├── Dockerfile                 #   Multi-stage build (Go + Python)
├── entrypoint.sh              #   Container startup script
├── validate-submission.sh     #   4-step submission validator
└── pyproject.toml             #   Python project config
```

## Key Design Decisions

1. **Go-native server**: The simulation engine serves all OpenEnv endpoints directly in Go. No Python server framework needed at runtime. This eliminates the `openenv-core` dependency for the server, making the container faster and more reliable.

2. **Cycle-accurate IPC**: Rather than analytical approximations, IPC is computed via a cycle-accurate pipeline simulator (~80μs per evaluation). This produces realistic IPC numbers that respond to cache sizing, branch predictor type, and execution unit count.

3. **Deterministic grading**: The `/grade` endpoint computes scores from episode replay data, not from the agent's self-reported metrics. Sub-scores are computed independently for each optimization dimension, with exploit detection for degenerate strategies.

4. **Heuristic fallback**: The inference script includes rule-based policies for each task that produce non-trivial scores even without an LLM. This handles API failures, rate limits, and provides baselines for benchmarking.

## Grading System

Scores are bounded to the open interval (0, 1) — never exactly 0 or 1.

**Exploit detection** penalizes:
- Repeating the same action >60% of the time
- Consecutive identical actions >1/3 of episode length
- Maximum penalty cap: 30% score reduction

## Testing

```bash
# Run grader tests (requires server running)
pytest tests/test_graders.py -v

# Run pre-submission validator
python tests/validate.py --env-url http://localhost:7860

# Run submission validator
./validate-submission.sh https://your-space.hf.space
```

## License

MIT
