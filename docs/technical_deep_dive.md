# WEGH: The Technical Deep-Dive

This document breaks down the underlying architecture, frameworks, and deployment strategies we used to pass Phase 2 of the Meta PyTorch OpenEnv Hackathon.

## System Architecture

Our project is a hybrid-architecture system split into two core layers:

1. **The Native Physics / DAG Engine (Go / Golang):**
   CPU architecture simulation is mathematically intense. Calculating Directed Acyclic Graph (DAG) state mutations across dozens of microarchitectural components requires speed. By writing this layer in Go, we achieve lightning-fast throughput. The Go binary spins up as a background daemon process and calculates all PPA (Power, Performance, Area) physics, bounds, and thermal hotspots.

2. **The OpenEnv Server Wrapper (Python / FastAPI):**
   To comply with the Hackathon's strict `openenv-core` framework requirements, we wrapped our high-speed Go daemon inside a Python FastAPI application (`server/app.py`). The Python server receives standardized `CPUAction` requests over WebSockets from external AI Agents, forwards the mutated state to the Go Daemon, and then relays the resulting `CPUObservation` state back to the AI.

## The OpenEnv Compliance & Networking
Getting this environment to successfully pass automated testing required a masterclass in networking compliance:

- **Client/Server Decoupling:** We refactored `inference.py` to act purely as an external WebSocket client (`WEGHEnv`). It completely removes all server-side imports, enforcing strict isolation. This ensures that when the remote evaluator runs the script in its test harnesses, `inference.py` acts natively without causing class import conflicts.
- **Port Management:** We configured the system to launch natively on `0.0.0.0:8000`. By shifting away from hardcoded ports in our Dockerfile `ENTRYPOINT` and switching to an overridable `CMD` instruction, we allowed the Scaler Automated Evaluator to spin up our image locally without triggering `[Errno 98] Address already in use`.
- **Score Formatting Bounds:** Our `wegh_graders.py` evaluation hook clamps final score outcomes firmly within `[0.001, 0.999]`. Since the evaluator mathematically extracts scores via string precision (`:.3f`), this ensures we never emit a strict `0.000` or `1.000`, guaranteeing we always pass the "strictly between 0 and 1" evaluation regex.

## The Three Graded Workloads
The AI agents are benchmarked across three configurations handled natively in `task_configs.py`:
1. `iot_8bit` (20 Max Steps) — Heavily weights Power (<50mW) and Area.
2. `rv32im` (30 Max Steps) — Balances Area (<10mm²) against achieving an IPC > 1.0 using standard 5-stage pipelines.
3. `mseries_superscalar` (40 Max Steps) — An advanced payload evaluating Heterogeneous designs. Introduces heavy thermal throttling models where exceeding 1.5 W/mm² directly damages cycle throughput.

## Deployment Stack
- **Dependencies:** Managed completely by `uv` inside the container for blazing-fast builds. Relying on `openenv-core[core]>=0.2.2`.
- **Hosting:** Successfully continuously deployed as a multi-stage Docker build to a live Hugging Face Space URL. It exposes all required 13 routes locally and remotely, including `/ws`, `/reset`, and `/health`.
