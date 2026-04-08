---
title: WEGH (OpenEnv)
emoji: 🔥
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - cpu-design
  - reinforcement-learning
  - hardware-architecture
---

# 🔥 WEGH: Macro-Architectural CPU Designer

> An RL environment where AI agents learn to design CPU microarchitectures — from IoT microcontrollers to Apple M-series superscalar processors.

**Built on the [OpenEnv framework](https://github.com/meta-pytorch/OpenEnv)** for the Meta PyTorch OpenEnv Hackathon. This implementation provides:

- **Go-native simulation engine** with DAG-based CPU component modeling, topological validation, and analytical PPA heuristics — all executing in <50μs per evaluation
- Text observations with dense engineering feedback, compatible with any LLM
- 3 progressive tasks spanning the full CPU complexity spectrum
- Dense reward shaping with constraint-aware scoring for RL training

## What Makes This Different

Traditional RL benchmarks (CartPole, Atari) test abstract decision-making. WEGH tests whether AI can make **real engineering trade-offs** in a domain that costs semiconductor companies billions:

- **Decisions have compounding consequences**: Adding execution units improves IPC but increases power density. If the thermal budget is exceeded, clock throttling negates the performance gain entirely
- **Multi-objective optimization is mandatory**: There is no single "good" move — every action trades off Performance, Power, and Area (PPA). The agent must learn Pareto-optimal design
- **Domain constraints are physical**: Cache hit rates follow logarithmic scaling against die area. Power scales with V²f. Thermal density causes real throttling. These aren't arbitrary penalties — they're the same equations real chip architects use at Apple, Intel, and TSMC

## Quick Start

```python
from models import CPUAction, CPUObservation
from server.wegh_env import WEGHEnvironment

env = WEGHEnvironment()
obs = env.reset(task="iot_8bit")

print(f"Task: {obs.task_name}")
print(f"IPC: {obs.current_estimated_ipc:.3f}")
print(f"Power: {obs.total_power_mw:.1f} mW")
print(f"Feedback: {obs.feedback_string}")

# Make an architectural decision
action = CPUAction(
    action_type="resize",
    target_component="l1d",
    parameter_name="size_kb",
    parameter_value=32.0,
    reasoning="Increase L1D cache for better hit rate"
)
obs = env.step(action)
print(f"Reward: {obs.reward:+.4f} | Done: {obs.done}")
```

**No local Go build needed** — the pre-compiled binary is included for local dev. For Docker deployment, the Go engine is built from source in-container.

### Run Inference (LLM Agent)

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your-token"
python inference.py
```

The inference script runs one episode per task (`iot_8bit` → `rv32im` → `mseries_superscalar`), printing evaluator-compliant stdout:

```
[START] task=8-Bit IoT Microcontroller env=WEGH model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action={"action_type":"resize",...} reward=0.04 done=false error=null
...
[END] success=true steps=20 rewards=0.04,0.02,...
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│              Docker Container (2 vCPU, 8GB RAM)           │
│                                                           │
│  Python/OpenEnv Server (port 8000)                       │
│  ├── FastAPI + Uvicorn (WebSocket + REST)                 │
│  ├── WEGHEnvironment(Environment)                        │
│  │   ├── sanitize_grade() — Phase 2 firewall             │
│  │   ├── Dense Reward Shaping (per-step + final)         │
│  │   └── LLM Engineering Feedback Generator              │
│  └── atexit + __del__ daemon lifecycle hooks              │
│           │                                               │
│           │ httpx (localhost JSON-over-HTTP)               │
│           ▼                                               │
│  Go Simulation Engine (dynamic port, internal only)      │
│  ├── DAG-based CPU component graph                       │
│  │   ├── Kahn's algorithm topological sort               │
│  │   └── Cycle detection on every edge mutation          │
│  ├── Analytical PPA Models                               │
│  │   ├── IPC: pipeline-aware with branch/cache/OoO stalls│
│  │   ├── Power: CMOS P=αCV²f with per-component breakdown│
│  │   ├── Area: 7nm process node scaling                  │
│  │   └── Thermal: junction temp + power density hotspots │
│  └── Per-episode state isolation (mutex-protected)       │
└──────────────────────────────────────────────────────────┘
```

### Hybrid Python↔Go Design

The performance-critical simulation (DAG mutations, topological validation, PPA evaluation) runs in a native Go daemon for sub-millisecond latency. The Python wrapper handles OpenEnv protocol compliance, reward shaping, and LLM feedback formatting. Communication is JSON-over-HTTP on a dynamically acquired localhost port — zero network overhead, zero serialization complexity.

## Three Progressive Tasks

| Task | Difficulty | Steps | Primary Constraint | Design Challenge |
|---|---|---|---|---|
| **8-Bit IoT Microcontroller** | Easy | 20 | Power < 50mW | Strip components for ultra-low power |
| **RV32IM 5-Stage Pipeline** | Medium | 30 | Area < 10mm² | Maximize IPC within die area budget |
| **M-Series Superscalar** | Hard | 40 | Power Density < 1.5 W/mm² | Balance P-cores vs E-cores without thermal throttling |

Each task presents fundamentally different trade-off surfaces:

### Task 0: IoT — `iot_8bit`

Battery-powered sensor with no active cooling. The agent must learn that *removing* components is often better than adding them. Initial design has an ALU, SRAM, GPIO, and PMU. Performance is secondary — survival on a coin cell battery is the goal.

**Scoring**: 60% power compliance + 40% throughput (IPC / 0.8)

### Task 1: RV32IM — `rv32im`

5-stage pipelined RISC-V core for cost-sensitive embedded. The classic IPC-vs-area trade-off: bigger caches improve hit rates but consume die area logarithmically. The agent must decide cache sizes, branch predictor complexity, and whether forwarding paths justify their area cost.

**Scoring**: 50% IPC + 40% area compliance + 10% performance-per-watt

### Task 2: M-Series — `mseries_superscalar`

Apple M-series inspired heterogeneous design. The thermal wall is the hard constraint — exceeding 1.5 W/mm² power density triggers throttling that *halves* the effective clock frequency. The agent must balance P-core burst performance against E-core sustained efficiency, ROB/RS sizing for out-of-order execution, and cache hierarchy depth.

**Scoring**: 35% throughput + 25% efficiency + 25% thermal compliance + 15% throttle factor

## Actions

The agent modifies the CPU design through 5 action types:

### Component Management

```python
# Add a new component to the microarchitecture
CPUAction(action_type="add_component", target_component="l2", reasoning="Add L2 cache")

# Remove an unnecessary component (severs all connections)
CPUAction(action_type="remove_component", target_component="fpu", reasoning="IoT doesn't need FPU")
```

### Parameter Tuning

```python
# Resize a component parameter (values are clamped to valid bounds)
CPUAction(action_type="resize", target_component="l1d", parameter_name="size_kb",
          parameter_value=64.0, reasoning="Increase L1D for better hit rate")

# Configure non-size parameters
CPUAction(action_type="configure", target_component="bp", parameter_name="type",
          parameter_value=3.0, reasoning="Upgrade to TAGE predictor")
```

### Connectivity

```python
# Connect two components in the datapath (cycle-checked by Go engine)
CPUAction(action_type="connect", source_node="l1d", target_node="l2",
          reasoning="Connect L1D miss path to L2")
```

### Available Components

| Component | ID | Key Parameters |
|---|---|---|
| Integer ALU | `alu` / `p_alu` | `count` |
| Multiply/Divide | `muldiv` / `p_muldiv` | `count` |
| FP Unit | `p_fpu` | `count` |
| SIMD Unit | `p_simd` | `count`, `width_bits` |
| Load/Store Units | `load_unit`, `store_unit` | `count` |
| Branch Predictor | `bp` | `type` (0=static, 1=bimodal, 2=gshare, 3=TAGE), `btb_entries`, `bht_size` |
| Reorder Buffer | `rob` | `entries` |
| Reservation Station | `rs` | `entries` |
| L1 I-Cache / D-Cache | `l1i`, `l1d` | `size_kb`, `associativity` |
| L2 Cache | `l2` | `size_kb`, `associativity` |
| L3 Cache | `l3` | `size_mb`, `associativity` |
| P-Core / E-Core | `pcore`, `ecore` | `count`, `pipeline_depth`, `issue_width`, `clock_ghz`, `voltage` |
| Memory Controller | `memctrl` | `channels`, `bandwidth_gbps` |
| NoC Interconnect | `noc` | `type` (0=bus, 1=ring, 2=mesh, 3=crossbar), `bandwidth_gbps` |
| TLB | `tlb` | `entries` |
| Prefetcher | `pf` | `type` (0=none, 1=stride, 2=stream, 3=multi) |
| PMU | `pmu` | `voltage`, `clock_mhz` / `clock_ghz` |
| SRAM (IoT) | `sram` | `size_kb` |
| GPIO (IoT) | `gpio` | `pin_count` |

## Reward System

WEGH uses a two-tier reward system designed for RL training effectiveness:

### Per-Step Dense Rewards

Step rewards are **not clamped** — they intentionally fluctuate between approximately -0.3 and +0.15 to provide a meaningful gradient signal. The reward function evaluates:

| Signal | Reward Range | Description |
|---|---|---|
| Valid action | +0.02 | Bonus for structurally valid modifications |
| Invalid action | -0.15 | Penalty for rejected operations |
| Validation errors | -0.05 per error | DAG structural violations |
| Constraint compliance | +0.02 to +0.04 | Proportional to headroom below limits |
| Constraint violation | -0.06 to -0.20 | Proportional to overshoot (clamped) |
| IPC / throughput improvement | +0.05 | Relative to previous step |
| Power density reduction | +0.04 | Rewarding thermal cooling moves |
| Throttle penalty | -0.03 | Proportional to throttle severity |
| Pareto improvement (≥3 axes) | +0.05 | Multi-objective bonus |

### Final Episode Scores

Final scores are bounded to **(0.0, 1.0) exclusive** and pass through `sanitize_grade()` to prevent Phase 2 evaluator crashes:

- Scores ≥ 1.0 → clamped to 0.999
- Scores ≤ 0.0 → clamped to 0.001
- None / NaN / numpy types → cast to Python `float`, fallback 0.001
- String artifacts → `float()` cast, fallback 0.001

Each task has a weighted scoring formula (see task descriptions above) that combines PPA metrics into a single normalized score.

## Go Simulation Engine

The Go engine implements analytical heuristic models — parameterized equations that capture the directional trade-offs real chip architects face, executing in <50μs per evaluation.

### IPC Model

Pipeline-aware IPC computation considering:
- **Issue width saturation**: IPC ≤ min(issue_width, total_execution_units)
- **Branch stalls**: Misprediction rate × pipeline depth × flush cost. Predictor types: static (20%), bimodal (12%), gshare (7%), TAGE (3%), perceptron (2.5%). BTB/BHT entries reduce rates logarithmically
- **Cache stalls**: Multi-level miss cascade (L1 → L2 → L3 → DRAM) with configurable latencies (10/30/200 cycles). Prefetcher reduces stalls by 20-40%
- **Structural stalls**: Execution unit saturation when issue width exceeds available units
- **OoO bonus**: ROB/RS sizing provides 0-30% IPC improvement based on pipeline coverage

### Power Model (CMOS)

Standard P = α × C × V² × f model with:
- Per-component activity factors and capacitance estimates
- Static leakage proportional to capacitance × voltage
- IoT uses milliwatt-scale simplified model
- M-Series computes P-core and E-core power independently with separate voltage domains

### Area Model (7nm)

Component-level area estimation at 7nm process node:
- Cache area scales with size × associativity bonus
- Execution units: ALU (0.12mm²), MulDiv (0.25mm²), FPU (0.35mm²), SIMD (0.5mm² × width_scale)
- OoO structures: ROB/RS at 0.001mm² per entry
- NoC: bus (0.5mm²) → ring (1.0) → mesh (2.0) → crossbar (4.0)

### Thermal Model

- **Junction temperature**: T_j = 35°C + θ_JA × P_total (θ_JA = 80°C/W for IoT, 20°C/W for active cooling)
- **Power density hotspots**: Per-component W/mm² tracking
- **Throttling**: Triggered at T_j > 95°C (aggressive) or PD > 1.5× safe limit (localized). Throttling factor ranges from 0.3 to 1.0 and directly multiplies effective clock frequency

## Deployment

### Hugging Face Spaces (Docker)

```bash
# Push to HF Space
git push  # Dockerfile is auto-detected by HF Spaces
```

### Local Docker

```bash
docker build -t wegh .
docker run -p 8000:8000 wegh
```

### Local Development (No Docker)

```bash
# 1. Build Go engine
cd engine && go build -o ../go-engine ./cmd/server && cd ..

# 2. Install Python dependencies
uv pip install openenv-core fastapi uvicorn httpx openai

# 3. Start the server (Go daemon spawns automatically)
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Specifications

| | Value |
|---|---|
| CPU | 2 vCPU (HF Spaces limit) |
| Memory | 8GB RAM |
| GPU | Not required |
| Go Engine | Go 1.22, compiled to static binary |
| Python | 3.12 with FastAPI + Uvicorn |
| Image Size | ~500MB (multi-stage Docker) |
| Startup Time | ~5 seconds |

### Configuration

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `meta-llama/Llama-3.1-8B-Instruct` | Model for inference agent |
| `HF_TOKEN` | *(required)* | Hugging Face API token |

## Phase 2 Evaluation Protocol

The Phase 2 automated evaluator reads `openenv.yaml` to boot the FastAPI server, then evaluates via REST:

1. **`openenv.yaml`** declares `spec_version: 1`, `runtime: fastapi`, `app: server.app:app`, `port: 8000`, and 3 task definitions
2. For each task, the evaluator calls `reset(task=<task_id>)` → runs step/observe loop → reads `done=True`
3. The final observation is serialized as a JSON dict (not a Pydantic object) and passed to the grader hook
4. Grader hooks (`wegh_graders:grade_<task>`) extract `final_score` from the dict and return a sanitized float in (0.0, 1.0) exclusive

### Known Evaluator Fragility Mitigations

| Issue | Mitigation |
|---|---|
| numpy floats crash the judge | `sanitize_grade()` casts all scores to Python `float()` |
| score=1.0 or 0.0 crashes | Exclusive boundary clamping: 0.999 / 0.001 |
| Observations arrive as dicts | Graders use `isinstance(obs, dict)` check before extraction |
| `score=` in [END] line | Removed per Scaler guideline — only `success`, `steps`, `rewards` |
| Zombie Go processes | `atexit.register()` + `__del__` hooks terminate daemon on Python exit |
| Port conflicts | `get_free_port()` acquires dynamic localhost port per instance |

## Testing

### Phase 2 Evaluator Simulation

A comprehensive test suite simulates the exact Phase 2 evaluation flow:

```bash
.venv/bin/python3 tests/phase2_evaluator_sim.py
```

This validates:
- `openenv.yaml` schema compliance
- Grader hook imports and boundary behavior
- FastAPI server boot + health check
- Full episode lifecycle (reset → step → grade) for all 3 tasks
- `inference.py` stdout regex compliance
- Dockerfile completeness
- Resource cleanup verification

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Environment Server | Python 3.12 + FastAPI | OpenEnv-compliant API |
| Simulation Engine | Go 1.22 (static binary) | DAG validation + PPA heuristics |
| IPC Protocol | HTTP/JSON (localhost) | Zero-latency Python↔Go bridge |
| Packaging | Docker multi-stage | 2 vCPU / 8GB RAM compliant |
| RL Framework | OpenEnv (Meta) | Gymnasium-style step/reset/state |
| Type Safety | Pydantic v2 | Action/Observation schema enforcement |

## File Structure

```
WEGH/
├── openenv.yaml            # Phase 2 evaluator config
├── inference.py            # Local LLM inference loop
├── models.py               # Pydantic Action/Observation/State models
├── wegh_graders.py         # Phase 2 grader hooks (sanitized scoring)
├── client.py               # Remote client (EnvClient subclass)
├── Dockerfile              # Multi-stage: Go build + Python runtime
├── entrypoint.sh           # Container entry point
├── pyproject.toml           # Python dependencies
├── server/
│   ├── app.py              # FastAPI app factory
│   ├── wegh_env.py         # Core environment (Environment subclass)
│   ├── go_client.py        # HTTP client for Go daemon
│   ├── reward.py           # Dense reward shaping + final scoring
│   └── task_configs.py     # 3 task definitions with constraints
├── engine/                 # Go simulation engine source
│   ├── cmd/server/main.go  # Go HTTP server entry point
│   ├── api/
│   │   ├── handlers.go     # REST handlers (reset/step/health)
│   │   └── models.go       # Go request/response structs
│   └── internal/
│       ├── graph/
│       │   ├── dag.go      # DAG with Kahn's toposort + cycle detection
│       │   └── templates.go # Task-specific initial graph templates
│       ├── simulator/
│       │   └── simulator.go # Analytical PPA models (IPC/Power/Area/Thermal)
│       └── episodes/
│           └── manager.go  # Per-episode graph isolation (mutex-protected)
└── tests/
    └── phase2_evaluator_sim.py  # Full evaluator simulation
```

## Limitations

- **Analytical models only**: PPA heuristics are parameterized equations, not cycle-accurate simulation. Results indicate directional trade-offs, not absolute chip performance
- **Fixed working set**: Cache miss models assume a 64KB working set. Real workload characterization would require trace-driven simulation
- **Static graph structure**: Initial component topology is fixed per task. The agent can add/remove/resize but cannot fundamentally restructure the pipeline
- **Single-agent**: No multi-agent or population-based optimization
- **No physical layout**: Area model doesn't account for floorplanning, wire delay, or placement constraints

## Resources

- **OpenEnv Framework**: [github.com/meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)
- **Meta PyTorch Hackathon**: [pytorch.org](https://pytorch.org/)
- **Computer Architecture References**:
  - Hennessy & Patterson, *Computer Architecture: A Quantitative Approach*
  - Weste & Harris, *CMOS VLSI Design*
  - Apple M-series architecture deep dives (AnandTech, Chips and Cheese)

## License

MIT
