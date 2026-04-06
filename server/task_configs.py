# WEGH — Task Configurations
# Defines the 3 progressive difficulty tasks for the CPU design environment.

TASKS = {
    0: {
        "name": "8-Bit IoT Microcontroller",
        "description": "Design a minimal 8-bit microcontroller for IoT sensors. Focus on ultra-low power.",
        "difficulty": "easy",
        "max_steps": 20,
        "constraints": {
            "max_power_mw": 50.0,
            "max_area_mm2": 5.0,
            "target_throughput": 0.01,  # GIPS — very low, it's 8-bit
        },
        "constraint_text": (
            "POWER BUDGET: Total power MUST stay below 50mW. "
            "This is a battery-powered IoT sensor with no active cooling. "
            "The design should prioritize power efficiency over raw performance. "
            "Remove unnecessary components to save power. "
            "A simple single-cycle or 2-stage pipeline is sufficient."
        ),
        "scoring_weights": {
            "power": 0.6,
            "throughput": 0.2,
            "area": 0.15,
            "thermal": 0.05,
        },
    },
    1: {
        "name": "RV32IM 5-Stage Pipelined Core",
        "description": "Design a RISC-V RV32IM core with a classic 5-stage pipeline. Balance IPC vs area.",
        "difficulty": "medium",
        "max_steps": 30,
        "constraints": {
            "max_area_mm2": 10.0,
            "max_power_mw": 5000.0,  # 5W
            "target_ipc": 1.0,
        },
        "constraint_text": (
            "AREA BUDGET: Total die area MUST stay below 10mm² (at 7nm process). "
            "This is a cost-sensitive embedded processor. "
            "Maximize Instructions Per Clock (IPC) while staying within the area budget. "
            "Key decisions: L1 cache sizes, branch predictor complexity, forwarding paths. "
            "A 5-stage pipeline (IF-ID-EX-MEM-WB) is the baseline. "
            "Hazard detection and data forwarding can significantly improve IPC."
        ),
        "scoring_weights": {
            "ipc": 0.5,
            "area": 0.3,
            "power": 0.15,
            "thermal": 0.05,
        },
    },
    2: {
        "name": "M-Series Heterogeneous Superscalar",
        "description": "Design an Apple M-series inspired heterogeneous CPU with P-cores and E-cores.",
        "difficulty": "hard",
        "max_steps": 40,
        "constraints": {
            "max_power_density": 1.5,    # W/mm²
            "max_thermal_celsius": 100.0,
            "max_area_mm2": 200.0,
            "max_power_mw": 150000.0,    # 150W TDP
        },
        "constraint_text": (
            "THERMAL CONSTRAINT: Maximum power density MUST stay below 1.5 W/mm². "
            "Exceeding this causes thermal throttling — the effective clock frequency is HALVED. "
            "This is a high-performance heterogeneous processor inspired by Apple M-series. "
            "Balance Performance cores (P-cores) for burst workloads vs. Efficiency cores (E-cores) "
            "for sustained, power-efficient operation. "
            "Key decisions: P-core vs E-core count ratio, superscalar width, ROB size, "
            "cache hierarchy (L1/L2/L3), branch predictor type (TAGE is best but costly), "
            "and interconnect topology. "
            "Poor thermal design will trigger throttling, negating the performance gains. "
            "The best designs achieve high throughput while keeping hotspots under control."
        ),
        "scoring_weights": {
            "throughput": 0.35,
            "efficiency": 0.25,
            "area": 0.15,
            "thermal": 0.25,
        },
    },
}


def get_task(task_id: int) -> dict:
    """Get task configuration by ID."""
    return TASKS.get(task_id, TASKS[1])


def get_task_config_for_go(task_id: int) -> dict:
    """Get the task config payload to send to the Go engine."""
    task = get_task(task_id)
    return {
        "name": task["name"],
        "max_nodes": 50,
        "max_steps": task["max_steps"],
        "constraints": task["constraints"],
    }
