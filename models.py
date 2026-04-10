# WEGH — OpenEnv Data Models
# Pure Pydantic models — NO openenv-core dependency.
# Mirrors the Go engine structs exactly for full schema compliance.

from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class CPUAction(BaseModel):
    """A single architectural decision made by the RL agent.

    The agent interacts with the CPU design by adding/removing components,
    resizing parameters, or connecting components in the microarchitecture DAG.
    """
    action_type: str = Field("resize", description="Action type: resize | add_component | remove_component | connect | configure")
    target_component: str = Field("", description="Component ID (e.g., 'l1_icache', 'pcore', 'rob')")
    parameter_name: str = Field("", description="Parameter to modify (e.g., 'size_kb', 'count', 'pipeline_depth')")
    parameter_value: float = Field(0.0, description="New value or delta for the parameter")
    source_node: str = Field("", description="For 'connect' actions: source component")
    target_node: str = Field("", description="For 'connect' actions: destination component")
    reasoning: str = Field("", description="Agent's reasoning (for logging)")

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        valid = {"resize", "add_component", "remove_component", "connect", "configure"}
        if v not in valid:
            return "resize"
        return v


class CPUObservation(BaseModel):
    """Complete observable state of the CPU design.

    Includes PPA metrics, architecture state, task context,
    and critically, a feedback_string for LLM in-context learning.
    """
    # === PPA Metrics (from Go simulation engine) ===
    current_estimated_ipc: float = Field(0.0, description="Instructions Per Clock cycle")
    throughput_gips: float = Field(0.0, ge=0.0, description="Throughput in GIPS")
    effective_clock_ghz: float = Field(0.0, ge=0.0, description="Effective clock speed (GHz)")
    total_power_mw: float = Field(0.0, ge=0.0, description="Total power consumption (mW)")
    total_area_mm2: float = Field(0.0, ge=0.0, description="Total die area (mm²)")
    max_power_density: float = Field(0.0, ge=0.0, description="Peak power density (W/mm²)")
    thermal_celsius: float = Field(0.0, description="Junction temperature (°C)")
    hotspot_count: int = Field(0, ge=0, description="Number of thermal hotspots")
    throttled_factor: float = Field(1.0, ge=0.0, le=1.0, description="Thermal throttle factor (1.0=no throttle)")
    perf_per_watt: float = Field(0.0, ge=0.0, description="Performance per watt ratio")

    # === Architecture State ===
    active_components: str = Field("[]", description="JSON string of active component IDs")
    component_count: int = Field(0, ge=0, description="Number of active components")
    connection_count: int = Field(0, ge=0, description="Number of connections in DAG")

    # === Task Context ===
    task_id: int = Field(0, ge=0, le=2, description="Task index (0=IoT, 1=RV32IM, 2=M-Series)")
    task_name: str = Field("", description="Human-readable task name")
    task_constraints: str = Field("", description="Human-readable constraint description")
    step_number: int = Field(0, ge=0, description="Current step in episode")
    max_steps: int = Field(30, ge=1, description="Maximum steps in this episode")

    # === Cumulative tracking ===
    cumulative_reward: float = Field(0.0, description="Running total reward this episode")

    # === LLM Feedback (critical for in-context learning) ===
    feedback_string: str = Field("", description="Engineering feedback explaining what happened")

    # === Final Score (only meaningful when done=True) ===
    final_score: float = Field(0.0, ge=0.0, le=1.0, description="Normalized final score [0, 1]")

    # === Available Actions ===
    available_actions: str = Field("[]", description="JSON string of available action types")

    # === Episode state ===
    done: bool = Field(False, description="Whether the episode is finished")
    reward: float = Field(0.0, description="Step reward")


class CPUState(BaseModel):
    """Episode metadata for tracking task and Go daemon graph mapping."""
    episode_id: str = Field("", description="Unique episode identifier")
    step_count: int = Field(0, ge=0, description="Current step count")
    task_id: int = Field(0, ge=0, description="Task index")
    task_name: str = Field("", description="Task name")
    max_steps: int = Field(30, ge=1, description="Maximum episode steps")
    go_graph_id: str = Field("", description="Maps to Go daemon's in-memory graph instance")
    cumulative_reward: float = Field(0.0, description="Running total reward")


# ── Action space schema (for LLM prompting) ────────────────────────────────
ACTION_SCHEMA = {
    "type": "object",
    "required": ["action_type", "target_component", "parameter_name", "parameter_value"],
    "properties": {
        "action_type": {
            "type": "string",
            "enum": ["resize", "add_component", "remove_component", "connect", "configure"],
            "description": "Type of architectural modification"
        },
        "target_component": {
            "type": "string",
            "description": "Component ID to modify (e.g., 'l1d', 'pcore', 'rob')"
        },
        "parameter_name": {
            "type": "string",
            "description": "Parameter name (e.g., 'size_kb', 'count', 'pipeline_depth')"
        },
        "parameter_value": {
            "type": "number",
            "description": "New value for the parameter"
        },
        "source_node": {
            "type": "string",
            "description": "Source component for 'connect' actions"
        },
        "target_node": {
            "type": "string",
            "description": "Target component for 'connect' actions"
        },
        "reasoning": {
            "type": "string",
            "description": "Agent's reasoning for this action"
        }
    }
}

# ── Observation space schema ───────────────────────────────────────────────
OBSERVATION_SCHEMA = {
    "type": "object",
    "properties": {
        "current_estimated_ipc": {"type": "number", "description": "Instructions Per Clock"},
        "throughput_gips": {"type": "number", "description": "Throughput (GIPS)"},
        "effective_clock_ghz": {"type": "number", "description": "Effective clock (GHz)"},
        "total_power_mw": {"type": "number", "description": "Total power (mW)"},
        "total_area_mm2": {"type": "number", "description": "Total area (mm²)"},
        "max_power_density": {"type": "number", "description": "Power density (W/mm²)"},
        "thermal_celsius": {"type": "number", "description": "Junction temperature (°C)"},
        "throttled_factor": {"type": "number", "minimum": 0, "maximum": 1},
        "perf_per_watt": {"type": "number"},
        "task_id": {"type": "integer", "minimum": 0, "maximum": 2},
        "step_number": {"type": "integer"},
        "max_steps": {"type": "integer"},
        "feedback_string": {"type": "string"},
        "cumulative_reward": {"type": "number"},
        "active_components": {"type": "string"},
    }
}
