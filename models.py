# WEGH — OpenEnv Data Models
# Uses Pydantic models as required by OpenEnv 0.2.3's actual API.

from typing import Optional, List
from openenv.core.env_server import Action, Observation, State


class CPUAction(Action):
    """A single architectural decision made by the RL agent.
    
    The agent interacts with the CPU design by adding/removing components,
    resizing parameters, or connecting components in the microarchitecture DAG.
    """
    action_type: str = "resize"  # "add_component" | "remove_component" | "resize" | "connect" | "configure"
    target_component: str = ""   # Component ID (e.g., "l1_icache", "pcore", "rob")
    parameter_name: str = ""     # Parameter to modify (e.g., "size_kb", "count", "pipeline_depth")
    parameter_value: float = 0.0 # New value or delta
    source_node: str = ""        # For "connect" actions: source component
    target_node: str = ""        # For "connect" actions: destination component
    reasoning: str = ""          # Agent's reasoning (for logging)


class CPUObservation(Observation):
    """Complete observable state of the CPU design.
    
    Inherits from Observation which provides: done (bool), reward (float), metadata (dict).
    Includes PPA metrics, architecture state, task context,
    and critically, a feedback_string for LLM in-context learning.
    """
    # === PPA Metrics (from Go simulation engine) ===
    current_estimated_ipc: float = 0.0
    throughput_gips: float = 0.0
    effective_clock_ghz: float = 0.0
    total_power_mw: float = 0.0
    total_area_mm2: float = 0.0
    max_power_density: float = 0.0
    thermal_celsius: float = 0.0
    hotspot_count: int = 0
    throttled_factor: float = 1.0
    perf_per_watt: float = 0.0
    
    # === Architecture State ===
    active_components: str = "[]"  # JSON string of component list
    component_count: int = 0
    connection_count: int = 0
    
    # === Task Context ===
    task_id: int = 0
    task_name: str = ""
    task_constraints: str = ""     # Human-readable constraint description
    step_number: int = 0
    max_steps: int = 30
    
    # === Cumulative tracking ===
    cumulative_reward: float = 0.0
    
    # === LLM Feedback (CRITICAL for in-context learning) ===
    feedback_string: str = ""      # Engineering feedback explaining what happened
    
    # === Final Score (only meaningful when done=True) ===
    final_score: float = 0.0       # Normalized to [0.0, 1.0]
    
    # === Available Actions ===
    available_actions: str = "[]"  # JSON string of available action types


class CPUState(State):
    """Episode metadata for tracking task and Go daemon graph mapping.
    
    Inherits from State which provides: episode_id (str), step_count (int).
    """
    task_id: int = 0
    task_name: str = ""
    max_steps: int = 30
    go_graph_id: str = ""          # Maps to Go daemon's in-memory graph instance
    cumulative_reward: float = 0.0
