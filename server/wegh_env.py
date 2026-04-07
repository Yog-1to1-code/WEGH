# WEGH — Core Environment (OpenEnv Server)
# Inherits from openenv.core.env_server.Environment
# Bridges the OpenEnv API to the Go simulation daemon.

import json
import logging
import uuid
import socket
import contextlib
import subprocess
import os
import atexit
from typing import Any, Optional

from openenv.core.env_server import Environment
from models import CPUAction, CPUObservation, CPUState
from server.go_client import GoEngineClient
from server.task_configs import get_task, get_task_config_for_go
from server.reward import compute_step_reward, compute_final_score

logger = logging.getLogger("wegh")

def sanitize_grade(raw_score: Any) -> float:
    """Sanitizes rewards and scores to pure Python floats with strict exclusive boundaries."""
    try:
        if raw_score is None:
            return 0.001
        score = float(raw_score)
    except (ValueError, TypeError):
        return 0.001

    if score >= 1.0:
        return 0.999
    elif score <= 0.0:
        return 0.001
    return score


def get_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class WEGHEnvironment(Environment[CPUAction, CPUObservation, CPUState]):
    """WEGH: Macro-Architectural CPU Designer
    
    An RL environment where agents learn to design CPU microarchitectures
    by adding, removing, and configuring components in a Directed Acyclic Graph (DAG).
    Three progressive tasks from IoT microcontroller to Apple M-series superscalar.
    """
    
    def __init__(self):
        super().__init__()
        
        # 1. Dynamically acquire a free local port
        self.go_port = get_free_port()
        self.go_url = f"http://127.0.0.1:{self.go_port}"
        
        # 2. Spawn the Go Simulator as a child daemon process
        bin_path = os.path.join(os.getcwd(), "go-engine")
        if not os.path.exists(bin_path) and os.path.exists("/app/go-engine"):
            bin_path = "/app/go-engine"
        elif not os.path.exists(bin_path):
            bin_path = "./go-engine"
            
        logger.info(f"Spawning native Go simulation daemon on {self.go_url}")
        self._go_process = subprocess.Popen(
            [bin_path, f"--bind=127.0.0.1:{self.go_port}", "--max-memory=4096"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Ensure cleanup if Python crashes
        atexit.register(self._cleanup_daemon)
        
        # 3. Mount Client
        self.go_client = GoEngineClient(base_url=self.go_url)
        if not self.go_client.wait_for_ready(max_wait=10.0):
            logger.error("Failed to internalize Go native daemon!")
            
        self._state = CPUState()
        self._prev_metrics = {}
        self._current_task_id = 0
        self._task_cycle = 0  # Cycles through tasks 0,1,2
        self._cumulative_reward = 0.0
    
    def _cleanup_daemon(self):
        if hasattr(self, '_go_process') and self._go_process.poll() is None:
            self._go_process.terminate()
            try:
                self._go_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._go_process.kill()

    def __del__(self):
        self._cleanup_daemon()
    
    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> CPUObservation:
        """Initialize a new episode.
        
        Purges old state, selects a task difficulty, initializes the Go daemon's
        DAG, and returns the baseline observation with constraints.
        """
        # Select task from kwargs or cycle through fallbacks
        task_override = kwargs.get("task")
        if task_override == "iot_8bit":
            task_id = 0
        elif task_override == "rv32im":
            task_id = 1
        elif task_override == "mseries_superscalar":
            task_id = 2
        else:
            task_id = self._task_cycle % 3
            self._task_cycle += 1
            
        self._current_task_id = task_id
        
        # Generate new episode ID
        ep_id = episode_id or str(uuid.uuid4())
        
        # Get task configuration
        task = get_task(task_id)
        task_config = get_task_config_for_go(task_id)
        
        # Initialize state
        self._state = CPUState(
            episode_id=ep_id,
            step_count=0,
            task_id=task_id,
            task_name=task["name"],
            max_steps=task["max_steps"],
            go_graph_id=ep_id,
            cumulative_reward=0.0,
        )
        self._cumulative_reward = 0.0
        
        # Reset Go engine
        go_response = self.go_client.reset(ep_id, task_id, task_config)
        
        if go_response.get("status") != "ok":
            logger.error("Go engine reset failed: %s", go_response.get("error"))
            return self._error_observation(task, "Engine initialization failed")
        
        metrics = go_response.get("initial_metrics", {})
        self._prev_metrics = dict(metrics)
        
        # Build initial observation
        components = go_response.get("components", [])
        connections = go_response.get("connections", [])
        
        obs = CPUObservation(
            # Inherited from Observation base
            done=False,
            reward=0.0,
            
            # PPA Metrics
            current_estimated_ipc=metrics.get("ipc", 0),
            throughput_gips=metrics.get("throughput_gips", 0),
            effective_clock_ghz=metrics.get("effective_clock_ghz", 0),
            total_power_mw=metrics.get("total_power_mw", 0),
            total_area_mm2=metrics.get("total_area_mm2", 0),
            max_power_density=metrics.get("max_power_density", 0),
            thermal_celsius=metrics.get("thermal_celsius", 0),
            hotspot_count=metrics.get("hotspot_count", 0),
            throttled_factor=metrics.get("throttled_factor", 1.0),
            perf_per_watt=metrics.get("perf_per_watt", 0),
            
            # Architecture state
            active_components=json.dumps(components),
            component_count=len(components),
            connection_count=len(connections),
            
            # Task context
            task_id=task_id,
            task_name=task["name"],
            task_constraints=task["constraint_text"],
            step_number=0,
            max_steps=task["max_steps"],
            
            # Reward
            cumulative_reward=0.0,
            
            # Feedback
            feedback_string=(
                f"=== WEGH: {task['name']} ===\n"
                f"Difficulty: {task['difficulty'].upper()}\n"
                f"Constraints: {task['constraint_text']}\n\n"
                f"Initial design loaded with {len(components)} components and {len(connections)} connections.\n"
                f"Current PPA: IPC={metrics.get('ipc', 0):.3f}, "
                f"Power={metrics.get('total_power_mw', 0):.1f}mW, "
                f"Area={metrics.get('total_area_mm2', 0):.3f}mm², "
                f"Thermal={metrics.get('thermal_celsius', 0):.1f}°C\n\n"
                f"Available actions: add_component, remove_component, resize, connect, configure.\n"
                f"You have {task['max_steps']} steps to optimize this design."
            ),
            
            final_score=sanitize_grade(0.0),
            available_actions=json.dumps(go_response.get("available_actions", [])),
        )
        
        return obs
    
    def step(self, action: CPUAction, timeout_s: Optional[float] = None, **kwargs: Any) -> CPUObservation:
        """Execute one architectural decision.
        
        Serializes the action, POSTs to Go daemon, receives updated PPA metrics,
        computes dense reward, generates feedback, and returns typed observation.
        """
        self._state.step_count += 1
        step_num = self._state.step_count
        task = get_task(self._state.task_id)
        
        # Serialize action for Go engine
        go_action = {
            "type": action.action_type,
            "component": action.target_component,
            "param_name": action.parameter_name,
            "value": action.parameter_value,
            "source_node": action.source_node,
            "target_node": action.target_node,
        }
        
        # Send to Go engine
        go_response = self.go_client.step(self._state.episode_id, go_action)
        
        metrics = go_response.get("metrics", {})
        action_valid = go_response.get("valid", False)
        validation_errors = go_response.get("validation_errors", [])
        engineering_notes = go_response.get("engineering_notes", "")
        
        # Compute dense reward
        raw_step_reward = compute_step_reward(
            metrics=metrics,
            prev_metrics=self._prev_metrics,
            action_valid=action_valid,
            validation_errors=validation_errors,
            task_id=self._state.task_id,
            task_constraints=task["constraints"],
        )
        step_reward = float(raw_step_reward)
        
        self._cumulative_reward += step_reward
        self._state.cumulative_reward = round(self._cumulative_reward, 4)
        
        # Check if episode is done
        done = step_num >= task["max_steps"]
        
        # Compute final score if done
        final_score = sanitize_grade(0.0)
        if done:
            final_score = sanitize_grade(compute_final_score(metrics, self._state.task_id, task["constraints"]))
        
        # Update prev metrics
        self._prev_metrics = dict(metrics)
        
        # Build feedback string
        components = go_response.get("components", [])
        connections = go_response.get("connections", [])
        
        feedback = self._build_feedback(
            action=action,
            action_valid=action_valid,
            engineering_notes=engineering_notes,
            metrics=metrics,
            step_num=step_num,
            max_steps=task["max_steps"],
            step_reward=step_reward,
            done=done,
            final_score=final_score,
            task=task,
        )
        
        obs = CPUObservation(
            # Inherited from Observation
            done=done,
            reward=step_reward,
            
            # PPA Metrics
            current_estimated_ipc=metrics.get("ipc", 0),
            throughput_gips=metrics.get("throughput_gips", 0),
            effective_clock_ghz=metrics.get("effective_clock_ghz", 0),
            total_power_mw=metrics.get("total_power_mw", 0),
            total_area_mm2=metrics.get("total_area_mm2", 0),
            max_power_density=metrics.get("max_power_density", 0),
            thermal_celsius=metrics.get("thermal_celsius", 0),
            hotspot_count=metrics.get("hotspot_count", 0),
            throttled_factor=metrics.get("throttled_factor", 1.0),
            perf_per_watt=metrics.get("perf_per_watt", 0),
            
            # Architecture state
            active_components=json.dumps(components),
            component_count=len(components),
            connection_count=len(connections),
            
            # Task context
            task_id=self._state.task_id,
            task_name=self._state.task_name,
            task_constraints=task["constraint_text"],
            step_number=step_num,
            max_steps=task["max_steps"],
            
            # Reward
            cumulative_reward=round(self._cumulative_reward, 4),
            
            # Feedback
            feedback_string=feedback,
            final_score=final_score,
            available_actions=json.dumps(["add_component", "remove_component", "resize", "connect", "configure"]),
        )
        
        return obs
    
    @property
    def state(self) -> CPUState:
        """Return current episode state."""
        return self._state
    
    def _build_feedback(self, action, action_valid, engineering_notes, metrics,
                        step_num, max_steps, step_reward, done, final_score, task):
        """Generate LLM-readable engineering feedback."""
        lines = []
        
        # Action result
        status = "✓ VALID" if action_valid else "✗ INVALID"
        lines.append(f"[Step {step_num}/{max_steps}] Action: {action.action_type} {action.target_component} "
                     f"({action.parameter_name}={action.parameter_value}) → {status}")
        
        # Engineering notes from Go
        if engineering_notes:
            lines.append(f"Engineer: {engineering_notes}")
        
        # Current PPA snapshot
        lines.append(
            f"PPA Status: IPC={metrics.get('ipc', 0):.3f} | "
            f"Power={metrics.get('total_power_mw', 0):.1f}mW | "
            f"Area={metrics.get('total_area_mm2', 0):.3f}mm² | "
            f"Thermal={metrics.get('thermal_celsius', 0):.1f}°C | "
            f"PowerDensity={metrics.get('max_power_density', 0):.3f}W/mm²"
        )
        
        # Constraint status
        constraints = task["constraints"]
        warnings = []
        if "max_power_mw" in constraints:
            power = metrics.get("total_power_mw", 0)
            limit = constraints["max_power_mw"]
            pct = power / limit * 100
            if power > limit:
                warnings.append(f"⚠ POWER EXCEEDED: {power:.1f}mW > {limit:.1f}mW ({pct:.0f}%)")
            else:
                warnings.append(f"Power: {power:.1f}/{limit:.1f}mW ({pct:.0f}%)")
        
        if "max_area_mm2" in constraints:
            area = metrics.get("total_area_mm2", 0)
            limit = constraints["max_area_mm2"]
            pct = area / limit * 100
            if area > limit:
                warnings.append(f"⚠ AREA EXCEEDED: {area:.3f}mm² > {limit:.1f}mm² ({pct:.0f}%)")
            else:
                warnings.append(f"Area: {area:.3f}/{limit:.1f}mm² ({pct:.0f}%)")
        
        if "max_power_density" in constraints:
            pd = metrics.get("max_power_density", 0)
            limit = constraints["max_power_density"]
            if pd > limit:
                warnings.append(f"🔥 THERMAL HOTSPOT: {pd:.3f}W/mm² > {limit:.1f}W/mm² — THROTTLING ACTIVE")
            elif pd > limit * 0.8:
                warnings.append(f"⚠ Thermal warning: {pd:.3f}W/mm² approaching {limit:.1f}W/mm² limit")
        
        if warnings:
            lines.append("Constraints: " + " | ".join(warnings))
        
        # Reward
        lines.append(f"Reward: {step_reward:+.4f} (cumulative: {self._cumulative_reward:.4f})")
        
        if done:
            lines.append(f"\n{'='*50}")
            lines.append(f"EPISODE COMPLETE — Final Score: {final_score:.4f}/1.0")
            lines.append(f"{'='*50}")
        
        return "\n".join(lines)
    
    def _error_observation(self, task, error_msg):
        """Return an observation for error cases."""
        return CPUObservation(
            done=True,
            reward=sanitize_grade(0.0),
            task_id=self._state.task_id,
            task_name=task["name"],
            task_constraints=task["constraint_text"],
            max_steps=task["max_steps"],
            feedback_string=f"ERROR: {error_msg}",
            final_score=sanitize_grade(0.0),
        )
