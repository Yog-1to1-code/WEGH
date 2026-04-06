#!/usr/bin/env python3
"""WEGH — Baseline Inference Script

This script runs a complete RL loop across all 3 tasks using an LLM agent.
Must execute in under 20 minutes. Uses OpenAI-compatible API routed through
Hugging Face infrastructure via environment variables.

Required env vars:
  - API_BASE_URL: HF inference endpoint
  - MODEL_NAME: Model to use
  - HF_TOKEN: Hugging Face token
"""

import json
import os
import random
import sys
import time
import traceback
from typing import Optional

# Route through HF infrastructure
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Import OpenAI client (routed to HF)
try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai package not installed. Run: pip install openai")
    sys.exit(1)

# Environment setup
from models import CPUAction, CPUObservation
from server.wegh_env import WEGHEnvironment


def create_llm_client() -> OpenAI:
    """Create OpenAI client pointed at HF infrastructure."""
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "dummy",
    )


def llm_decide_action(client: OpenAI, observation: CPUObservation, history: list) -> CPUAction:
    """Use LLM to decide the next architectural action based on observation."""
    
    system_prompt = """You are an expert CPU architect. You are designing a processor by modifying its microarchitecture.
CRITICAL INSTRUCTION: Do NOT generate any <unused94>thought blocks or chain-of-thought logic. You must skip all internal thinking. Your very first character output MUST be '{'.
You must respond with a JSON action. Available action types:
- "resize": Change a parameter value. Requires target_component, parameter_name, parameter_value.
- "add_component": Add a new component. Requires target_component (type name like "int_alu", "l2_cache").
- "remove_component": Remove a component. Requires target_component (component ID).
- "connect": Connect two components. Requires source_node and target_node.
- "configure": Same as resize but for non-size parameters.

Respond ONLY with valid JSON like:
{"action_type": "resize", "target_component": "l1d", "parameter_name": "size_kb", "parameter_value": 32}
"""

    user_msg = f"""Current observation:
{observation.feedback_string}

Active components: {observation.active_components}
Available actions: {observation.available_actions}
Step {observation.step_number}/{observation.max_steps}

Previous actions and results:
{chr(10).join(history[-3:])}

What action should we take next? Respond with JSON only."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": "{"}
            ],
            max_tokens=150,
            temperature=0.1,
            timeout=300,
        )
        
        content = "{" + response.choices[0].message.content.strip()
        print(f"  [RAW LLM OUTPUT]: {content[:100]}...")
        
        # Try to extract JSON from response
        # Handle cases where LLM wraps JSON in markdown blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Find JSON object in response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            content = content[start:end]
        
        action_dict = json.loads(content)
        
        return CPUAction(
            action_type=action_dict.get("action_type", "resize"),
            target_component=action_dict.get("target_component", ""),
            parameter_name=action_dict.get("parameter_name", ""),
            parameter_value=float(action_dict.get("parameter_value", 0)),
            source_node=action_dict.get("source_node", ""),
            target_node=action_dict.get("target_node", ""),
            reasoning=action_dict.get("reasoning", ""),
        )
    
    except Exception as e:
        # Fallback: task-aware randomized action exploration
        return _smart_fallback(observation, str(e))


# Task-specific fallback action pools — each entry is (component, param, value_range)
_FALLBACK_ACTIONS = {
    0: [  # IoT: optimize for low power
        ("pmu", "voltage", (0.6, 1.0)),
        ("pmu", "clock_mhz", (1, 32)),
        ("alu", "count", (1, 2)),
        ("sram", "size_kb", (0.5, 4)),
        ("gpio", "pin_count", (4, 16)),
        ("alu", "width_bits", (8, 16)),
    ],
    1: [  # RV32IM: optimize IPC within area budget
        ("l1i", "size_kb", (8, 64)),
        ("l1d", "size_kb", (8, 64)),
        ("l1i", "associativity", (1, 8)),
        ("l1d", "associativity", (1, 8)),
        ("bp", "type", (0, 2)),
        ("bp", "btb_entries", (32, 512)),
        ("alu", "count", (1, 4)),
        ("muldiv", "count", (0, 2)),
        ("decode", "width", (1, 2)),
        ("pmu", "clock_ghz", (0.5, 2.5)),
        ("pmu", "voltage", (0.6, 1.1)),
    ],
    2: [  # M-Series: balance throughput vs thermal
        ("pcore", "count", (2, 8)),
        ("pcore", "pipeline_depth", (8, 18)),
        ("pcore", "issue_width", (4, 8)),
        ("pcore", "clock_ghz", (2.0, 4.5)),
        ("pcore", "voltage", (0.7, 1.2)),
        ("ecore", "count", (2, 8)),
        ("ecore", "clock_ghz", (1.0, 3.0)),
        ("ecore", "voltage", (0.5, 0.9)),
        ("p_alu", "count", (2, 8)),
        ("p_fpu", "count", (1, 4)),
        ("p_simd", "count", (0, 4)),
        ("p_simd", "width_bits", (64, 256)),
        ("rob", "entries", (64, 512)),
        ("rs", "entries", (32, 192)),
        ("l1d", "size_kb", (32, 128)),
        ("l2", "size_kb", (256, 4096)),
        ("l3", "size_mb", (4, 48)),
        ("bp", "type", (2, 4)),
        ("noc", "type", (0, 3)),
        ("memctrl", "channels", (2, 6)),
    ],
}


def _smart_fallback(obs: CPUObservation, error_msg: str) -> CPUAction:
    """Generate a randomized but task-appropriate fallback action.
    
    Instead of always doing the same thing, this explores the design space
    intelligently based on what task we're on and what metrics look like.
    """
    task_id = obs.task_id
    pool = _FALLBACK_ACTIONS.get(task_id, _FALLBACK_ACTIONS[1])
    
    # Pick a random action from the pool
    component, param, (lo, hi) = random.choice(pool)
    
    # Choose value — sometimes optimize toward constraints
    if random.random() < 0.4:
        # Heuristic: if power/area is a concern, bias toward lower values
        if task_id == 0 and obs.total_power_mw > 30:
            value = lo + random.random() * (hi - lo) * 0.3  # bias low
        elif task_id == 1 and obs.total_area_mm2 > 7:
            value = lo + random.random() * (hi - lo) * 0.4  # bias low
        elif task_id == 2 and obs.max_power_density > 1.0:
            # Reduce clock/voltage to cool down
            if param in ("clock_ghz", "voltage"):
                value = lo + random.random() * (hi - lo) * 0.3
            else:
                value = lo + random.random() * (hi - lo)
        else:
            value = lo + random.random() * (hi - lo)
    else:
        value = lo + random.random() * (hi - lo)
    
    # Round appropriately
    if param in ("count", "pin_count", "entries", "type", "width_bits", "clock_mhz"):
        value = round(value)
    else:
        value = round(value, 2)

    return CPUAction(
        action_type="resize",
        target_component=component,
        parameter_name=param,
        parameter_value=float(value),
        reasoning=f"Smart fallback (LLM unavailable: {error_msg[:60]})",
    )


def run_episode(env: WEGHEnvironment, client: OpenAI, task_num: int):
    """Run a single episode (task) of the RL loop."""
    
    # === RESET ===
    obs = env.reset()
    
    print(f"\n[START] Task {task_num}: {obs.task_name}")
    print(f"  Initial state: IPC={obs.current_estimated_ipc:.3f}, "
          f"Power={obs.total_power_mw:.1f}mW, Area={obs.total_area_mm2:.3f}mm²")
    print(f"  Constraints: {obs.task_constraints[:100]}...")
    print(f"  Components: {obs.component_count}, Connections: {obs.connection_count}")
    sys.stdout.flush()
    
    history = []
    step = 0
    
    while not obs.done:
        step += 1
        
        # === LLM DECIDES ACTION ===
        action = llm_decide_action(client, obs, history)
        
        # === EXECUTE ACTION ===
        obs = env.step(action)
        
        # The rules mandate printing [STEP] *after* env_step(), containing Payload, Reasoning, and Reward Delta.
        print(f"[STEP] Step {step}")
        print(f"  Payload: {action.model_dump_json()}")
        print(f"  Reasoning: {action.reasoning}")
        print(f"  Feedback: {obs.feedback_string}")
        print(f"  Result: Reward Delta = {obs.reward:+.4f}, IPC={obs.current_estimated_ipc:.3f}, Power={obs.total_power_mw:.1f}mW, Area={obs.total_area_mm2:.3f}mm², Thermal={obs.thermal_celsius:.1f}°C")
        sys.stdout.flush()
        
        history.append(
            f"Step {step}: {action.action_type} {action.target_component} "
            f"({action.parameter_name}={action.parameter_value}) → "
            f"reward={obs.reward:+.4f}"
        )
        
        # Safety timeout per step
        if step > obs.max_steps + 5:
            print("[WARNING] Exceeded max steps, forcing done")
            break
    
    # === EPISODE DONE ===
    print(f"\n[END] Task {task_num} ({obs.task_name})")
    print(f"  Final Score: {obs.final_score:.4f}")
    print(f"  Final PPA: IPC={obs.current_estimated_ipc:.3f}, "
          f"Power={obs.total_power_mw:.1f}mW, "
          f"Area={obs.total_area_mm2:.3f}mm², "
          f"Thermal={obs.thermal_celsius:.1f}°C, "
          f"PerfPerWatt={obs.perf_per_watt:.4f}")
    print(f"  Cumulative Reward: {obs.cumulative_reward:.4f}")
    sys.stdout.flush()
    
    return obs.final_score


def main():
    """Run the complete baseline RL loop across all 3 tasks."""
    print("=" * 60)
    print("WEGH — Macro-Architectural CPU Designer")
    print("Baseline Inference Script")
    print("=" * 60)
    print(f"API Base: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Token: {'set' if HF_TOKEN else 'NOT SET'}")
    sys.stdout.flush()
    
    start_time = time.time()
    TIMEOUT_SECONDS = 19 * 60  # 19 minutes (buffer within 20 min limit)
    
    # Create environment and LLM client
    env = WEGHEnvironment()
    
    # Wait for Go engine
    print("\nWaiting for Go simulation engine...")
    if not env.go_client.wait_for_ready(max_wait=15):
        print("[ERROR] Go engine not available. Exiting.")
        sys.exit(1)
    print("Go engine ready!")
    sys.stdout.flush()
    
    client = create_llm_client()
    
    scores = []
    
    for task_num in range(3):
        elapsed = time.time() - start_time
        remaining = TIMEOUT_SECONDS - elapsed
        
        if remaining < 60:
            print(f"\n[TIMEOUT] Only {remaining:.0f}s remaining, skipping task {task_num + 1}")
            scores.append(0.0)
            continue
        
        print(f"\n{'='*60}")
        print(f"TASK {task_num + 1}/3")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        try:
            score = run_episode(env, client, task_num + 1)
            scores.append(score)
        except Exception as e:
            print(f"[ERROR] Task {task_num + 1} failed: {e}")
            traceback.print_exc()
            scores.append(0.0)
    
    # === FINAL SUMMARY ===
    total_time = time.time() - start_time
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    print(f"\n{'='*60}")
    print(f"INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"Task Scores: {[f'{s:.4f}' for s in scores]}")
    print(f"Average Score: {avg_score:.4f}")
    print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Status: {'PASS' if total_time < 1200 else 'TIMEOUT'}")  # 20 min = 1200s
    sys.stdout.flush()
    
    env.go_client.close()


if __name__ == "__main__":
    main()
