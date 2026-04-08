#!/usr/bin/env python3
"""
WEGH — Inference Script (OpenEnv Phase 2 Compliant)

Connects to the ALREADY-RUNNING WEGH FastAPI server via WebSocket client.
Does NOT instantiate the server-side WEGHEnvironment directly.
"""

import json
import os
import sys
from typing import List

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai package not installed.")
    sys.exit(1)

from pydantic import ValidationError
from models import CPUAction, CPUObservation
from client import WEGHEnv

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# The server is already running inside the container on this port
ENV_PORT = os.getenv("ENV_PORT", "7860")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", f"http://127.0.0.1:{ENV_PORT}")

def build_system_prompt() -> str:
    return (
        "You are an expert systems engineer and CPU architect. "
        "Your task is to design a CPU architecture ranging from strict IoT microcontrollers to M-Series Superscalar configurations. "
        "You operate by modifying a microarchitectural DAG. "
        "Your primary objective is to maximize the final score derived from PPA heuristics without thermal throttling. "
        "Output ONLY a valid JSON object matching this schema: "
        '{"action_type": "<action>", "target_component": "<id>", "parameter_name": "<name>", '
        '"parameter_value": <float>, "source_node": "<id>", "target_node": "<id>", "reasoning": "<string>"}\n'
        "Valid action types: resize, add_component, remove_component, connect, configure."
    )

def build_user_prompt(observation: CPUObservation, step: int, max_steps: int) -> str:
    # Truncate feedback to first 500 chars to reduce input tokens and LLM latency
    feedback_truncated = observation.feedback_string[:500] if observation.feedback_string else ""
    return (
        f"Step {step}/{max_steps}\n"
        f"Task: {observation.task_name} | Constraints: {observation.task_constraints}\n"
        f"IPC:{observation.current_estimated_ipc:.2f} Throughput:{observation.throughput_gips:.2f}GIPS "
        f"Power:{observation.total_power_mw:.1f}mW Area:{observation.total_area_mm2:.2f}mm² "
        f"PD:{observation.max_power_density:.2f} Thermal:{observation.thermal_celsius:.1f}°C "
        f"PPW:{observation.perf_per_watt:.2f} Throttle:{observation.throttled_factor:.2f}\n"
        f"Components: {observation.active_components}\n"
        f"Feedback: {feedback_truncated}\n"
        "Respond with a single JSON action."
    )

def extract_json_payload(raw_content: str) -> str:
    content = raw_content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    
    start_index = content.find("{")
    end_index = content.rfind("}") + 1
    
    if start_index >= 0 and end_index > start_index:
        return content[start_index:end_index]
    return content

def get_model_action(client: OpenAI, observation: CPUObservation, step: int) -> CPUAction:
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(observation, step, observation.max_steps)}
    ]
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=150,
            temperature=0.1,
            timeout=10.0
        )
        
        raw_output = response.choices[0].message.content or "{}"
        json_string = extract_json_payload(raw_output)
        return CPUAction(**json.loads(json_string))
        
    except (json.JSONDecodeError, ValidationError) as error:
        # Fallback to prevent termination on parse failure
        return CPUAction(
            action_type="resize",
            target_component="l1d",
            parameter_name="size_kb",
            parameter_value=32.0,
            reasoning=f"Parse fallback: {error}"
        )
    except Exception as error:
        return CPUAction(
            action_type="resize",
            target_component="l1d",
            parameter_name="size_kb",
            parameter_value=32.0,
            reasoning=f"Network fallback: {error}"
        )

def execute_task(env, llm_client: OpenAI, task_name: str) -> float:
    """Run one episode for a given task via the env client."""
    result = env.reset(task=task_name)
    observation = result.observation
    step_count = 0
    rewards: List[float] = []

    print(f"[START] task={observation.task_name} env=WEGH model={MODEL_NAME}", flush=True)

    while not result.done:
        step_count += 1
        action = get_model_action(llm_client, observation, step_count)
        
        try:
            result = env.step(action)
            observation = result.observation
            error_msg = "null"
        except Exception as error:
            error_msg = str(error).replace("\n", " ")
            # On error, mark done with zero reward
            result = type('FakeResult', (), {'done': True, 'reward': 0.0, 'observation': observation})()
            observation.done = True
            observation.reward = 0.0
            
        step_reward = result.reward if result.reward is not None else 0.0
        rewards.append(step_reward)
        formatted_reward = f"{step_reward:.2f}"
        is_done = str(result.done).lower()
        
        action_str = action.model_dump_json().replace(" ", "")
        
        print(f"[STEP] step={step_count} action={action_str} reward={formatted_reward} done={is_done} error={error_msg}", flush=True)

        if step_count >= observation.max_steps:
            break

    final_score = getattr(observation, 'final_score', 0.0)
    is_success = str(final_score > 0.5).lower()
    formatted_rewards = ",".join(f"{r:.2f}" for r in rewards)

    # Compliant output with the new Scaler guidelines (score field removed)
    print(f"[END] success={is_success} steps={step_count} rewards={formatted_rewards}", flush=True)
    return final_score

TASK_IDS: List[str] = ["iot_8bit", "rv32im", "mseries_superscalar"]

def main() -> None:
    if HF_TOKEN is None:
        print("[WARNING] HF_TOKEN missing.", file=sys.stderr)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")
    
    # Connect to the ALREADY RUNNING server via WebSocket client
    # The server is started by the container's entrypoint.sh, NOT by this script
    print(f"[INFO] Connecting to WEGH environment at {ENV_BASE_URL}", file=sys.stderr)
    
    with WEGHEnv(base_url=ENV_BASE_URL).sync() as env:
        for task_name in TASK_IDS:
            execute_task(env, llm_client, task_name)

if __name__ == "__main__":
    main()
