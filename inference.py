#!/usr/bin/env python3

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
from server.wegh_env import WEGHEnvironment

# Configuration
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

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
    return (
        f"Step {step}/{max_steps}\n"
        f"Task: {observation.task_name} | Constraints: {observation.task_constraints}\n"
        "-- Current PPA Metrics --\n"
        f"Estimated IPC: {observation.current_estimated_ipc:.2f}\n"
        f"Throughput (GIPS): {observation.throughput_gips:.2f}\n"
        f"Power: {observation.total_power_mw:.2f} mW\n"
        f"Area: {observation.total_area_mm2:.2f} mm²\n"
        f"Thermal: {observation.thermal_celsius:.2f} °C\n"
        f"Power Density: {observation.max_power_density:.2f}\n"
        f"Performance/Watt: {observation.perf_per_watt:.2f}\n"
        f"Throttled Factor: {observation.throttled_factor:.2f}\n\n"
        "-- Architecture State --\n"
        f"Active Components: {observation.active_components}\n"
        f"Available Actions: {observation.available_actions}\n"
        f"Feedback: {observation.feedback_string}\n\n"
        "Generate your next action in JSON format."
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
            max_tokens=250,
            temperature=0.1,
            timeout=30.0
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

def execute_task(env: WEGHEnvironment, client: OpenAI, task_name: str) -> float:
    observation = env.reset(task=task_name)
    step_count = 0
    rewards: List[float] = []

    print(f"[START] task={observation.task_name} env=WEGH model={MODEL_NAME}", flush=True)

    while not observation.done:
        step_count += 1
        action = get_model_action(client, observation, step_count)
        
        try:
            observation = env.step(action)
            error_msg = "null"
        except Exception as error:
            error_msg = str(error).replace("\n", " ")
            observation.done = True
            observation.reward = 0.0
            
        rewards.append(observation.reward)
        formatted_reward = f"{observation.reward:.2f}"
        is_done = str(observation.done).lower()
        
        action_str = action.model_dump_json().replace(" ", "")
        
        print(f"[STEP] step={step_count} action={action_str} reward={formatted_reward} done={is_done} error={error_msg}", flush=True)

        if step_count > observation.max_steps + 5:
            break

    is_success = str(observation.final_score > 0.5).lower()
    formatted_rewards = ",".join(f"{r:.2f}" for r in rewards)

    # Compliant output with the new Scaler guidelines (score field removed)
    print(f"[END] success={is_success} steps={step_count} rewards={formatted_rewards}", flush=True)
    return observation.final_score

TASK_IDS: List[str] = ["iot_8bit", "rv32im", "mseries_superscalar"]

def main() -> None:
    if not HF_TOKEN:
        print("[WARNING] HF_TOKEN missing.", file=sys.stderr)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")
    env = WEGHEnvironment()

    if not env.go_client.wait_for_ready(max_wait=15):
        print("[ERROR] WEGH Engine daemon failed.", file=sys.stderr)
        sys.exit(1)

    for task_name in TASK_IDS:
        execute_task(env, client, task_name)

    env.go_client.close()

if __name__ == "__main__":
    main()
