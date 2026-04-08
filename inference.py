#!/usr/bin/env python3
"""
inference.py — WEGH OpenEnv Agent
==================================
Runs an LLM agent through all 3 CPU design tasks and emits structured stdout logs.

Required environment variables:
    API_BASE_URL      LLM API endpoint
    MODEL_NAME        Model identifier
    HF_TOKEN          HuggingFace / API key
    LOCAL_IMAGE_NAME  (optional) Docker image to launch as env server

Stdout format (must not deviate):
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import json
import os
import sys
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI
from pydantic import ValidationError

from models import CPUAction, CPUObservation
from client import WEGHEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
BENCHMARK    = "WEGH"

TASKS = ["iot_8bit", "rv32im", "mseries_superscalar"]
SUCCESS_THRESHOLD = 0.5
TEMPERATURE = 0.1
MAX_TOKENS  = 150

# ---------------------------------------------------------------------------
# Logging helpers — must match the spec exactly
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert systems engineer and CPU architect. "
    "Your task is to design a CPU architecture ranging from strict IoT microcontrollers to M-Series Superscalar configurations. "
    "You operate by modifying a microarchitectural DAG. "
    "Your primary objective is to maximize the final score derived from PPA heuristics without thermal throttling. "
    "Output ONLY a valid JSON object matching this schema: "
    '{"action_type": "<action>", "target_component": "<id>", "parameter_name": "<name>", '
    '"parameter_value": <float>, "source_node": "<id>", "target_node": "<id>", "reasoning": "<string>"}\n'
    "Valid action types: resize, add_component, remove_component, connect, configure."
)

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_model_action(client: OpenAI, observation: CPUObservation, step: int) -> CPUAction:
    """Ask the LLM for a CPU design action."""
    feedback_truncated = observation.feedback_string[:500] if observation.feedback_string else ""
    user_prompt = (
        f"Step {step}/{observation.max_steps}\n"
        f"Task: {observation.task_name} | Constraints: {observation.task_constraints}\n"
        f"IPC:{observation.current_estimated_ipc:.2f} Throughput:{observation.throughput_gips:.2f}GIPS "
        f"Power:{observation.total_power_mw:.1f}mW Area:{observation.total_area_mm2:.2f}mm² "
        f"PD:{observation.max_power_density:.2f} Thermal:{observation.thermal_celsius:.1f}°C "
        f"PPW:{observation.perf_per_watt:.2f} Throttle:{observation.throttled_factor:.2f}\n"
        f"Components: {observation.active_components}\n"
        f"Feedback: {feedback_truncated}\n"
        "Respond with a single JSON action."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=False,
        )

        raw_output = response.choices[0].message.content or "{}"
        json_string = _extract_json(raw_output)
        return CPUAction(**json.loads(json_string))

    except (json.JSONDecodeError, ValidationError) as error:
        print(f"[DEBUG] Parse error: {error}", flush=True)
        return _fallback_action(f"parse: {error}")
    except Exception as error:
        print(f"[DEBUG] LLM request failed: {error}", flush=True)
        return _fallback_action(f"network: {error}")


def _extract_json(raw: str) -> str:
    content = raw.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        return content[start:end]
    return content


def _fallback_action(reason: str) -> CPUAction:
    return CPUAction(
        action_type="resize",
        target_component="l1d",
        parameter_name="size_kb",
        parameter_value=32.0,
        reasoning=f"Fallback: {reason}"
    )

# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

def run_task(llm_client: OpenAI, task_name: str) -> None:
    """Run one full episode for the given task, emitting [START]/[STEP]/[END] logs."""

    if IMAGE_NAME:
        env_instance = WEGHEnv.from_docker_image(IMAGE_NAME, task=task_name)
    else:
        env_instance = WEGHEnv(base_url=ENV_BASE_URL)

    rewards: List[float] = []
    steps_taken   = 0
    score         = 0.0
    success       = False
    last_error: Optional[str] = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        with env_instance.sync() as env:
            if not IMAGE_NAME:
                result = env.reset(task=task_name)
            else:
                result = env.reset()

            for step in range(1, (result.observation.max_steps or 30) + 1):
                if result.done:
                    break

                observation = result.observation
                action = get_model_action(llm_client, observation, step)

                try:
                    result = env.step(action)
                    last_error = None
                except Exception as exc:
                    last_error = str(exc).replace("\n", " ")
                    result.done = True

                reward = result.reward or 0.0
                done   = result.done
                rewards.append(reward)
                steps_taken = step

                action_str = action.model_dump_json().replace(" ", "")
                log_step(step=step, action=action_str, reward=reward, done=done, error=last_error)

                if done:
                    break

        # Compute score after env context exits cleanly
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(1e-6, min(score, 1 - 1e-6))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

# ---------------------------------------------------------------------------
# Main — iterate all tasks
# ---------------------------------------------------------------------------

def main() -> None:
    if not HF_TOKEN:
        print("[WARNING] HF_TOKEN not set.", file=sys.stderr)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

    for task_name in TASKS:
        run_task(llm_client, task_name)


if __name__ == "__main__":
    main()
