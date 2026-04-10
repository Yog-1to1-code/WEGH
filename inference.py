#!/usr/bin/env python3
"""
inference.py — WEGH OpenEnv Agent
==================================
Runs an LLM agent through all 3 CPU design tasks.
Features:
- Heuristic fallback policy for API failures/rate limits
- Action reuse to conserve LLM credits (--llm-every N)
- Fast mode for heuristic-only evaluation (--fast-mode)
- Deterministic seeding for reproducible runs
- Automatic server discovery and startup
- Baseline scores output to JSON

Required environment variables:
    API_BASE_URL      LLM API endpoint
    MODEL_NAME        Model identifier
    HF_TOKEN          HuggingFace / API key

Stdout format (must not deviate):
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
BENCHMARK    = "WEGH"

TASKS = [
    {"name": "iot_8bit",            "id": 0, "max_steps": 20},
    {"name": "rv32im",              "id": 1, "max_steps": 30},
    {"name": "mseries_superscalar", "id": 2, "max_steps": 40},
]

TEMPERATURE    = 0.1
MAX_TOKENS     = 200
REWARD_MIN     = 0.10
REWARD_MAX     = 0.90

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
# Heuristic Fallback Policies
# ---------------------------------------------------------------------------

def heuristic_iot(step: int, observation: Dict) -> Dict:
    """Rule-based IoT design policy — ultra-low power focus."""
    power = observation.get("total_power_mw", 50)
    ipc = observation.get("current_estimated_ipc", 0)
    
    # Progressive strategy: minimize power first, then tune performance
    if step <= 3:
        return {"action_type": "resize", "target_component": "pmu", "parameter_name": "voltage", "parameter_value": 0.7, "reasoning": "Lower voltage for power savings"}
    elif step <= 5:
        return {"action_type": "resize", "target_component": "pmu", "parameter_name": "clock_mhz", "parameter_value": 8, "reasoning": "Low clock for IoT"}
    elif step <= 8 and power > 30:
        return {"action_type": "resize", "target_component": "sram", "parameter_name": "size_kb", "parameter_value": 1, "reasoning": "Minimize SRAM to save power"}
    elif step <= 10:
        return {"action_type": "resize", "target_component": "gpio", "parameter_name": "pin_count", "parameter_value": 4, "reasoning": "Reduce GPIO pins"}
    elif power > 45:
        return {"action_type": "resize", "target_component": "pmu", "parameter_name": "clock_mhz", "parameter_value": 4, "reasoning": "Emergency power reduction"}
    elif ipc < 0.5:
        return {"action_type": "resize", "target_component": "alu", "parameter_name": "count", "parameter_value": 1, "reasoning": "Ensure minimum compute"}
    else:
        return {"action_type": "resize", "target_component": "pmu", "parameter_name": "voltage",
                "parameter_value": max(0.5, 0.7 - step * 0.01), "reasoning": "Incremental voltage reduction"}


def heuristic_rv32im(step: int, observation: Dict) -> Dict:
    """Rule-based RV32IM design policy — maximize IPC within area."""
    area = observation.get("total_area_mm2", 10)
    ipc = observation.get("current_estimated_ipc", 0)
    
    strategies = [
        {"action_type": "resize", "target_component": "l1d", "parameter_name": "size_kb", "parameter_value": 32, "reasoning": "Increase L1D for better hit rate"},
        {"action_type": "resize", "target_component": "l1i", "parameter_name": "size_kb", "parameter_value": 32, "reasoning": "Increase L1I cache"},
        {"action_type": "resize", "target_component": "bp", "parameter_name": "type", "parameter_value": 2, "reasoning": "Use gshare predictor"},
        {"action_type": "resize", "target_component": "bp", "parameter_name": "btb_entries", "parameter_value": 256, "reasoning": "Increase BTB"},
        {"action_type": "resize", "target_component": "l1d", "parameter_name": "associativity", "parameter_value": 4, "reasoning": "Better L1D associativity"},
        {"action_type": "resize", "target_component": "l1i", "parameter_name": "associativity", "parameter_value": 4, "reasoning": "Better L1I associativity"},
        {"action_type": "resize", "target_component": "alu", "parameter_name": "count", "parameter_value": 2, "reasoning": "More ALUs for ILP"},
        {"action_type": "resize", "target_component": "decode", "parameter_name": "width", "parameter_value": 2, "reasoning": "Wider decode for IPC"},
        {"action_type": "resize", "target_component": "load_unit", "parameter_name": "count", "parameter_value": 2, "reasoning": "More load units"},
        {"action_type": "resize", "target_component": "muldiv", "parameter_name": "count", "parameter_value": 1, "reasoning": "Ensure MulDiv present"},
    ]

    # If area is getting tight, back off on cache sizes
    if area > 8.5:
        return {"action_type": "resize", "target_component": "l1d", "parameter_name": "size_kb",
                "parameter_value": 16, "reasoning": "Reduce cache for area compliance"}
    
    idx = min(step - 1, len(strategies) - 1)
    return strategies[idx]


def heuristic_mseries(step: int, observation: Dict) -> Dict:
    """Rule-based M-Series design policy — balance P/E cores and thermals."""
    pd = observation.get("max_power_density", 0)
    throughput = observation.get("throughput_gips", 0)
    throttle = observation.get("throttled_factor", 1.0)
    
    strategies = [
        {"action_type": "resize", "target_component": "pcore", "parameter_name": "count", "parameter_value": 4, "reasoning": "Set P-core count"},
        {"action_type": "resize", "target_component": "ecore", "parameter_name": "count", "parameter_value": 4, "reasoning": "Set E-core count"},
        {"action_type": "resize", "target_component": "l2", "parameter_name": "size_kb", "parameter_value": 2048, "reasoning": "Large L2 for hit rate"},
        {"action_type": "resize", "target_component": "l3", "parameter_name": "size_mb", "parameter_value": 16, "reasoning": "L3 for multi-core sharing"},
        {"action_type": "resize", "target_component": "bp", "parameter_name": "type", "parameter_value": 3, "reasoning": "TAGE predictor for accuracy"},
        {"action_type": "resize", "target_component": "rob", "parameter_name": "entries", "parameter_value": 256, "reasoning": "Large ROB for OoO"},
        {"action_type": "resize", "target_component": "rs", "parameter_name": "entries", "parameter_value": 96, "reasoning": "Reservation stations"},
        {"action_type": "resize", "target_component": "l1d", "parameter_name": "size_kb", "parameter_value": 64, "reasoning": "Per-core L1D"},
        {"action_type": "resize", "target_component": "l1i", "parameter_name": "size_kb", "parameter_value": 64, "reasoning": "Per-core L1I"},
        {"action_type": "resize", "target_component": "pcore", "parameter_name": "issue_width", "parameter_value": 6, "reasoning": "Wide issue P-core"},
        {"action_type": "resize", "target_component": "pcore", "parameter_name": "pipeline_depth", "parameter_value": 14, "reasoning": "Deep P-core pipeline"},
        {"action_type": "resize", "target_component": "p_simd", "parameter_name": "count", "parameter_value": 2, "reasoning": "SIMD units for throughput"},
        {"action_type": "resize", "target_component": "p_fpu", "parameter_name": "count", "parameter_value": 2, "reasoning": "FP units"},
        {"action_type": "resize", "target_component": "p_alu", "parameter_name": "count", "parameter_value": 4, "reasoning": "Integer ALUs"},
        {"action_type": "resize", "target_component": "noc", "parameter_name": "type", "parameter_value": 2, "reasoning": "Mesh NoC"},
        {"action_type": "resize", "target_component": "memctrl", "parameter_name": "channels", "parameter_value": 4, "reasoning": "Memory bandwidth"},
    ]

    # Thermal management: if power density too high, reduce clocks
    if pd > 1.3 or throttle < 0.8:
        return {"action_type": "resize", "target_component": "pcore", "parameter_name": "clock_ghz",
                "parameter_value": 3.0, "reasoning": "Reduce P-core clock to manage thermals"}
    
    if step > len(strategies):
        # Fine-tuning phase
        if pd > 1.0:
            return {"action_type": "resize", "target_component": "pcore", "parameter_name": "voltage",
                    "parameter_value": 0.9, "reasoning": "Reduce voltage for thermal margin"}
        else:
            return {"action_type": "resize", "target_component": "ecore", "parameter_name": "count",
                    "parameter_value": 6, "reasoning": "Add E-cores for efficiency"}
    
    idx = min(step - 1, len(strategies) - 1)
    return strategies[idx]


HEURISTIC_POLICIES = {
    0: heuristic_iot,
    1: heuristic_rv32im,
    2: heuristic_mseries,
}

# ---------------------------------------------------------------------------
# Environment Client (direct HTTP — no openenv-core dependency)
# ---------------------------------------------------------------------------

class EnvClient:
    """Direct HTTP client for the WEGH Go engine."""
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)
    
    def health(self) -> bool:
        try:
            r = self.client.get("/health")
            return r.status_code == 200
        except Exception:
            return False
    
    def reset(self, task: str, seed: Optional[int] = None) -> Dict:
        payload: Dict[str, Any] = {"task": task}
        if seed is not None:
            payload["seed"] = seed
        r = self.client.post("/reset", json=payload)
        r.raise_for_status()
        return r.json()
    
    def step(self, action: Dict) -> Dict:
        r = self.client.post("/step", json=action)
        r.raise_for_status()
        return r.json()
    
    def grade(self) -> Dict:
        r = self.client.get("/grade")
        r.raise_for_status()
        return r.json()
    
    def close(self):
        self.client.close()

# ---------------------------------------------------------------------------
# Server Discovery & Auto-Start
# ---------------------------------------------------------------------------

def start_environment_server(env_url: str) -> Optional[subprocess.Popen]:
    """Try to discover or start the Go server."""
    # Check if already running
    client = EnvClient(env_url)
    if client.health():
        print(f"[INFO] Server already running at {env_url}", flush=True)
        return None
    
    # Try to find and start Go binary
    binary_paths = [
        "./go-engine",
        "./engine/go-engine",
        "/app/go-engine",
    ]
    
    for path in binary_paths:
        if os.path.exists(path):
            port = env_url.split(":")[-1].split("/")[0]
            print(f"[INFO] Starting Go engine from {path}...", flush=True)
            proc = subprocess.Popen(
                [path, f"--bind=0.0.0.0:{port}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Wait for startup
            for _ in range(20):
                if client.health():
                    print(f"[INFO] Go engine ready at {env_url}", flush=True)
                    return proc
                time.sleep(0.5)
            print(f"[WARN] Go engine failed to start from {path}", flush=True)
            proc.kill()
    
    print(f"[WARN] Could not start Go engine. Ensure it's running at {env_url}", flush=True)
    return None

# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

class LLMAgent:
    """LLM-based CPU design agent with heuristic fallback."""
    
    def __init__(self, llm_client: Optional[OpenAI], llm_every: int = 1, fast_mode: bool = False):
        self.llm_client = llm_client
        self.llm_every = llm_every
        self.fast_mode = fast_mode
        self.use_heuristic = fast_mode
        self.last_llm_action: Optional[Dict] = None
        self.credits_exhausted = False
    
    def choose_action(self, observation: Dict, task_id: int, step: int) -> Dict:
        """Choose an action — LLM with heuristic fallback."""
        
        # Fast mode or credits exhausted → heuristic only
        if self.use_heuristic or self.credits_exhausted:
            return self._heuristic_action(task_id, step, observation)
        
        # Action reuse: only call LLM every N steps
        if self.last_llm_action is not None and step % self.llm_every != 1:
            return self.last_llm_action
        
        # Try LLM
        try:
            action = self._llm_action(observation, step)
            self.last_llm_action = action
            return action
        except Exception as e:
            error_str = str(e)
            # Detect rate limit / credit exhaustion
            if "402" in error_str or "429" in error_str or "quota" in error_str.lower():
                print(f"[WARN] LLM credits exhausted, switching to heuristic mode", flush=True)
                self.credits_exhausted = True
            else:
                print(f"[DEBUG] LLM error, using heuristic: {e}", flush=True)
            return self._heuristic_action(task_id, step, observation)
    
    def _llm_action(self, observation: Dict, step: int) -> Dict:
        """Get action from LLM."""
        max_steps = observation.get("max_steps", 30)
        feedback = str(observation.get("feedback_string", ""))[:500]
        
        user_prompt = (
            f"Step {step}/{max_steps}\n"
            f"Task: {observation.get('task_name', '')} | Constraints: {observation.get('task_constraints', '')}\n"
            f"IPC:{observation.get('current_estimated_ipc', 0):.2f} "
            f"Throughput:{observation.get('throughput_gips', 0):.2f}GIPS "
            f"Power:{observation.get('total_power_mw', 0):.1f}mW "
            f"Area:{observation.get('total_area_mm2', 0):.2f}mm² "
            f"PD:{observation.get('max_power_density', 0):.2f} "
            f"Thermal:{observation.get('thermal_celsius', 0):.1f}°C "
            f"PPW:{observation.get('perf_per_watt', 0):.2f} "
            f"Throttle:{observation.get('throttled_factor', 1.0):.2f}\n"
            f"Components: {observation.get('active_components', '[]')}\n"
            f"Feedback: {feedback}\n"
            "Respond with a single JSON action."
        )
        
        response = self.llm_client.chat.completions.create(
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
        action_data = json.loads(json_string)
        
        # Validate required fields
        if "action_type" not in action_data:
            raise ValueError("Missing action_type")
        
        return {
            "action_type": action_data.get("action_type", "resize"),
            "target_component": action_data.get("target_component", "l1d"),
            "parameter_name": action_data.get("parameter_name", "size_kb"),
            "parameter_value": float(action_data.get("parameter_value", 32)),
            "source_node": action_data.get("source_node", ""),
            "target_node": action_data.get("target_node", ""),
            "reasoning": action_data.get("reasoning", ""),
        }
    
    def _heuristic_action(self, task_id: int, step: int, observation: Dict) -> Dict:
        """Get action from heuristic policy."""
        policy_fn = HEURISTIC_POLICIES.get(task_id, heuristic_rv32im)
        action = policy_fn(step, observation)
        action.setdefault("source_node", "")
        action.setdefault("target_node", "")
        action.setdefault("reasoning", f"heuristic_t{task_id}_s{step}")
        return action


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

# ---------------------------------------------------------------------------
# Reward Normalization
# ---------------------------------------------------------------------------

def normalize_reward(reward: float) -> float:
    """Normalize reward to [REWARD_MIN, REWARD_MAX]."""
    # Typical raw reward range is roughly [-0.2, 0.2]
    RAW_MIN, RAW_MAX = -0.3, 0.3
    if RAW_MAX == RAW_MIN:
        return (REWARD_MIN + REWARD_MAX) / 2.0
    normalized = (reward - RAW_MIN) / (RAW_MAX - RAW_MIN)
    return REWARD_MIN + normalized * (REWARD_MAX - REWARD_MIN)

# ---------------------------------------------------------------------------
# Single Task Runner
# ---------------------------------------------------------------------------

def run_task(
    agent: LLMAgent,
    env: EnvClient,
    task: Dict,
    seed: int,
    verbose: bool = False,
) -> Dict:
    """Run one full episode. Returns result dict."""
    task_name = task["name"]
    task_id = task["id"]
    max_steps = task["max_steps"]
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_resp = env.reset(task=task_name, seed=seed)
        obs = reset_resp.get("observation", {})
        if not obs:
            # Fallback: extract obs from reset response
            obs = {"task_id": task_id, "task_name": task_name, "max_steps": max_steps}

        for step in range(1, max_steps + 1):
            # Get action from agent
            action = agent.choose_action(obs, task_id, step)

            # Send action to environment
            try:
                # Map to Go engine format
                step_payload = {
                    "action": {
                        "type": action["action_type"],
                        "component": action["target_component"],
                        "param_name": action["parameter_name"],
                        "value": float(action.get("parameter_value", 0)),
                        "source_node": action.get("source_node", ""),
                        "target_node": action.get("target_node", ""),
                    }
                }
                step_resp = env.step(step_payload)
                last_error = None
            except Exception as exc:
                last_error = str(exc).replace("\n", " ")[:100]
                step_resp = {"reward": 0, "done": True, "observation": obs}

            reward = step_resp.get("reward", 0.0)
            done = step_resp.get("done", False)
            obs = step_resp.get("observation", obs)

            normalized_reward = normalize_reward(reward)
            rewards.append(normalized_reward)
            steps_taken = step

            action_str = json.dumps(action, separators=(",", ":"))
            log_step(step=step, action=action_str, reward=normalized_reward, done=done, error=last_error)

            if verbose:
                print(f"  IPC={obs.get('current_estimated_ipc', 0):.3f} "
                      f"Power={obs.get('total_power_mw', 0):.1f}mW "
                      f"Area={obs.get('total_area_mm2', 0):.3f}mm²", flush=True)

            if done:
                break

        # Query grade from server
        try:
            grade = env.grade()
            score = grade.get("score", 0.001)
        except Exception:
            score = sum(rewards) / len(rewards) if rewards else 0.001

        score = max(0.001, min(score, 0.999))
        success = score >= 0.3

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task_name,
        "task_id": task_id,
        "score": score,
        "steps": steps_taken,
        "success": success,
        "rewards": rewards,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="WEGH OpenEnv Inference Agent")
    parser.add_argument("--env-url", type=str, default="http://localhost:7860", help="Environment server URL")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per task")
    parser.add_argument("--llm-every", type=int, default=1, help="Call LLM every N steps (reuse action between)")
    parser.add_argument("--fast-mode", action="store_true", help="Heuristic-only mode (no LLM)")
    parser.add_argument("--output", type=str, default="", help="Save baseline scores to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-step metrics")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    if not HF_TOKEN and not args.fast_mode:
        print("[WARNING] HF_TOKEN not set. Use --fast-mode for heuristic-only.", file=sys.stderr)

    # Start server if needed
    server_proc = start_environment_server(args.env_url)

    # Create LLM client
    llm_client = None
    if not args.fast_mode:
        llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

    agent = LLMAgent(
        llm_client=llm_client,
        llm_every=args.llm_every,
        fast_mode=args.fast_mode,
    )
    
    env = EnvClient(args.env_url)

    # Run all tasks
    all_results = []
    for ep in range(args.episodes):
        for task in TASKS:
            seed = args.seed + task["id"] * 100 + ep
            result = run_task(agent, env, task, seed=seed, verbose=args.verbose)
            all_results.append(result)

    env.close()

    # Compute summary
    per_task_avg = {}
    for task in TASKS:
        task_results = [r for r in all_results if r["task"] == task["name"]]
        if task_results:
            avg_score = sum(r["score"] for r in task_results) / len(task_results)
            per_task_avg[task["name"]] = round(avg_score, 4)

    overall_avg = sum(per_task_avg.values()) / len(per_task_avg) if per_task_avg else 0.0

    print(f"\n{'='*50}", flush=True)
    print(f"WEGH Baseline Scores", flush=True)
    for task_name, avg in per_task_avg.items():
        print(f"  {task_name}: {avg:.4f}", flush=True)
    print(f"  Overall: {overall_avg:.4f}", flush=True)
    print(f"{'='*50}", flush=True)

    # Save to JSON
    if args.output:
        output_data = {
            "model": MODEL_NAME,
            "mode": "heuristic" if args.fast_mode else "llm",
            "episodes_per_task": args.episodes,
            "per_task": per_task_avg,
            "overall": round(overall_avg, 4),
            "results": all_results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"[INFO] Scores saved to {args.output}", flush=True)

    # Cleanup
    if server_proc:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server_proc.kill()


if __name__ == "__main__":
    main()
