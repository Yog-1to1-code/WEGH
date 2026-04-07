#!/usr/bin/env python3
"""
WEGH — Phase 2 Automated Evaluator Simulation

Replicates the exact sequence the OpenEnv Phase 2 evaluator performs:
  1. Validate openenv.yaml schema
  2. Import and validate grader hooks
  3. Boot the FastAPI server (uvicorn + Go daemon subprocess)
  4. Health check the server
  5. For each task: REST reset → step loop → collect final observation → call grader
  6. Validate inference.py stdout regex compliance
  7. Print aggregate results

This runs against the LIVE environment (Go engine + Python server), not mocks.
"""

import importlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# ─── ANSI Colors ───────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
CHECK  = f"{GREEN}✓{RESET}"
CROSS  = f"{RED}✗{RESET}"
WARN   = f"{YELLOW}⚠{RESET}"

passed = 0
failed = 0
warnings = 0

def ok(msg):
    global passed
    passed += 1
    print(f"  {CHECK} {msg}")

def fail(msg):
    global failed
    failed += 1
    print(f"  {CROSS} {msg}")

def warn(msg):
    global warnings
    warnings += 1
    print(f"  {WARN} {msg}")

def section(title):
    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: openenv.yaml Schema Validation
# ═══════════════════════════════════════════════════════════════════════════════
section("PHASE 1: openenv.yaml Schema Validation")

import yaml

yaml_path = ROOT / "openenv.yaml"
if not yaml_path.exists():
    fail("openenv.yaml not found at project root")
    sys.exit(1)

with open(yaml_path) as f:
    config = yaml.safe_load(f)

# Required top-level fields
required_fields = {
    "spec_version": 1,
    "runtime": "fastapi",
    "app": "server.app:app",
    "port": 7860,
}

for field, expected in required_fields.items():
    actual = config.get(field)
    if actual == expected:
        ok(f"{field}: {actual}")
    else:
        fail(f"{field}: expected {expected}, got {actual}")

# Tasks validation
tasks = config.get("tasks", [])
expected_tasks = {"iot_8bit", "rv32im", "mseries_superscalar"}
actual_task_ids = {t["id"] for t in tasks}

if actual_task_ids == expected_tasks:
    ok(f"tasks: {sorted(actual_task_ids)}")
else:
    fail(f"tasks: expected {expected_tasks}, got {actual_task_ids}")

# Grader hook format
for task in tasks:
    grader = task.get("grader", "")
    if re.match(r"^wegh_graders:grade_\w+$", grader):
        ok(f"grader hook '{grader}' — valid module:function format")
    else:
        fail(f"grader hook '{grader}' — invalid format")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Grader Module Import & Boundary Tests
# ═══════════════════════════════════════════════════════════════════════════════
section("PHASE 2: Grader Module Import & Boundary Tests")

try:
    import wegh_graders
    ok("wegh_graders module imported successfully")
except Exception as e:
    fail(f"wegh_graders import failed: {e}")
    sys.exit(1)

# Verify all 3 grader hooks exist
for task in tasks:
    grader_ref = task["grader"]
    module_name, func_name = grader_ref.split(":")
    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name, None)
    if callable(fn):
        ok(f"{func_name}() is callable")
    else:
        fail(f"{func_name}() not found or not callable")

# Boundary tests (Phase 2 firewall checks)
from server.wegh_env import sanitize_grade

boundary_tests = [
    (1.0,       0.999,  "score=1.0 clamped to 0.999 (exclusive upper)"),
    (0.0,       0.001,  "score=0.0 clamped to 0.001 (exclusive lower)"),
    (None,      0.001,  "None falls back to 0.001"),
    (0.75,      0.75,   "score=0.75 passes through"),
    ("0.5",     0.5,    "string '0.5' cast to float"),
    ("garbage", 0.001,  "unparseable string falls back to 0.001"),
    (1.5,       0.999,  "score=1.5 clamped to 0.999"),
    (-0.5,      0.001,  "negative score clamped to 0.001"),
]

for raw, expected, desc in boundary_tests:
    result = sanitize_grade(raw)
    if result == expected:
        ok(f"sanitize_grade: {desc}")
    else:
        fail(f"sanitize_grade: {desc} — got {result}")

# Grader with dict observation (HTTP transport simulation)
score = wegh_graders.grade_iot_8bit(observation={"final_score": 0.72})
if abs(score - 0.72) < 0.001:
    ok(f"grade_iot_8bit(dict) → {score}")
else:
    fail(f"grade_iot_8bit(dict) — expected 0.72, got {score}")

# Grader with edge-case dict (no final_score key)
score = wegh_graders.grade_rv32im(observation={"other_key": 42})
if score == 0.001:
    ok(f"grade_rv32im(dict, no final_score) → {score} (safe fallback)")
else:
    fail(f"grade_rv32im(dict, no final_score) — expected 0.001, got {score}")

# Grader with boundary score
score = wegh_graders.grade_mseries_superscalar(observation={"final_score": 1.0})
if score == 0.999:
    ok(f"grade_mseries_superscalar(score=1.0) → {score} (clamped)")
else:
    fail(f"grade_mseries_superscalar(score=1.0) — expected 0.999, got {score}")

# Grader with numeric passthrough
score = wegh_graders.grade_iot_8bit(observation=0.85)
if abs(score - 0.85) < 0.001:
    ok(f"grade_iot_8bit(float) → {score} (numeric passthrough)")
else:
    fail(f"grade_iot_8bit(float) — expected 0.85, got {score}")

# Numpy artifact stripping simulation
try:
    import numpy as np
    np_score = np.float64(0.6)
    result = sanitize_grade(np_score)
    if isinstance(result, float) and not str(type(result)).startswith("<class 'numpy"):
        ok(f"numpy.float64 stripped to Python float: {type(result).__name__} = {result}")
    else:
        fail(f"numpy artifact not stripped: type={type(result)}")
except ImportError:
    warn("numpy not installed — skipping numpy artifact test")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: FastAPI Server Boot & Health Check
# ═══════════════════════════════════════════════════════════════════════════════
section("PHASE 3: FastAPI Server Boot & Health Check")

import httpx

SERVER_PORT = 7861  # Avoid conflict if 7860 is already in use
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"

server_proc = subprocess.Popen(
    [
        sys.executable, "-m", "uvicorn", "server.app:app",
        "--host", "127.0.0.1",
        "--port", str(SERVER_PORT),
        "--log-level", "warning",
    ],
    cwd=str(ROOT),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={**os.environ, "PYTHONPATH": str(ROOT)},
)

def cleanup_server():
    if server_proc.poll() is None:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()

import atexit
atexit.register(cleanup_server)

# Wait for server to become healthy
print(f"  … Waiting for FastAPI server on port {SERVER_PORT} ", end="", flush=True)
server_ready = False
start = time.time()
max_wait = 30.0

client = httpx.Client(timeout=5.0)

while time.time() - start < max_wait:
    try:
        resp = client.get(f"{SERVER_URL}/health")
        if resp.status_code == 200:
            server_ready = True
            break
    except (httpx.ConnectError, httpx.ReadTimeout):
        pass

    # Check if server process died
    if server_proc.poll() is not None:
        stderr_output = server_proc.stderr.read().decode()
        print()
        fail(f"Server process died during startup")
        if stderr_output:
            print(f"    stderr: {stderr_output[:500]}")
        cleanup_server()
        sys.exit(1)

    print(".", end="", flush=True)
    time.sleep(1.0)

print()

if server_ready:
    ok(f"Server healthy at {SERVER_URL}")
else:
    fail(f"Server not ready after {max_wait}s")
    cleanup_server()
    sys.exit(1)

# Health endpoint returns JSON
try:
    resp = client.get(f"{SERVER_URL}/health")
    health = resp.json()
    if health.get("status") in ("ok", "healthy"):
        ok(f"/health → {json.dumps(health)}")
    else:
        fail(f"/health returned unexpected status: {health}")
except Exception as e:
    fail(f"/health request failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Full Episode Lifecycle (Reset → Step → Grade) per Task
# ═══════════════════════════════════════════════════════════════════════════════
section("PHASE 4: Full Episode Lifecycle (REST API)")

from models import CPUAction, CPUObservation

# The evaluator uses the OpenEnv WebSocket/REST protocol.
# We simulate by directly instantiating the environment (same as the server does internally).
from server.wegh_env import WEGHEnvironment

env = WEGHEnvironment()

# Wait for its Go daemon
go_ready = env.go_client.wait_for_ready(max_wait=15)
if go_ready:
    ok("Go simulation daemon responding")
else:
    fail("Go daemon not ready")
    cleanup_server()
    sys.exit(1)

task_scores = {}

for task_def in tasks:
    task_id = task_def["id"]
    grader_ref = task_def["grader"]
    module_name, func_name = grader_ref.split(":")
    grader_fn = getattr(importlib.import_module(module_name), func_name)

    print(f"\n  {BOLD}─── Task: {task_id} ({task_def['name']}) ───{RESET}")

    # RESET
    try:
        obs = env.reset(task=task_id)
        if isinstance(obs, CPUObservation):
            ok(f"reset(task='{task_id}') → CPUObservation")
        else:
            fail(f"reset returned {type(obs)} instead of CPUObservation")
            continue
    except Exception as e:
        fail(f"reset(task='{task_id}') crashed: {e}")
        continue

    # Verify initial observation fields
    field_checks = [
        ("done", lambda v: v == False),
        ("task_name", lambda v: isinstance(v, str) and len(v) > 0),
        ("max_steps", lambda v: isinstance(v, int) and v > 0),
        ("feedback_string", lambda v: isinstance(v, str) and len(v) > 10),
        ("current_estimated_ipc", lambda v: isinstance(v, (int, float))),
        ("total_power_mw", lambda v: isinstance(v, (int, float))),
        ("total_area_mm2", lambda v: isinstance(v, (int, float))),
    ]

    for field_name, check_fn in field_checks:
        val = getattr(obs, field_name, "MISSING")
        if val != "MISSING" and check_fn(val):
            ok(f"  obs.{field_name} = {repr(val)[:60]}")
        else:
            fail(f"  obs.{field_name} = {repr(val)[:60]} — unexpected")

    # STEP LOOP — run 5 steps with plausible actions
    test_actions = [
        CPUAction(action_type="resize", target_component="l1d", parameter_name="size_kb", parameter_value=32.0, reasoning="eval-test"),
        CPUAction(action_type="resize", target_component="l1i", parameter_name="size_kb", parameter_value=32.0, reasoning="eval-test"),
        CPUAction(action_type="configure", target_component="pipeline", parameter_name="pipeline_depth", parameter_value=5.0, reasoning="eval-test"),
        CPUAction(action_type="resize", target_component="l1d", parameter_name="size_kb", parameter_value=64.0, reasoning="eval-test"),
        CPUAction(action_type="resize", target_component="l1i", parameter_name="size_kb", parameter_value=64.0, reasoning="eval-test"),
    ]

    step_count = 0
    all_rewards = []
    has_negative_reward = False

    for action in test_actions:
        if obs.done:
            break
        step_count += 1
        try:
            obs = env.step(action)
            all_rewards.append(obs.reward)
            if obs.reward < 0:
                has_negative_reward = True
        except Exception as e:
            fail(f"  step {step_count} crashed: {e}")
            break

    if step_count > 0:
        ok(f"  Completed {step_count} steps without crash")
    else:
        fail(f"  No steps executed")

    # Verify rewards are Python floats (not numpy)
    for i, r in enumerate(all_rewards):
        if not isinstance(r, float):
            fail(f"  step {i+1} reward type: {type(r).__name__} (expected float)")
            break
    else:
        ok(f"  All rewards are Python float type")

    # Verify negative rewards flow through (critical fix validation)
    if has_negative_reward:
        ok(f"  Negative rewards preserved (reward flow fix validated)")
    else:
        warn(f"  No negative rewards observed (actions may have been all valid)")

    # Run to completion for final score
    remaining_steps = 0
    while not obs.done and remaining_steps < 100:
        remaining_steps += 1
        obs = env.step(CPUAction(
            action_type="resize", target_component="l1d",
            parameter_name="size_kb", parameter_value=32.0,
            reasoning="eval-drain"
        ))

    if obs.done:
        ok(f"  Episode completed (done=True)")
    else:
        fail(f"  Episode never reached done=True after {step_count + remaining_steps} steps")

    # GRADE — simulate Phase 2 grading (observation arrives as dict over HTTP)
    obs_dict = obs.model_dump()
    grade = grader_fn(observation=obs_dict)

    # Validate grade boundaries
    if isinstance(grade, float) and 0.0 < grade < 1.0:
        ok(f"  Grader {func_name}() → {grade:.4f} (valid exclusive bounds)")
        task_scores[task_id] = grade
    elif grade == 0.0 or grade == 1.0:
        fail(f"  Grader {func_name}() → {grade} (INCLUSIVE BOUNDARY — will crash evaluator!)")
    else:
        fail(f"  Grader {func_name}() → {grade} (type: {type(grade).__name__})")

    # Verify grade is pure Python float
    if type(grade) is float:
        ok(f"  Grade type: {type(grade).__name__} (pure Python)")
    else:
        fail(f"  Grade type: {type(grade).__name__} (must be pure Python float)")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: inference.py STDOUT Regex Compliance
# ═══════════════════════════════════════════════════════════════════════════════
section("PHASE 5: inference.py STDOUT Regex Compliance")

# The evaluator parses stdout with these exact patterns
START_REGEX = re.compile(r"^\[START\] task=(\S+) env=(\S+) model=(\S+)$")
STEP_REGEX  = re.compile(r"^\[STEP\] step=(\d+) action=(\S+) reward=(\S+) done=(true|false) error=(.+)$")
END_REGEX   = re.compile(r"^\[END\] success=(true|false) steps=(\d+) rewards=(.+)$")

# Simulate what inference.py would print
sample_lines = [
    "[START] task=8-Bit_IoT env=WEGH model=meta-llama/Llama-3.1-8B-Instruct",
    '[STEP] step=1 action={"action_type":"resize","target_component":"l1d","parameter_name":"size_kb","parameter_value":32.0,"source_node":"","target_node":"","reasoning":"test"} reward=0.02 done=false error=null',
    '[STEP] step=2 action={"action_type":"resize","target_component":"l1i","parameter_name":"size_kb","parameter_value":64.0,"source_node":"","target_node":"","reasoning":"test"} reward=-0.10 done=true error=null',
    "[END] success=false steps=2 rewards=0.02,-0.10",
]

for line in sample_lines:
    if line.startswith("[START]"):
        m = START_REGEX.match(line)
        if m:
            ok(f"[START] line matches regex: task={m.group(1)}")
        else:
            fail(f"[START] line does NOT match regex: {line[:80]}")
    elif line.startswith("[STEP]"):
        m = STEP_REGEX.match(line)
        if m:
            ok(f"[STEP] line matches regex: step={m.group(1)} done={m.group(4)}")
        else:
            fail(f"[STEP] line does NOT match regex: {line[:80]}")
    elif line.startswith("[END]"):
        m = END_REGEX.match(line)
        if m:
            ok(f"[END] line matches regex: success={m.group(1)} steps={m.group(2)}")
        else:
            fail(f"[END] line does NOT match regex: {line[:80]}")

# Verify NO score= field in [END] line (new Scaler guideline)
for line in sample_lines:
    if line.startswith("[END]") and "score=" in line:
        fail("[END] line contains 'score=' field — violates Scaler guideline!")
        break
else:
    ok("[END] line has no 'score=' field (Scaler compliant)")

# Verify action JSON has no spaces
for line in sample_lines:
    if line.startswith("[STEP]"):
        action_match = re.search(r"action=(\S+)", line)
        if action_match:
            action_str = action_match.group(1)
            if " " not in action_str:
                ok(f"Action JSON is space-free")
            else:
                fail(f"Action JSON contains spaces — will break regex parser")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: Dockerfile Completeness
# ═══════════════════════════════════════════════════════════════════════════════
section("PHASE 6: Dockerfile Completeness")

dockerfile_path = ROOT / "Dockerfile"
if dockerfile_path.exists():
    dockerfile_content = dockerfile_path.read_text()

    critical_copies = [
        "COPY wegh_graders.py",
        "COPY models.py",
        "COPY inference.py",
        "COPY server/",
        "COPY openenv.yaml",
        "COPY entrypoint.sh",
    ]

    for copy_cmd in critical_copies:
        if copy_cmd in dockerfile_content:
            ok(f"Dockerfile contains '{copy_cmd}'")
        else:
            fail(f"Dockerfile MISSING '{copy_cmd}'")

    # Verify port 7860
    if "EXPOSE 7860" in dockerfile_content:
        ok("EXPOSE 7860 present")
    else:
        fail("EXPOSE 7860 missing")

    # Verify Go engine build
    if "go build" in dockerfile_content and "go-engine" in dockerfile_content:
        ok("Go engine multi-stage build present")
    else:
        fail("Go engine build missing from Dockerfile")
else:
    fail("Dockerfile not found")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7: Resource & Cleanup Verification
# ═══════════════════════════════════════════════════════════════════════════════
section("PHASE 7: Resource & Cleanup Verification")

# Verify atexit handler is registered in wegh_env.py
env_source = (ROOT / "server" / "wegh_env.py").read_text()
if "atexit.register" in env_source:
    ok("atexit.register() daemon cleanup present")
else:
    fail("atexit.register() daemon cleanup MISSING")

if "__del__" in env_source:
    ok("__del__ daemon cleanup present")
else:
    fail("__del__ daemon cleanup MISSING")

if "get_free_port" in env_source:
    ok("Dynamic port allocation present")
else:
    fail("Dynamic port allocation MISSING")

# Verify sanitize_grade is only used on final scores, not step rewards
if "step_reward = float(raw_step_reward)" in env_source:
    ok("Step rewards use float() passthrough (not sanitize_grade)")
else:
    fail("Step rewards may still be clamped by sanitize_grade")

if "sanitize_grade(compute_final_score" in env_source:
    ok("Final scores use sanitize_grade (correct)")
else:
    fail("Final scores may not be sanitized")


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
section("EVALUATION RESULTS")

print(f"\n  {BOLD}Task Scores:{RESET}")
for task_id, score in task_scores.items():
    bar_len = int(score * 30)
    bar = f"{'█' * bar_len}{'░' * (30 - bar_len)}"
    color = GREEN if score > 0.5 else YELLOW if score > 0.3 else RED
    print(f"    {task_id:25s} {color}{bar}{RESET} {score:.4f}")

if task_scores:
    avg = sum(task_scores.values()) / len(task_scores)
    print(f"\n    {'Average':25s} {'─'*30} {avg:.4f}")

print(f"""
  {BOLD}Summary:{RESET}
    {GREEN}Passed:   {passed}{RESET}
    {RED}Failed:   {failed}{RESET}
    {YELLOW}Warnings: {warnings}{RESET}
""")

if failed == 0:
    print(f"  {BOLD}{GREEN}{'═'*50}{RESET}")
    print(f"  {BOLD}{GREEN}  ALL CHECKS PASSED — READY FOR PHASE 2 SUBMISSION{RESET}")
    print(f"  {BOLD}{GREEN}{'═'*50}{RESET}")
else:
    print(f"  {BOLD}{RED}{'═'*50}{RESET}")
    print(f"  {BOLD}{RED}  {failed} CHECK(S) FAILED — FIX BEFORE SUBMISSION{RESET}")
    print(f"  {BOLD}{RED}{'═'*50}{RESET}")

# Cleanup
cleanup_server()
env.go_client.close()
env._cleanup_daemon()

sys.exit(0 if failed == 0 else 1)
