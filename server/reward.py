# WEGH — Dense Reward Shaping
# Implements the "Inverse Specification Reward" — fractional positive rewards for
# valid actions, penalties for violations, and [0.0, 1.0] bounded final scores.

import math
from typing import Dict, Any


def compute_step_reward(
    metrics: Dict[str, Any],
    prev_metrics: Dict[str, Any],
    action_valid: bool,
    validation_errors: list,
    task_id: int,
    task_constraints: Dict[str, float],
) -> float:
    """Compute dense per-step reward.
    
    Returns a reward that fluctuates based on the architectural efficacy of the action.
    NOT bounded to [0,1] — that's only for the final score.
    """
    reward = 0.0
    
    # === Validity Bonus/Penalty ===
    if not action_valid:
        reward -= 0.15  # Invalid action penalty
    else:
        reward += 0.02  # Valid action bonus
    
    if validation_errors:
        reward -= 0.05 * len(validation_errors)  # Structural issues
    
    # === Constraint Compliance ===
    if task_id == 0:  # IoT: Power constraint
        power = metrics.get("total_power_mw", 999)
        max_power = task_constraints.get("max_power_mw", 50)
        if power <= max_power:
            # Positive reward proportional to power headroom
            headroom = (max_power - power) / max_power
            reward += 0.03 * headroom
        else:
            # Penalty proportional to overbudget
            overshoot = (power - max_power) / max_power
            reward -= 0.10 * min(overshoot, 2.0)
    
    elif task_id == 1:  # RV32IM: Area constraint
        area = metrics.get("total_area_mm2", 999)
        max_area = task_constraints.get("max_area_mm2", 10)
        ipc = metrics.get("ipc", 0)
        
        if area <= max_area:
            headroom = (max_area - area) / max_area
            reward += 0.02 * headroom
        else:
            overshoot = (area - max_area) / max_area
            reward -= 0.10 * min(overshoot, 2.0)
        
        # IPC improvement bonus
        prev_ipc = prev_metrics.get("ipc", 0)
        if ipc > prev_ipc:
            reward += 0.05 * min((ipc - prev_ipc) / max(prev_ipc, 0.1), 1.0)
    
    elif task_id == 2:  # M-Series: Thermal constraint
        pd = metrics.get("max_power_density", 999)
        max_pd = task_constraints.get("max_power_density", 1.5)
        throttle = metrics.get("throttled_factor", 1.0)
        throughput = metrics.get("throughput_gips", 0)
        prev_throughput = prev_metrics.get("throughput_gips", 0)
        prev_pd = prev_metrics.get("max_power_density", 999)
        
        # Thermal compliance
        if pd <= max_pd:
            headroom = (max_pd - pd) / max_pd
            reward += 0.04 * headroom
        else:
            # Logarithmic penalty — steep at first but doesn't overwhelm at extreme values
            overshoot_ratio = pd / max_pd
            reward -= 0.06 * min(math.log2(overshoot_ratio + 1), 3.0)
        
        # Throttling penalty (mild — agent needs room to explore)
        if throttle < 1.0:
            reward -= 0.03 * (1.0 - throttle)
        
        # Throughput improvement (always rewarded to guide exploration)
        if throughput > prev_throughput and prev_throughput > 0:
            reward += 0.05 * min((throughput - prev_throughput) / prev_throughput, 1.0)
        
        # Power density improvement (critical: reward cooling moves)
        if pd < prev_pd and prev_pd > 0:
            reward += 0.04 * min((prev_pd - pd) / max(prev_pd, 0.01), 1.0)
    
    # === Improvement Bonus (all tasks) ===
    # Pareto-like: reward if improved on multiple fronts
    improvements = 0
    for key in ["ipc", "perf_per_watt"]:
        if metrics.get(key, 0) > prev_metrics.get(key, 0):
            improvements += 1
    for key in ["total_power_mw", "total_area_mm2", "max_power_density"]:
        if metrics.get(key, 999) < prev_metrics.get(key, 999):
            improvements += 1
    if improvements >= 3:
        reward += 0.05  # Multi-objective improvement bonus
    
    return round(reward, 4)


def compute_final_score(metrics: Dict[str, Any], task_id: int, task_constraints: Dict[str, float]) -> float:
    """Compute final episode score, STRICTLY bounded to [0.0, 1.0].
    
    This is the score printed when done=True. Static scores are forbidden.
    """
    if task_id == 0:
        # IoT: Score = f(power compliance, throughput)
        power = metrics.get("total_power_mw", 999)
        max_power = task_constraints.get("max_power_mw", 50)
        ipc = metrics.get("ipc", 0)
        
        if power > max_power:
            power_score = max(0, 1.0 - (power - max_power) / max_power)
            power_score *= 0.3  # Heavy penalty for exceeding budget
        else:
            power_score = 1.0 - (power / max_power) * 0.3  # Reward low power
        
        throughput_score = min(1.0, ipc / 0.8)
        score = 0.6 * power_score + 0.4 * throughput_score
    
    elif task_id == 1:
        # RV32IM: Score = f(IPC, area compliance)
        area = metrics.get("total_area_mm2", 999)
        max_area = task_constraints.get("max_area_mm2", 10)
        ipc = metrics.get("ipc", 0)
        
        if area > max_area:
            area_score = max(0, 1.0 - (area - max_area) / max_area)
            area_score *= 0.3
        else:
            area_score = 1.0 - (area / max_area) * 0.2
        
        ipc_score = min(1.0, ipc / 1.5)
        score = 0.5 * ipc_score + 0.4 * area_score + 0.1 * min(1.0, metrics.get("perf_per_watt", 0) / 5.0)
    
    elif task_id == 2:
        # M-Series: Score = f(throughput, efficiency, thermal compliance)
        throughput = metrics.get("throughput_gips", 0)
        ppw = metrics.get("perf_per_watt", 0)
        pd = metrics.get("max_power_density", 999)
        max_pd = task_constraints.get("max_power_density", 1.5)
        throttle = metrics.get("throttled_factor", 1.0)
        
        throughput_score = min(1.0, throughput / 50.0)
        efficiency_score = min(1.0, ppw / 2.0)
        
        if pd > max_pd:
            thermal_score = max(0, 1.0 - (pd - max_pd) / max_pd)
        else:
            thermal_score = 1.0
        
        # Throttle penalty
        thermal_score *= throttle
        
        score = 0.35 * throughput_score + 0.25 * efficiency_score + 0.25 * thermal_score + 0.15 * min(1.0, throttle)
    
    else:
        score = 0.0
    
    # HARD CLAMP to [0.0, 1.0]
    return round(max(0.0, min(1.0, score)), 4)
