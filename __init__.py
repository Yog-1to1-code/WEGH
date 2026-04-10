# WEGH — Workload Evaluation for Generative Hardware
"""
WEGH OpenEnv Package.
Exports: CPUAction, CPUObservation, CPUState, WEGHEnv
"""

__version__ = "1.0.0"

from models import CPUAction, CPUObservation, CPUState
from client import WEGHEnv

__all__ = ["CPUAction", "CPUObservation", "CPUState", "WEGHEnv"]
