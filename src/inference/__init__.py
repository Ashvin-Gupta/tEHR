"""Inference module for EHR models."""

from .generator import run_ehr_inference
from .trajectory_generator import EHRTrajectoryGenerator

__all__ = [
    "run_ehr_inference",
    "EHRTrajectoryGenerator",
]
