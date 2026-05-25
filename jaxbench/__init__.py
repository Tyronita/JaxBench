"""JaxBench — a KernelBench-style benchmark for evolving jaxlib kernels.

Each task pins a JAX op, the jaxlib source file that backs it, the targeted build +
hot-swap recipe, and the correctness + speedup evaluation. ShinkaEvolve mutates the
file; JaxBench builds, gates on correctness, and scores the speedup.
"""
from . import (challenges, workloads, challenge_runner,  # the challenge-dataset core
               registry, ops, reference, correctness, bench, metrics, build, runner,
               sharding)  # noqa: F401

__version__ = "0.2.0"
__all__ = ["challenges", "workloads", "challenge_runner",
           "registry", "ops", "reference", "correctness", "bench", "metrics",
           "build", "runner", "sharding"]
