"""Parameterised correctness over every JaxBench task (all N sizes x dtype).

This is the gate ShinkaEvolve must keep green: a mutation that breaks any task's
residual is rejected regardless of speed. CPU tasks run everywhere; GPU tasks are
skipped automatically when no GPU is present (see conftest.py).
"""
import pytest
from jaxbench import registry, correctness

ALL_TASKS = registry.tasks()


@pytest.mark.parametrize("task", ALL_TASKS, ids=[t.id for t in ALL_TASKS])
def test_task_correct(task):
    c = correctness.check_task(task, seed=1)
    assert c.passed, (
        f"{task.id}: worst residual {c.worst_residual:.3e} > tol {c.tol:.0e} "
        f"(per-size: {c.per_size})"
    )


@pytest.mark.parametrize("task", [t for t in ALL_TASKS if t.device == "cpu"][:1] or [None])
def test_determinism(task):
    """Same seed -> identical residual (rules out nondeterministic kernels)."""
    if task is None:
        pytest.skip("no cpu task")
    a = correctness.check_task(task, seed=7).worst_residual
    b = correctness.check_task(task, seed=7).worst_residual
    assert a == b
