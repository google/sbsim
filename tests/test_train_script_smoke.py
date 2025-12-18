"""Smoke test for train.py script - runs end-to-end training."""

import os
import sys
import subprocess
import uuid
import pytest
from pathlib import Path
from tf_agents.drivers import py_driver

@pytest.mark.slow
def test_train_script_smoke_runs_and_saves(tmp_path):
    if os.environ.get("RUN_SLOW_TESTS", "0") != "1":
        pytest.skip("set RUN_SLOW_TESTS=1 to run smoke training test")

    exp_name = f"exp_smoke_{uuid.uuid4().hex[:8]}"

    cmd = [
        sys.executable, "-m",
        "smart_control.reinforcement_learning.scripts.train",
        f"--experiment_name={exp_name}",
        "--starter_buffer_name=default",
        "--agent_type=sac",
        "--train_iterations=1",
        "--collect_steps_per_training_iteration=1",
        "--learner_iterations=1",
        "--eval_interval=1",
        "--checkpoint_interval=1",
        "--use_uniform_replay_fallback=True",
    ]

    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"

    proc = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=1200
    )

    # Always print captured logs for debugging
    print("\n--- STDOUT ---\n", proc.stdout)
    print("\n--- STDERR ---\n", proc.stderr, file=sys.stderr)

    # 1) Must exit cleanly
    assert proc.returncode == 0, f"Non-zero exit: {proc.returncode}"

    # 2) Must log completion (logs can be on stderr)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    assert ("Agent training completed" in combined) or ("Training complete" in combined), \
        "Completion message not found in logs."

    # 3) Must create expected outputs
    base = Path("smart_control/reinforcement_learning/data/experiment_results")
    results_dir = base / exp_name
    assert results_dir.is_dir(), f"Missing results dir: {results_dir}"
    assert (results_dir / "policies").is_dir(), "Missing policies dir"
    assert (results_dir / "replay_buffer").is_dir(), "Missing replay_buffer dir"
    assert (results_dir / "experiment_parameters.json").is_file(), "Missing params json"
