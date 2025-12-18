"""Tests for ReplayBufferManager with uniform fallback."""

import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory

from smart_control.reinforcement_learning.replay_buffer.replay_buffer import ReplayBufferManager


def _collect_data_spec():
    """Create a simple trajectory spec for testing."""
    obs_spec = tensor_spec.TensorSpec(shape=(3,), dtype=tf.float32, name="observation")
    act_spec = tensor_spec.BoundedTensorSpec(
        shape=(2,), dtype=tf.float32, minimum=-1.0, maximum=1.0, name="action"
    )
    time_spec = tensor_spec.TensorSpec((), tf.int32, name="step_type")
    rew_spec = tensor_spec.TensorSpec((), tf.float32, name="reward")
    disc_spec = tensor_spec.BoundedTensorSpec(
        (), tf.float32, minimum=0.0, maximum=1.0, name="discount"
    )
    # Build a Trajectory spec (policy_info is empty tuple)
    return trajectory.Trajectory(
        step_type=time_spec,
        observation=obs_spec,
        action=act_spec,
        policy_info=(),
        next_step_type=time_spec,
        reward=rew_spec,
        discount=disc_spec,
    )


def _make_traj(step_type, obs, action, reward, discount, next_step_type):
    """Create a single trajectory for testing."""
    return trajectory.Trajectory(
        step_type=tf.constant(step_type, dtype=tf.int32),
        observation=tf.constant(obs, dtype=tf.float32),
        action=tf.constant(action, dtype=tf.float32),
        policy_info=(),
        next_step_type=tf.constant(next_step_type, dtype=tf.int32),
        reward=tf.constant(reward, dtype=tf.float32),
        discount=tf.constant(discount, dtype=tf.float32),
    )


def test_uniform_replay_buffer_adds_frames(tmp_path, monkeypatch):
    """Test that uniform replay buffer correctly adds trajectory frames."""
    # Force uniform fallback
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "2")

    # Create fake checkpoint dir
    ckpt_dir = tmp_path / "rb_ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Create replay buffer manager
    rbm = ReplayBufferManager(
        data_spec=_collect_data_spec(),
        capacity=50,
        checkpoint_dir=str(ckpt_dir),
        sequence_length=2,
    )

    # Create buffer and observer
    rb, observer = rbm.create_replay_buffer()
    assert rbm.num_frames() == 0

    # Add 5 simple transitions
    for _ in range(5):
        traj = _make_traj(
            step_type=ts.StepType.FIRST,
            obs=[0.1, 0.2, 0.3],
            action=[0.0, 0.5],
            reward=0.0,
            discount=1.0,
            next_step_type=ts.StepType.MID,
        )
        observer(traj)

    # Verify 5 frames were added
    assert rbm.num_frames() == 5
