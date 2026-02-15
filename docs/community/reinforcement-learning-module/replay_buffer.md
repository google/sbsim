---
layout: "default"
title: "Replay Buffer"
nav_order: 4
parent: "Reinforcement Learning Module"
grand_parent: "Smart Control Project Documentation"
---

# Replay Buffer

The replay buffer stores agent experiences for later reuse, enabling efficient learning from past interactions.

This project provides a wrapper class around a Reverb replay buffer to facilitate interaction with it.

## ReplayBufferManager

The `ReplayBufferManager` class simplifies the creation and management of replay buffers:

```python
from smart_control.reinforcement_learning.replay_buffer.replay_buffer import ReplayBufferManager

# Create a replay buffer manager
replay_manager = ReplayBufferManager(
    data_spec=agent.collect_data_spec, # agent is a TF-Agents agent
    capacity=50000,
    checkpoint_dir="path/to/checkpoint/dir",
    sequence_length=2
)

# Create a new replay buffer
replay_buffer, replay_buffer_observer = replay_manager.create_replay_buffer()

# Or load an existing replay buffer
replay_buffer, replay_buffer_observer = replay_manager.load_replay_buffer()
```

To add experiences to the replay buffer, you can add the `replay_buffer_observer` object returned above. For example:

```python
# Combine observers
    replay_buffer, replay_buffer_observer = replay_manager.load_replay_buffer()

    collect_actor = actor.Actor(
        ...,
        observers=[replay_buffer_observer],
        ...,
    )
```

### Key Methods

- **`create_replay_buffer()`**: Creates a new replay buffer and observer
- **`load_replay_buffer()`**: Loads an existing replay buffer from a checkpoint
- **`get_dataset(batch_size, num_steps)`**: Creates a TensorFlow dataset for sampling
- **`num_frames()`**: Returns the current number of frames in the buffer
- **`clear()`**: Clears all data from the buffer
- **`close()`**: Closes the buffer server and cleans up resources

## Populating the Buffer

### Initial Population

To pre-populate the buffer with some initial experiences (e.g. training an off-policy algorithm) you can use the `populate_starter_buffer.py` script, at `scripts/populate_replay_buffer.py`. This uses the baseline schedule policy from `policies/schedule_policy.py` to pre-populate the buffer:

```bash
# Populate a starter buffer using a baseline policy
python scripts/populate_starter_buffer.py \
    --buffer-name my-starter-buffer \
    --capacity 50000 \
    --steps-per-run 672 \
    --num-runs 10
```

## Sampling from the Buffer

For training, experiences are sampled from the buffer as batches:

```python
# Create a dataset for sampling
dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3
).prefetch(3)
```

## Checkpointing

Replay buffers can be checkpointed to disk for persistence:

```python
# Save the current state
replay_buffer.py_client.checkpoint()

# Load from checkpoint (done through ReplayBufferManager)
replay_manager = ReplayBufferManager(
    data_spec=agent.collect_data_spec,
    capacity=50000,
    checkpoint_dir="path/to/checkpoint/dir"
)
replay_buffer, observer = replay_manager.load_replay_buffer()
```

---

[Back to Reinforcement Learning](../reinforcement-learning-module.md)

[Back to Home](../../index.md)
