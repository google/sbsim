---
layout: "default"
title: "Observers"
nav_order: 5
parent: "Reinforcement Learning Module"
grand_parent: "Smart Control Project Documentation"
---

# Observers

Observers monitor agent behavior during training and evaluation by processing trajectory data at each step. Observers can be used to log information, visualize agent performance, or save data for later analysis.
They are designed to be modular by using the Observer design pattern, allowing you to easily add or remove observers to your agent/actor depending
on your needs.

## Observer Interface

All observers implement the `Observer` abstract base class, defined in `observers.base_observer.py`:

```python
class Observer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, trajectory: trajectory_lib.Trajectory) -> None:
        """Process a trajectory."""
        pass

    def reset(self) -> None:
        """Reset observer state between episodes."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass
```

## Available Observers

The following observers are implemented and available.

### PrintStatusObserver

Logs training progress information to the console:

```python
from smart_control.reinforcement_learning.observers.print_status_observer import PrintStatusObserver

print_observer = PrintStatusObserver(
    status_interval_steps=10,
    environment=env, # tf environment
    replay_buffer=replay_buffer # replay buffer being used. Used to print replay buffer size.
                                # Can be None if not using replay buffer
)
```

### TrajectoryRecorderObserver

Records trajectory data for later analysis and visualization. This is very useful for evaluation runs:

```python
from smart_control.reinforcement_learning.observers.trajectory_recorder_observer import TrajectoryRecorderObserver

trajectory_observer = TrajectoryRecorderObserver(
    save_dir=trajectory_dit, # directory where to save plots/data
    environment=env # tf environment
)
```

### CompositeObserver

Combines multiple observers into a single observer. This is useful to make it simpler to pass observers
to the agent/actor:

```python
from smart_control.reinforcement_learning.observers.composite_observer import CompositeObserver

combined_observers = CompositeObserver([print_observer, replay_buffer_observer])
# now, just need to pass observers = combined_observers to the agent/actor
```

[Back to Reinforcement Learning](../reinforcement-learning-module.md)

[Back to Home](../../index.md)
