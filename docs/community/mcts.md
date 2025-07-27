---
layout: "default"
title: "Monte Carlo Tree Search"
nav_order: 1
parent: "Learning Algorithms"
grand_parent: "Smart Control Project Documentation"
---

# Monte Carlo Tree Search

MCTS is not technically an RL algorithm, but can be attempted and leveraged as a potential baseline performance metric.

In the proposed setup, each node in the MCTS tree is identified by a given sequence of actions taken since the episode's start. Nodes in different depths of the tree are a fixed number of `expand_steps` apart, such that the MCTS agent can only decide on a new action every `expand_steps` number of steps.

Implementing MCTS in this setup comes with a few challenges, which we highlight here:

## Slow Rollouts and Parallelism

Typically, a rollout in an MCTS consists of choosing some default policy (e.g. random moves in a game), and then playing the episode to completion. In this case, doing this is very slow, as running an environment simulation is time consuming. Because of this, we only perform rollouts for a smaller fixed number of steps `rollout_steps` after the corresponding node's timestamp.

To address this problem, we also adapt the implementation to support parallelism using `multiprocessing`. This was favored instead of multithreading due to the GIL lock. Because we are not dealing with async operations, multithreading should not provide a performance benefit, as only one thread is allowed to execute code at a time. Multiprocessing however, allows multiple distinct python processes, which should then obtain a speedup.

The design of `SbsimMonteCarloTreeSearchNode` and `SbSimMonteCarloTreeSearch` was adapted in order to support `multiprocessing`. For example, since the node objects can't be serialized and thus can't be passed into workers, methods such as `SbsimMonteCarloTreeSearchNode.run_rollout` are made static to allow workers to execute them independently.

## Node Selection

As described above, our implementation performs many concurrent rollouts. To choose which nodes to expand, the method `SbSimMonteCarloTreeSearch.get_nodes_for_expansion` is used. This method chooses the `@ param num_nodes` nodes to expand by finding adding a node if it is not fully expanded, and then recursively calling itself on the node's children, in order from the highest to lowest score. The constant `@ param c_param` should control the balance between exploration and exploitation.

## Node Evaluation

Node evaluation: to evaluate a node, we consider its performance in comparison to the a baseline schedule policy. A node's score is given by the sum of the differences (when compared to the baseline policy return) of the rollout returns of all its children. To keep things fair, this is then normalized by the sum of the number of hours elapsed until the rollout end for all children. In summary, this means that each node is scored by the *average rollout return improvement per hour for all it's children*.

---

[Back to Learning Algorithms](learning-algorithms.md)

[Back to Home](../index.md)