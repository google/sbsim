import numpy as np
import json
from smart_control.mcts.execute_policy_utils import compute_avg_return, load_environment
from mctspy.tree.nodes import MonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch


SECONDS_IN_AN_HOUR = 60 * 60


"""
    This class contains the state of the environment that corresponds to a certain 
    node in the Monter Carlo Tree Search tree.
"""
class NodeEnvironmentState:

    def __init__(self, node_temps, node_timestamp, node_state_return, node_previous_step):
        self.node_temps = node_temps
        self.node_timestamp = node_timestamp
        self.node_previous_step = node_previous_step
        self.node_state_return = node_state_return

    def __repr__(self):

        state = {
            "node_timestamp": str(self.node_timestamp),
            "node_state_return": self.node_state_return,
        }

        return f"NodeEnvironmentState(\nf{json.dumps(state, indent=4)}\n)"


class SbsimMonteCarloTreeSearchNode(MonteCarloTreeSearchNode):

    def __init__(self, node_environment_state: NodeEnvironmentState, # this holds data for the state of the environment at this node
                 parent=None
                 ):
        
        super().__init__(None, parent)
        self.children = { }

        # these values are used to keep track of the node score
        self._q = 0
        self._n = 1
        self._cumulative_number_of_hours = 1

        self.node_environment_state = node_environment_state # this is the state of the simulation at this node


    def __repr__(self):
        state = {
            "timestamp": str(self.node_environment_state.node_timestamp),
            "state_return": self.node_environment_state.node_state_return,
            "cumulative_number_of_hours": self._cumulative_number_of_hours,
            "children": list(self.children.keys())
        }
        return f"SbsimMonteCarloTreeSearchNode(\n{ json.dumps(state, indent=4) }\n)"


    def untried_actions(self, possible_actions):
        return possible_actions - set(self.children.keys())


    def is_fully_expanded(self, possible_actions):
        return len(self.untried_actions(possible_actions)) == 0
    

    @ property
    def n(self):
        return self._n


    @ property
    def q(self):
        return self._q
    

    @ staticmethod
    def run_expansion(node_environment_state, policy):
        env = SbSimMonteCarloTreeSearch.get_node_environment(node_environment_state) # get the environment at this node
        return_val, _ = compute_avg_return(env, policy, time_zone="US/Pacific", render_interval_steps=1e30, trajectory_observers=None, num_steps=1)

        new_node_environment_state = NodeEnvironmentState(
            node_temps=env.building._simulator._building.temp,
            node_timestamp=env.building._simulator._current_timestamp,
            node_state_return=node_environment_state.node_state_return + return_val,
            node_previous_step=env.current_time_step()
        )

        return new_node_environment_state
    

    def add_child(self, action, child_node_environment_state):
        new_node = SbsimMonteCarloTreeSearchNode(child_node_environment_state, parent=self)
        self.children[action] = new_node
        return new_node


    def backpropagate(self, result, number_of_hours):
        self._q += result
        self._n += 1
        self._cumulative_number_of_hours += number_of_hours
        if self.parent:
            self.parent.backpropagate(result, number_of_hours)


    def calculate_score(self, c_param=1e-5):
        return (self.q / self._cumulative_number_of_hours) + c_param * np.sqrt((2 * np.log(self.parent.n) / self.n))
    

    """
        This method runs a rollout from the current node to a certain number of steps
        and returns the difference between the rollout return value and the baseline return 
        value, as well as the number of hours elapsed from the start of the episode to
        the timestamp when the rollout ends

        Args:
            env: Environment
            node_environment_state: NodeEnvironmentState
            rollout_policy: Policy
            baseline_return_by_timestamp: Dict
            rollout_steps: int
        
        Returns:
            Tuple[float, float]: The return value of the rollout and the number of hours that the rollout took
    """
    @ staticmethod
    def run_rollout(env, # environment from which the rollout should begin
                    node_environment_state, # environment state of the environment before rollout
                    rollout_policy, # policy to use for the rollout
                    baseline_return_by_timestamp, # baseline return value by timestamp (obtained using the baseline policy)
                    rollout_steps # number of steps to rollout
                    ):
        return_val, _ = compute_avg_return(env, rollout_policy, render_interval_steps=1e30, num_steps=rollout_steps)

        rollout_return = node_environment_state.node_state_return + return_val
        rollout_timestamp = env.building._simulator._current_timestamp

        env.reset()
        episode_start_timestamp = env.current_simulation_timestamp

        return_delta = (baseline_return_by_timestamp[rollout_timestamp] - rollout_return)
        number_of_hours = (rollout_timestamp - episode_start_timestamp).total_seconds() / SECONDS_IN_AN_HOUR
        
        return return_delta, number_of_hours
    

    def is_terminal_node(self):
        return False
    

    def best_child(self):
        best_child, best_score = None, -1e30

        for key in self.children:
            c = self.children[key]
            score = c.calculate_score()

            if score > best_score:
                best_child = c
                best_score = score
        
        return best_child
    

    """
        This method spins up an environment using the config from @ param base_env_config
        and then restores the state from @ param node_environment_state

        Args:
            base_env_config: Dict
            node_environment_state: NodeEnvironmentState

        Returns:
            Environment: The environment with the state restored
    """
    @ staticmethod
    def get_node_environment(base_env_config, node_environment_state):
        env = load_environment(base_env_config)

        env.building._simulator._building.temp = node_environment_state.node_temps
        env.building._simulator._current_timestamp = node_environment_state.node_timestamp
        env._current_time_step = node_environment_state.node_previous_step

        return env


    def expand(self):
        pass
    

    def rollout(self):
        pass

    def depth(self):
        if not self.children:
            return 1
        return 1 + max([child.depth() for child in self.children.values()])

    
class SbSimMonteCarloTreeSearch(MonteCarloTreeSearch):
    
    def __init__(self,
                 node,
                 default_return_by_timestamp,
                 possible_actions,
                 base_env_config,
                 action_sequence,
                 rollout_steps=1,
                 expand_steps=1
                 ):
        
        super().__init__(node)
        self.default_return_by_timestamp = default_return_by_timestamp
        self.possible_actions = possible_actions
        self.base_env_config = base_env_config
        self.action_sequence = action_sequence
        self.rollout_steps = rollout_steps
        self.expand_steps = expand_steps
    

    """
        This is a recursive method to get the best @ param num_nodes nodes
        in the MCTS tree

        Args:
            num_nodes: int
            c_param: float
        
        Returns:
            List[tuple(SbsimMonteCarloTreeSearchNode, Tuple[float, float])]: List of the 
                nodes/action pairs that should be expanded
    """
    def get_nodes_for_expansion(self, num_nodes=5, c_param=1e-5):
        nodes_for_expansion = [] # each element is a tuple of (parent, node_to_expand, action)

        def recur(root):
            if len(nodes_for_expansion) == num_nodes: return

            if not root.is_fully_expanded(self.possible_actions):
                action = root.untried_actions(self.possible_actions).pop()
                nodes_for_expansion.append((root, action))
            
            nodes = [root.children[key] for key in root.children.keys()]
            nodes.sort(key=lambda x: -(x.q / x._cumulative_number_of_hours) - c_param * np.sqrt((2 * np.log(root.n) / x.n)))
            for node in nodes:
                recur(node)
        
        recur(self.root)
        return nodes_for_expansion
    

    """
        This method takes in the results of the node expansions simulations (which are parallelized), and then
        takes care of actually adding the new nodes onto the tree

        Args:
            expansion_results: List[tuple(SbsimMonteCarloTreeSearchNode, tuple, NodeEnvironmentState)]
        
        Returns:
            List[SbsimMonteCarloTreeSearchNode]: List of the newly created nodes
    """
    def perform_expansions(self, expansion_results):
        
        new_nodes = []

        for parent, action, state in expansion_results:
            new_nodes.append(parent.add_child(action, state))

        return new_nodes
    
    """
        This method takes in the results of the rollouts (which are parallelized), and then takes care
        of actually backpropagating the results of the rollouts

        Args:
            rollout_results: List[tuple(SbsimMonteCarloTreeSearchNode, float, NodeEnvironmentState)]

        Returns:
    """
    def perform_backpropagations(self, rollout_results):
        for node, result, number_of_hours in rollout_results:
            node.backpropagate(result, number_of_hours)

    
    def get_tree_depth(self):
        return self.root.depth()
    