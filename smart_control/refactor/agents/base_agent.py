import abc
import logging
from typing import Any, Dict

import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.policies import tf_policy


logger = logging.getLogger(__name__)


class BaseAgent(abc.ABC):
    """Abstract base class for all RL agents.
    
    This class defines the core interface that all agents must implement.
    """
    
    @abc.abstractmethod
    def initialize(self) -> None:
        """Initialize the agent.
        
        This method should be called before using the agent.
        """
        pass
    
    @abc.abstractmethod
    def train(self, experience) -> Dict[str, Any]:
        """Train the agent on a batch of experience.
        
        Args:
            experience: A batch of experience data for training.
            
        Returns:
            A dictionary of loss metrics from training.
        """
        pass
    
    @property
    @abc.abstractmethod
    def policy(self) -> tf_policy.TFPolicy:
        """Returns the agent's main policy."""
        pass
    
    @property
    @abc.abstractmethod
    def collect_policy(self) -> tf_policy.TFPolicy:
        """Returns the agent's collection policy."""
        pass
    
    @property
    @abc.abstractmethod
    def collect_data_spec(self):
        """Returns the agent's data collection specification."""
        pass
    
    @property
    @abc.abstractmethod
    def train_step_counter(self) -> tf.Variable:
        """Returns the agent's training step counter."""
        pass


class TFAgentWrapper(BaseAgent):
    """Wrapper class for TF-Agents agents to conform to BaseAgent interface."""
    
    def __init__(self, tf_agent_instance: tf_agent.TFAgent):
        """Initialize with a TF-Agents agent instance.
        
        Args:
            tf_agent_instance: A TF-Agents agent instance.
        """
        self._agent = tf_agent_instance
    
    def initialize(self) -> None:
        """Initialize the agent."""
        self._agent.initialize()
    
    def train(self, experience) -> Dict[str, Any]:
        """Train the agent on a batch of experience.
        
        Args:
            experience: A batch of experience data for training.
            
        Returns:
            A dictionary of loss metrics from training.
        """
        loss_info = self._agent.train(experience)
        
        result = {'loss': loss_info.loss.numpy()}
        
        # Handle different types of extra info that might be returned by different agents
        if hasattr(loss_info, 'extra'):
            logger.info('Extra loss info found in agent training result')
            logger.info('Extra loss info type: %s', loss_info.extra)
            extra = loss_info.extra
            
            # SAC agent's extra is a LossInfo with fields like actor_loss, critic_loss, alpha_loss
            extra_dict = {}
            for attr_name in dir(extra):
                # Skip private attributes and methods
                if not attr_name.startswith('_') and not callable(getattr(extra, attr_name)):
                    attr_value = getattr(extra, attr_name)
                    # Convert TensorFlow tensors to numpy arrays
                    if hasattr(attr_value, 'numpy'):
                        extra_dict[attr_name] = attr_value.numpy()
                    else:
                        extra_dict[attr_name] = attr_value
            
            result['extra'] = extra_dict
        
        return result
    
    @property
    def policy(self) -> tf_policy.TFPolicy:
        """Returns the agent's main policy."""
        return self._agent.policy
    
    @property
    def collect_policy(self) -> tf_policy.TFPolicy:
        """Returns the agent's collection policy."""
        return self._agent.collect_policy
    
    @property
    def collect_data_spec(self):
        """Returns the agent's data collection specification."""
        return self._agent.collect_data_spec
    
    @property
    def train_step_counter(self) -> tf.Variable:
        """Returns the agent's training step counter."""
        return self._agent.train_step_counter
