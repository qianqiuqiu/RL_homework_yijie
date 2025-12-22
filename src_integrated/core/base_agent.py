from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for Reinforcement Learning agents.
    """
    
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def predict(self, state):
        """
        Select an action based on the current state (inference mode).
        Args:
            state: The current state.
        Returns:
            action: The selected action.
        """
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        """
        Train the agent.
        This method should implement the training loop or call specific update methods.
        """
        pass
    
    def update(self, *args, **kwargs):
        """
        Update the agent's knowledge (e.g., Q-table, Value function, Weights).
        This is a hook for the Template Method pattern if needed.
        """
        pass
