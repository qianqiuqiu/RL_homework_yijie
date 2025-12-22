from abc import ABC, abstractmethod

class BaseEnvironment(ABC):
    """
    Abstract base class for environments, following a Gym-like interface.
    """
    
    @abstractmethod
    def reset(self):
        """
        Resets the environment to an initial state and returns the initial observation.
        Returns:
            observation: The initial state of the environment.
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        Args:
            action: An action provided by the agent.
        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            done: Whether the episode has ended.
            info: Contains auxiliary diagnostic information.
        """
        pass

    @abstractmethod
    def render(self):
        """
        Renders the environment.
        """
        pass
    
    @property
    @abstractmethod
    def action_space(self):
        """
        Returns the action space.
        """
        pass

    @property
    @abstractmethod
    def observation_space(self):
        """
        Returns the observation space.
        """
        pass
