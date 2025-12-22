import numpy as np
from core.base_agent import BaseAgent
from .sgd_optimizer import SGDOptimizer

class TDLinearAgent(BaseAgent):
    def __init__(self, env, feature_extractor, gamma=0.9, optimizer=None):
        super().__init__(env)
        self.feature_extractor = feature_extractor
        self.gamma = gamma
        self.w = np.zeros(feature_extractor.get_feature_dim())
        
        if optimizer is None:
            self.optimizer = SGDOptimizer(learning_rate=0.01)
        else:
            self.optimizer = optimizer

    def predict(self, state):
        # This agent only estimates Value function V(s), not Policy.
        # So predict might not be applicable unless we have a model to do lookahead,
        # or if we were doing SARSA/Q-learning with approximation.
        # For HW6 it seems it was just evaluating V(s).
        # But to fit BaseAgent, we can implement a greedy policy based on V(s) if we have access to model,
        # or just raise NotImplementedError.
        # Given we have env.get_transition_model (if it's GridWorld), we can do lookahead.
        
        if hasattr(self.env, 'get_transition_model'):
            best_action = None
            best_val = float('-inf')
            for action_idx in self.env.action_space:
                next_state, reward = self.env.get_transition_model(state, action_idx)
                val = reward + self.gamma * self.get_value(next_state)
                if val > best_val:
                    best_val = val
                    best_action = action_idx
            return best_action
        else:
            raise NotImplementedError("Prediction requires a model for Value-based agents")

    def get_value(self, state):
        features = self.feature_extractor.get_features(state)
        return np.dot(self.w, features)

    def train(self, num_episodes=1000, max_steps=1000):
        # TD(0) on generated episodes
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            steps = 0
            while not done and steps < max_steps:
                steps += 1
                # Random policy for evaluation (as per HW6 usually)
                action = np.random.choice(self.env.action_space)
                next_state, reward, done, _ = self.env.step(action)
                
                self.update(state, reward, next_state)
                state = next_state

    def update(self, state, reward, next_state):
        phi_s = self.feature_extractor.get_features(state)
        phi_next_s = self.feature_extractor.get_features(next_state)
        
        v_s = np.dot(self.w, phi_s)
        v_next_s = np.dot(self.w, phi_next_s)
        
        td_target = reward + self.gamma * v_next_s
        td_error = td_target - v_s
        
        # Gradient of Loss = - td_error * phi_s
        gradient = -td_error * phi_s
        
        self.w = self.optimizer.update(self.w, gradient)
