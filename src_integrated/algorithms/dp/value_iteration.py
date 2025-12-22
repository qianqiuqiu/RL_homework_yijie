import numpy as np
from core.base_agent import BaseAgent

class ValueIterationAgent(BaseAgent):
    def __init__(self, env, gamma=0.9, theta=1e-4):
        super().__init__(env)
        self.gamma = gamma
        self.theta = theta
        self.V = {state: 0.0 for state in env.get_all_states()}
        self.policy = {} # state -> action_idx

    def train(self):
        """
        Executes Value Iteration.
        """
        states = self.env.get_all_states()
        actions = self.env.action_space
        
        iteration = 0
        while True:
            delta = 0
            iteration += 1
            
            for state in states:
                v = self.V[state]
                # Bellman Optimality Equation
                # V(s) = max_a [ r + gamma * V(s') ]
                q_values = []
                for action in actions:
                    next_state, reward = self.env.get_transition_model(state, action)
                    q_val = reward + self.gamma * self.V[next_state]
                    q_values.append(q_val)
                
                self.V[state] = max(q_values)
                delta = max(delta, abs(v - self.V[state]))
            
            if delta < self.theta:
                break
                
        self._derive_policy()
        return self.V, self.policy

    def _derive_policy(self):
        states = self.env.get_all_states()
        actions = self.env.action_space
        
        for state in states:
            best_action = None
            best_q = float('-inf')
            
            for action in actions:
                next_state, reward = self.env.get_transition_model(state, action)
                q_val = reward + self.gamma * self.V[next_state]
                
                if q_val > best_q:
                    best_q = q_val
                    best_action = action
            
            self.policy[state] = best_action

    def predict(self, state):
        return self.policy.get(state, 0) # Default to 0 if not found
