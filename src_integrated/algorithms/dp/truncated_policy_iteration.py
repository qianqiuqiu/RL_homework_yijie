import numpy as np
from core.base_agent import BaseAgent

class TruncatedPolicyIterationAgent(BaseAgent):
    def __init__(self, env, gamma=0.9, theta=1e-4, k=10):
        """
        Args:
            env: The environment.
            gamma: Discount factor.
            theta: Convergence threshold for early stopping (optional).
            k: Number of policy evaluation steps per iteration.
        """
        super().__init__(env)
        self.gamma = gamma
        self.theta = theta
        self.k = k
        self.V = {state: 0.0 for state in env.get_all_states()}
        # Initialize random policy
        self.policy = {state: np.random.choice(env.action_space) for state in env.get_all_states()}

    def train(self, v_star=None):
        """
        Executes Truncated Policy Iteration.
        Args:
            v_star: Optimal value function (dict) for error calculation.
        """
        iteration = 0
        history = [] 
        
        while True:
            iteration += 1
            self._policy_evaluation(self.k)
            policy_stable = self._policy_improvement()
            
            if v_star:
                # Calculate Max Error ||V - V*||_inf
                error = max([abs(self.V[s] - v_star[s]) for s in self.env.get_all_states()])
                history.append(error)
            else:
                # Fallback: Sum of V(s)
                total_value = sum(self.V.values())
                history.append(total_value)
            
            if policy_stable:
                break
        
        return self.V, self.policy, history

    def _policy_evaluation(self, k):
        states = self.env.get_all_states()
        max_delta = 0
        
        for _ in range(k):
            delta = 0
            for state in states:
                v = self.V[state]
                action = self.policy[state]
                next_state, reward = self.env.get_transition_model(state, action)
                self.V[state] = reward + self.gamma * self.V[next_state]
                delta = max(delta, abs(v - self.V[state]))
            max_delta = max(max_delta, delta)
            
            # Optional: Early stopping if it converges faster than k steps
            if delta < self.theta:
                break
        return max_delta

    def _policy_improvement(self):
        policy_stable = True
        states = self.env.get_all_states()
        actions = self.env.action_space
        
        for state in states:
            old_action = self.policy[state]
            
            # Find best action
            best_action = None
            best_q = float('-inf')
            
            for action in actions:
                next_state, reward = self.env.get_transition_model(state, action)
                q_val = reward + self.gamma * self.V[next_state]
                
                if q_val > best_q:
                    best_q = q_val
                    best_action = action
            
            self.policy[state] = best_action
            
            if old_action != self.policy[state]:
                policy_stable = False
                
        return policy_stable

    def predict(self, state):
        return self.policy.get(state, 0)
