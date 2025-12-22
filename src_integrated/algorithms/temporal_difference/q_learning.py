import numpy as np
from core.base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(self, env, gamma=0.9, alpha=0.1):
        super().__init__(env)
        self.gamma = gamma
        self.alpha = alpha
        self.Q = {}
        self.actions = env.action_space

    def get_q(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(len(self.actions))
        return self.Q[state]

    def predict(self, state):
        # Greedy policy
        q_values = self.get_q(state)
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)

    def train(self, num_episodes=1000, max_steps=100, behavior_policy=None, v_star=None):
        """
        Train using Q-Learning.
        Args:
            behavior_policy: A function that takes (agent, state) and returns an action.
                             Implements Strategy Pattern.
            v_star: Optimal value function (dict) for error calculation.
        """
        if behavior_policy is None:
            behavior_policy = self._epsilon_greedy_policy

        history = []
        all_states = self.env.get_all_states()
        
        for _ in range(num_episodes):
            state = self.env.reset()
            steps = 0
            
            for _ in range(max_steps):
                steps += 1
                action = behavior_policy(self, state)
                next_state, reward, done, _ = self.env.step(action)
                
                self.update(state, action, reward, next_state)
                
                if done:
                    break
                state = next_state
            
            if v_star:
                # Calculate Max Error ||V_approx - V*||_inf
                max_error = 0
                for s in all_states:
                    q_values = self.get_q(s)
                    v_approx = np.max(q_values)
                    error = abs(v_approx - v_star[s])
                    if error > max_error:
                        max_error = error
                history.append(max_error)
            else:
                history.append(steps)
        return history

    def update(self, state, action, reward, next_state):
        q_values = self.get_q(state)
        next_q_values = self.get_q(next_state)
        
        max_next_q = np.max(next_q_values)
        
        # Q(S, A) <- Q(S, A) + alpha * (R + gamma * max_a Q(S', a) - Q(S, A))
        q_values[action] += self.alpha * (reward + self.gamma * max_next_q - q_values[action])

    # --- Built-in policies ---
    @staticmethod
    def _epsilon_greedy_policy(agent, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.choice(agent.actions)
        else:
            return agent.predict(state)
