import numpy as np
from core.base_agent import BaseAgent

class MCAgent(BaseAgent):
    def __init__(self, env, epsilon=0, gamma=0.9, alpha=0.01):
        super().__init__(env)
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        
        # Q-table: state -> [q_val_action_0, q_val_action_1, ...]
        # Since states are tuples, we can use a dict or map tuples to indices.
        # Let's use a dict for flexibility.
        self.Q = {} 
        self.actions = env.action_space

    def get_q(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(len(self.actions))
        return self.Q[state]

    def predict(self, state):
        # Greedy action
        q_values = self.get_q(state)
        # Random tie-breaking
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)

    def choose_action(self, state):
        # Epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return self.predict(state)

    def train(self, num_episodes=1000, max_steps=100, exploring_starts=True, v_star=None):
        history = []
        all_states = self.env.get_all_states()
        
        for _ in range(num_episodes):
            episode = self._generate_episode(max_steps, exploring_starts)
            self.update(episode)
            
            if v_star:
                # Calculate Max Error ||V_approx - V*||_inf
                # V_approx(s) = max_a Q(s, a)
                max_error = 0
                for s in all_states:
                    q_values = self.get_q(s)
                    v_approx = np.max(q_values)
                    error = abs(v_approx - v_star[s])
                    if error > max_error:
                        max_error = error
                history.append(max_error)
            else:
                # Track episode length (steps) as a proxy for error/performance
                history.append(len(episode))
        return history

    def _generate_episode(self, max_steps, exploring_starts):
        episode = []
        
        if exploring_starts:
            # Randomly select any state from the observation space
            all_states = self.env.observation_space
            start_idx = np.random.randint(len(all_states))
            state = all_states[start_idx]
            
            # Randomly select a first action
            action = np.random.choice(self.actions)
            
            # Set environment to this state
            # Note: We need to manually set the state in the environment
            self.env.state = state
            
            # Execute the first step
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            
            if done:
                return episode
                
            state = next_state
        else:
            state = self.env.reset()
        
        for _ in range(max_steps):
            action = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            
            if done:
                break
            state = next_state
            
        return episode

    def update(self, episode):
        G = 0
        # Every-visit MC
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            
            q_values = self.get_q(state)
            # Q(S, A) <- Q(S, A) + alpha * (G - Q(S, A))
            q_values[action] += self.alpha * (G - q_values[action])
