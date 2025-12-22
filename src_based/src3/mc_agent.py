import numpy as np

class MCAgent:
    def __init__(self, env, epsilon=0.1, gamma=0.9, alpha=0.01):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha # Learning rate
        self.num_states = env.rows * env.cols
        self.num_actions = len(env.actions)
        
        # Q-values: (num_states, num_actions)
        # Initialize with zeros (or small random values if needed, but zeros is standard for this)
        self.Q = np.zeros((self.num_states, self.num_actions))
        
        # For policy extraction
        self.policy = np.zeros((self.num_states, self.num_actions))

    def choose_action(self, state_idx):
        # Epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            # Greedy action (with random tie-breaking)
            max_q = np.max(self.Q[state_idx])
            best_actions = np.where(self.Q[state_idx] == max_q)[0]
            # If multiple best actions, pick randomly
            return np.random.choice(best_actions)

    def generate_episode(self, max_steps=100, exploring_starts=True):
        episode = []
        
        if exploring_starts:
            # Random start state
            while True:
                r = np.random.randint(self.env.rows)
                c = np.random.randint(self.env.cols)
                state = (r, c)
                # We can start in forbidden states? Usually yes for full coverage.
                # But maybe not target?
                break
            
            # Random first action
            action = np.random.choice(self.num_actions)
            
            self.env.state = state
            state_idx = self.env.get_state_index(state)
            
            # Execute first action
            next_state, reward, done = self.env.step(action)
            episode.append((state_idx, action, reward))
            
            state = next_state
            state_idx = self.env.get_state_index(state)
        else:
            state = self.env.reset()
            state_idx = self.env.get_state_index(state)
        
        for _ in range(max_steps):
            action = self.choose_action(state_idx)
            next_state, reward, done = self.env.step(action)
            next_state_idx = self.env.get_state_index(next_state)
            
            episode.append((state_idx, action, reward))
            
            if done:
                break
            
            state = next_state
            state_idx = next_state_idx
            
        return episode

    def update(self, episode):
        G = 0
        # First-visit or Every-visit?
        # For constant alpha, Every-visit is natural and simpler.
        for t in range(len(episode) - 1, -1, -1):
            state_idx, action, reward = episode[t]
            G = self.gamma * G + reward
            
            # Update Q
            # Q(S, A) <- Q(S, A) + alpha * (G - Q(S, A))
            self.Q[state_idx, action] += self.alpha * (G - self.Q[state_idx, action])

    def get_optimal_policy_map(self):
        # Returns a grid of best actions and values
        policy_map = np.zeros((self.env.rows, self.env.cols), dtype=int)
        value_map = np.zeros((self.env.rows, self.env.cols))
        
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                s_idx = self.env.get_state_index((r, c))
                best_action = np.argmax(self.Q[s_idx])
                policy_map[r, c] = best_action
                value_map[r, c] = np.max(self.Q[s_idx])
                
        return policy_map, value_map
