import numpy as np
from core.base_env import BaseEnvironment

class GridWorld(BaseEnvironment):
    def __init__(self, grid_size=(5, 5)):
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        
        # Define special states (0-indexed)
        # Using coordinates from src2/src5 but ensuring 0-indexing
        # src2: forbidden: [(1,1), (1,2), (2,2), (3,3), (3,1), (4,1)] (seems to be 0-indexed already or mixed?)
        # src5: obstacle_states = [(2, 2), (2, 3), (3, 3), (4, 2), (4, 4), (5, 2)] (1-indexed)
        # Let's map src5 (1-indexed) to 0-indexed:
        # (2,2)->(1,1), (2,3)->(1,2), (3,3)->(2,2), (4,2)->(3,1), (4,4)->(3,3), (5,2)->(4,1)
        # This matches src2 exactly!
        
        self.forbidden_states = [
            (1, 1), (1, 2), 
            (2, 2), 
            (3, 1), (3, 3), 
            (4, 1)
        ]
        
        # Target: src5 (4,3) -> (3,2) in 0-indexed. src2 says (3,2). Matches.
        self.target_state = (3, 2)
        
        self.actions = ['up', 'down', 'left', 'right', 'stay']
        self.action_map = {
            0: 'up', 1: 'down', 2: 'left', 3: 'right', 4: 'stay'
        }
        
        # Rewards
        self.r_boundary = -1
        self.r_forbidden = -1
        self.r_target = 1
        self.r_step = 0
        
        self.state = None

    @property
    def action_space(self):
        return list(range(len(self.actions)))

    @property
    def observation_space(self):
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def reset(self):
        # Start at (0, 0) or random? src5 uses (1,1) -> (0,0)
        self.state = (0, 0)
        return self.state

    def step(self, action_idx):
        """
        Execute action.
        Args:
            action_idx: int index of action
        """
        if self.state is None:
            self.reset()
            
        r, c = self.state
        action = self.action_map[action_idx]
        
        next_r, next_c = r, c
        hit_wall = False
        
        if action == 'up':
            next_r -= 1
        elif action == 'down':
            next_r += 1
        elif action == 'left':
            next_c -= 1
        elif action == 'right':
            next_c += 1
        elif action == 'stay':
            pass
            
        # Boundary check
        if next_r < 0 or next_r >= self.rows or next_c < 0 or next_c >= self.cols:
            hit_wall = True
            next_r, next_c = r, c # Stay in place
            
        next_state = (next_r, next_c)
        reward = self.r_step
        done = False
        
        if hit_wall:
            reward = self.r_boundary
        elif next_state in self.forbidden_states:
            reward = self.r_forbidden
        elif next_state == self.target_state:
            reward = self.r_target
            # done = True 
            # In the original HW3, the task is continuous (can stay at target to get +1 repeatedly).
            # So we should NOT terminate at target.
            done = False
            
        self.state = next_state
        return next_state, reward, done, {}

    def render(self):
        """
        Simple text-based rendering.
        """
        grid = np.full((self.rows, self.cols), '.', dtype=str)
        
        # Mark special states
        for r, c in self.forbidden_states:
            grid[r, c] = 'X'
        
        tr, tc = self.target_state
        grid[tr, tc] = 'T'
        
        # Mark agent
        if self.state is not None:
            ar, ac = self.state
            grid[ar, ac] = 'A'
            
        print("\nGridWorld:")
        print(grid)
        print("")

    # --- Model-based interface for DP ---
    
    def get_all_states(self):
        return self.observation_space

    def get_transition_model(self, state, action_idx):
        """
        Returns (next_state, reward) for a deterministic environment.
        Used by DP algorithms.
        """
        # Temporarily save current state
        saved_state = self.state
        
        self.state = state
        next_state, reward, _, _ = self.step(action_idx)
        
        # Restore state
        self.state = saved_state
        
        return next_state, reward
