import numpy as np

class GridWorld:
    def __init__(self):
        self.rows = 5
        self.cols = 5
        self.start_state = (0, 0)
        self.target_state = (3, 2)
        # Forbidden states based on the image (yellow blocks)
        # 0-indexed coordinates corresponding to:
        # (2,2), (2,3), (3,3), (4,2), (4,4), (5,2) in 1-based indexing
        self.forbidden_states = [
            (1, 1), (1, 2),
            (2, 2),
            (3, 1), (3, 3),
            (4, 1)
        ]
        # Actions: 0:Up, 1:Right, 2:Down, 3:Left, 4:Stay
        self.actions = [0, 1, 2, 3, 4]
        self.action_names = ['Up', 'Right', 'Down', 'Left', 'Stay']

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        # action: 0:Up, 1:Right, 2:Down, 3:Left, 4:Stay
        r, c = self.state
        
        next_r, next_c = r, c
        
        if action == 0:   # Up
            next_r -= 1
        elif action == 1: # Right
            next_c += 1
        elif action == 2: # Down
            next_r += 1
        elif action == 3: # Left
            next_c -= 1
        elif action == 4: # Stay
            pass

        reward = 0
        
        # Check boundary
        if next_r < 0 or next_r >= self.rows or next_c < 0 or next_c >= self.cols:
            reward = -1
            next_r, next_c = r, c # Stay in previous state
        # Check forbidden
        elif (next_r, next_c) in self.forbidden_states:
            reward = -10
            # Depending on the problem definition, hitting a wall/forbidden might keep you in place or move you there.
            # The prompt says "forbidden area -10". Usually implies you enter it and get penalized, or bounce back.
            # The previous code allowed entering forbidden states? 
            # "elif (next_r, next_c) in self.forbidden_states: reward = -10"
            # It updated self.state to next_r, next_c.
            # Let's stick to: You CAN enter forbidden states, but get -10.
        else:
            # Normal or Target
            if (next_r, next_c) == self.target_state:
                reward = 1
            else:
                reward = 0
        
        self.state = (next_r, next_c)
        # Task is continuous (can stay at target to get +1 repeatedly)
        done = False 
        
        return self.state, reward, done

    def get_state_index(self, state):
        return state[0] * self.cols + state[1]
