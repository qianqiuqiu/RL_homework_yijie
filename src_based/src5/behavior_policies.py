import random
from grid_world import GridWorldEnv

class BehaviorPolicies:
    def __init__(self):
        self.env = GridWorldEnv()
        
        # 为每个策略定义箭头映射。
        # 键: 状态 (row, col)
        # 值: 首选动作列表 (箭头)
        
        # 图 1a: Epsilon = 1 (随机)
        self.policy_map_1a = {} 

        # 图 1b: Epsilon = 0.5
        self.policy_map_1b = {}
        for state in self.env.get_all_states():
            self.policy_map_1b[state] = ['right']

        # 图 1c: Epsilon = 0.1
        self.policy_map_1c = {}
        for state in self.env.get_all_states():
            self.policy_map_1c[state] = ['right']

        # 图 1d: Epsilon = 0.1 (变体)
        self.policy_map_1d = {
            (1, 1): ['down'], (1, 3): ['left'], (1, 4): ['right'], (1, 5): ['down'],
            (2, 1): ['right'], (2, 4): ['down'], (2, 5): ['left'],
            (3, 1): ['right'], (3, 2): ['left'], (3, 4): ['right'], (3, 5): ['down'],
            (4, 1): ['up'], (4, 5): ['up'],
            (5, 1): ['right'], (5, 2): ['down'], (5, 4): ['left'], (5, 5): ['up']
        }

    def _get_action(self, state, policy_map, epsilon):
        """
        基于策略映射和 epsilon 选择动作的通用方法。
        逻辑:
        1. 以 epsilon 的概率，选择随机有效动作。
        2. 以 1-epsilon 的概率，从箭头中选择（如果可用）。
           如果没有箭头，回退到随机。
        """
        valid_actions = self.env.get_valid_actions(state)
        
        # Epsilon-greedy 逻辑
        if random.random() < epsilon:
            # 探索: 随机动作
            if valid_actions:
                return random.choice(valid_actions)
        else:
            # 利用: 跟随箭头
            if state in policy_map and policy_map[state]:
                arrows = [a for a in policy_map[state] if a in valid_actions]
                if arrows:
                    return random.choice(arrows)
            
            # 如果没有箭头或箭头无效，回退
            if valid_actions:
                return random.choice(valid_actions)
                
        return random.choice(['up', 'down', 'left', 'right'])

    def get_policy_fig1a(self, state):
        # Epsilon = 1.0
        return self._get_action(state, self.policy_map_1a, epsilon=1.0)

    def get_policy_fig1b(self, state):
        # Epsilon = 0.5
        return self._get_action(state, self.policy_map_1b, epsilon=0.5)

    def get_policy_fig1c(self, state):
        # Epsilon = 0.1
        return self._get_action(state, self.policy_map_1c, epsilon=0.1)

    def get_policy_fig1d(self, state):
        # Epsilon = 0.1
        return self._get_action(state, self.policy_map_1d, epsilon=0.1)
