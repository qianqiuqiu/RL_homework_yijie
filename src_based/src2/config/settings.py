# GridWorld 环境与算法的配置项

# 环境设置
GRID_SIZE = (5, 5)  # 网格大小（行, 列）
SPECIAL_CELLS = {
    'goal': (4, 4),  # 目标状态坐标
    'trap': (1, 1)   # 陷阱状态坐标
}

# 奖励
REWARD_GOAL = 1.0
REWARD_TRAP = -1.0
REWARD_DEFAULT = 0.0

# 折扣因子
GAMMA = 0.9  # 未来回报的折扣因子

# 收敛阈值
THETA = 1e-6  # 收敛判断的阈值

# 初始策略
INITIAL_POLICY = None  # 在算法模块中定义