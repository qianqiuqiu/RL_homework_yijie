import numpy as np

def compute_ground_truth(env, gamma=0.9, theta=1e-6):
    states = env.get_all_states()
    V = {s: 0.0 for s in states}
    
    while True:
        delta = 0
        for s in states:
            v = V[s]
            new_v = 0
            # 策略是均匀随机的 (5个动作每个概率0.2)
            prob = 0.2
            for a in env.actions:
                next_s, r = env.step(s, a)
                new_v += prob * (r + gamma * V[next_s])
            
            V[s] = new_v
            delta = max(delta, abs(v - new_v))
        
        if delta < theta:
            break
            
    return V

def generate_episodes(env, num_episodes=500, steps_per_episode=500):
    episodes = []
    states = env.get_all_states()
    
    for _ in range(num_episodes):
        episode = []
        # 随机起始状态
        s_idx = np.random.randint(len(states))
        curr_s = states[s_idx]
        
        # 随机起始动作 (根据提示 "随机状态-动作对")
        # 实际上, 我们只需要执行这第一个动作。
        # 但随后的动作遵循策略。
        # 让我们模拟轨迹。
        
        # 第一步: 随机动作
        a_start = np.random.choice(env.actions)
        next_s, r = env.step(curr_s, a_start)
        episode.append((curr_s, r, next_s))
        curr_s = next_s
        
        # 剩余步骤
        for _ in range(steps_per_episode - 1):
            # 策略是均匀随机的
            a = np.random.choice(env.actions)
            next_s, r = env.step(curr_s, a)
            episode.append((curr_s, r, next_s))
            curr_s = next_s
            
        episodes.append(episode)
        
    return episodes

def td_linear(env, episodes, feature_extractor, ground_truth_V, alpha=0.0005, gamma=0.9):
    dim = feature_extractor.get_feature_dim()
    w = np.zeros(dim)
    
    rmse_history = []
    states = env.get_all_states()
    
    # 预计算所有状态的特征以加速 RMSE 计算
    state_features = {s: feature_extractor.get_features(s) for s in states}
    true_values = np.array([ground_truth_V[s] for s in states])
    
    for episode in episodes:
        for s, r, next_s in episode:
            phi_s = state_features[s]
            phi_next_s = state_features[next_s]
            
            v_s = np.dot(w, phi_s)
            v_next_s = np.dot(w, phi_next_s)
            
            td_target = r + gamma * v_next_s
            td_error = td_target - v_s
            
            w += alpha * td_error * phi_s
            
        # 计算 RMSE
        approx_values = np.array([np.dot(w, state_features[s]) for s in states])
        mse = np.mean((approx_values - true_values)**2)
        rmse = np.sqrt(mse)
        rmse_history.append(rmse)
        
    # 最终值
    final_values = {s: np.dot(w, state_features[s]) for s in states}
    return final_values, rmse_history, w
