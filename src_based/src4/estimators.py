import numpy as np

def calculate_analytical_mean():
    """
    计算解析期望值 E[X]。
    由于分布在 [-15, 15] x [-15, 15] 上是均匀的，
    质心位于 (0, 0)。
    
    返回:
        np.ndarray: 期望值 (0, 0)。
    """
    return np.array([0.0, 0.0])

def sgd_estimator(data, lr_strategy='A', num_iterations=200, initial_w=None):
    """
    使用随机梯度下降 (SGD) 估计 E[X]。
    
    更新规则: w_{k+1} = w_k - alpha_k * (w_k - x_k)
    
    参数:
        data (np.ndarray): 样本数据集。
        lr_strategy (str): 'A' 代表 alpha_k = 1/k, 'B' 代表 alpha_k = 0.005。
        num_iterations (int): 迭代次数。
        initial_w (list or np.ndarray): w 的初始猜测值。默认为 [50, 50]。
        
    返回:
        list: 包含 w 估计值序列的 numpy 数组列表。
    """
    if initial_w is None:
        w = np.array([50.0, 50.0])
    else:
        w = np.array(initial_w, dtype=float)
        
    w_history = [w.copy()]
    n_samples = data.shape[0]
    
    for k in range(1, num_iterations + 1):
        # 随机选择一个样本
        idx = np.random.randint(0, n_samples)
        x_k = data[idx]
        
        # 确定学习率
        if lr_strategy == 'A':
            alpha_k = 1.0 / k
        elif lr_strategy == 'B':
            alpha_k = 0.005
        else:
            raise ValueError("Unknown learning rate strategy. Use 'A' or 'B'.")
            
        # 更新规则: w_{k+1} = w_k - alpha_k * (w_k - x_k)
        # 1/2 ||x - w||^2 关于 w 的梯度是 -(x - w) = w - x。
        # 所以 w_{k+1} = w_k - alpha * (w_k - x_k) 是正确的。
        w = w - alpha_k * (w - x_k)
        
        w_history.append(w.copy())
        
    return w_history

def mbgd_estimator(data, batch_size=10, num_iterations=200, initial_w=None):
    """
    使用小批量梯度下降 (MBGD) 估计 E[X]。
    
    更新规则: w_{k+1} = w_k - alpha_k * (1/m) * sum(w_k - x_{k,j})
    
    参数:
        data (np.ndarray): 样本数据集。
        batch_size (int): 批量大小 (m)。
        num_iterations (int): 迭代次数。
        initial_w (list or np.ndarray): w 的初始猜测值。默认为 [50, 50]。
        
    返回:
        list: 包含 w 估计值序列的 numpy 数组列表。
    """
    if initial_w is None:
        w = np.array([50.0, 50.0])
    else:
        w = np.array(initial_w, dtype=float)
        
    w_history = [w.copy()]
    n_samples = data.shape[0]
    
    for k in range(1, num_iterations + 1):
        # 随机选择一批样本
        indices = np.random.choice(n_samples, batch_size, replace=True) # 在有限数据集上进行 SGD/MBGD 时，通常使用有放回采样
        batch_samples = data[indices]
        
        # 确定学习率 (在此任务中 MBGD 固定为 1/k)
        alpha_k = 1.0 / k
        
        # 计算梯度估计: (1/m) * sum(w_k - x_{k,j})
        # 这等同于 w_k - mean(batch_samples)
        gradient_estimate = w - np.mean(batch_samples, axis=0)
        
        # 更新规则
        w = w - alpha_k * gradient_estimate
        
        w_history.append(w.copy())
        
    return w_history
