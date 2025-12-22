import numpy as np

def generate_data(num_samples=400):
    """
    生成来自二维均匀分布的随机样本。
    
    该分布定义在一个以 (0,0) 为中心、边长为 30 的正方形区域上。
    因此，x 和 y 坐标的范围在 [-15, 15] 之间。
    
    参数:
        num_samples (int): 要生成的样本数量。默认为 400。
        
    返回:
        np.ndarray: 一个包含生成样本的 (num_samples, 2) 数组。
    """
    low = -15
    high = 15
    # 独立生成 x 和 y 坐标的样本
    samples = np.random.uniform(low, high, size=(num_samples, 2))
    return samples

if __name__ == "__main__":
    # 测试生成器
    data = generate_data()
    print(f"Generated data shape: {data.shape}")
    print(f"Min values: {data.min(axis=0)}")
    print(f"Max values: {data.max(axis=0)}")
