import numpy as np
import itertools

class FeatureExtractor:
    def __init__(self, feature_type, order=None):
        self.feature_type = feature_type
        self.order = order
        
    def get_features(self, state):
        # state 是 (行, 列)
        # 归一化到 [0, 1]
        # 假设 5x5 网格, 最大索引是 4
        y = state[0] / 4.0 # 行
        x = state[1] / 4.0 # 列
        
        if self.feature_type == 'polynomial':
            return self._polynomial_features(x, y)
        elif self.feature_type == 'fourier':
            return self._fourier_features(x, y)
        else:
            raise ValueError("Unknown feature type")
            
    def _polynomial_features(self, x, y):
        # 阶数大致对应提示逻辑中的维度
        # 但提示直接指定了维度
        if self.order == 3: # R^3
            return np.array([1, x, y])
        elif self.order == 6: # R^6
            return np.array([1, x, y, x**2, y**2, x*y])
        elif self.order == 10: # R^10
            return np.array([1, x, y, x**2, y**2, x*y, x**3, y**3, (x**2)*y, x*(y**2)])
        else:
            raise ValueError("Unsupported polynomial dimension")

    def _fourier_features(self, x, y):
        q = self.order
        features = []
        for c1 in range(q + 1):
            for c2 in range(q + 1):
                c = np.array([c1, c2])
                s = np.array([x, y])
                val = np.cos(np.pi * np.dot(c, s))
                features.append(val)
        return np.array(features)

    def get_feature_dim(self):
        if self.feature_type == 'polynomial':
            return self.order
        elif self.feature_type == 'fourier':
            return (self.order + 1) ** 2
