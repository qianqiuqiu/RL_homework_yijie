import numpy as np

class FeatureExtractor:
    def __init__(self, feature_type='polynomial', order=3, grid_size=(5,5)):
        self.feature_type = feature_type
        self.order = order
        self.grid_size = grid_size
        
        if self.feature_type == 'fourier':
            # Precompute coefficients for Fourier features
            self.fourier_coeffs = np.array([[c1, c2] 
                                          for c1 in range(self.order + 1) 
                                          for c2 in range(self.order + 1)])
        
    def get_features(self, state):
        # state is (row, col)
        # Normalize to [0, 1]
        y = state[0] / (self.grid_size[0] - 1.0)
        x = state[1] / (self.grid_size[1] - 1.0)
        
        if self.feature_type == 'polynomial':
            return self._polynomial_features(x, y)
        elif self.feature_type == 'fourier':
            return self._fourier_features(x, y)
        else:
            raise ValueError("Unknown feature type")
            
    def _polynomial_features(self, x, y):
        if self.order == 3: # R^3
            return np.array([1, x, y])
        elif self.order == 6: # R^6
            return np.array([1, x, y, x**2, y**2, x*y])
        elif self.order == 10: # R^10
            return np.array([1, x, y, x**2, y**2, x*y, x**3, y**3, (x**2)*y, x*(y**2)])
        else:
            # Generic polynomial expansion could be implemented here
            return np.array([1, x, y])

    def _fourier_features(self, x, y):
        # Vectorized implementation
        s = np.array([x, y])
        # Ensure coefficients exist (in case order was changed or init skipped)
        if not hasattr(self, 'fourier_coeffs') or len(self.fourier_coeffs) != (self.order + 1)**2:
             self.fourier_coeffs = np.array([[c1, c2] 
                                          for c1 in range(self.order + 1) 
                                          for c2 in range(self.order + 1)])
        
        return np.cos(np.pi * np.dot(self.fourier_coeffs, s))

    def get_feature_dim(self):
        if self.feature_type == 'polynomial':
            return self.order
        elif self.feature_type == 'fourier':
            return (self.order + 1) ** 2
