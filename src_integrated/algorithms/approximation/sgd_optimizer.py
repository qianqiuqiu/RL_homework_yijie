import numpy as np

class SGDOptimizer:
    def __init__(self, learning_rate=0.01, decay_type=None):
        """
        Args:
            learning_rate: Initial learning rate.
            decay_type: 'inverse' (1/k) or None (constant).
        """
        self.learning_rate = learning_rate
        self.decay_type = decay_type
        self.iterations = 0

    def update(self, w, gradient):
        self.iterations += 1
        
        if self.decay_type == 'inverse':
            alpha = 1.0 / self.iterations # Or self.learning_rate / self.iterations
        else:
            alpha = self.learning_rate
            
        w = w - alpha * gradient
        return w
