import numpy as np

class BatchNorm:
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon
        
        self.gamma = None
        self.beta = None
        
        
        
    def forward(self,x):
        
        
        
        batch_mean = np.mean(x,axis=0,keepdims=True)
        
        batch_variance = np.var(x,axis=0,keepdims=True)
        
        normalized = (x - batch_mean) / np.sqrt(batch_variance + self.epsilon)
        
        if(self.gamma is None):
            num_features = x.shape[1]
            
            self.gamma = np.ones((1,num_features))
            self.beta = np.zeros((1,num_features))
        
        
        scaled_x =  normalized * self.gamma + self.beta
        
        return scaled_x

        
        
        
        
    