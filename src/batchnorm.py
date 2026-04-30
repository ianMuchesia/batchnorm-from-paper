import numpy as np

class BatchNorm:
    def __init__(self, epsilon=1e-5,momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        
        self.gamma = None
        self.beta = None
        
        
        self.running_mean = None
        self.running_var = None
        
        
        
    def forward(self,x,mode="train"):
        
        
        if self.gamma is None:
            num_features = x.shape[1]
            self.gamma = np.ones((1,num_features))
            self.beta = np.zeros((1,num_features))
            
            self.running_mean = np.zeros((1,num_features))
            self.running_var = np.ones((1,num_features))
            
            
        if mode == 'train':
            self.x = x
            self.batch_mean = np.mean(x,axis=0,keepdims=True)
            self.batch_variance = np.var(x,axis=0,keepdims=True)
            
            self.running_mean = (self.momentum * self.running_mean) + (1-self.momentum) * self.batch_mean
            
            self.running_var  = (self.momentum * self.running_var) + (1-self.momentum) * self.batch_variance
        
        
            self.normalized = (x - self.batch_mean) / np.sqrt(self.batch_variance + self.epsilon)
            
            
        elif mode == 'eval':
            self.normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            
        scaled_x =  self.normalized * self.gamma + self.beta
        
        return scaled_x
        
        # self.x = x
        # self.batch_mean = np.mean(x,axis=0,keepdims=True)
        
        # self.batch_variance = np.var(x,axis=0,keepdims=True)
        
        # self.normalized = (x - self.batch_mean) / np.sqrt(self.batch_variance + self.epsilon)
        
        # if(self.gamma is None):
        #     num_features = x.shape[1]
            
        #     self.gamma = np.ones((1,num_features))
        #     self.beta = np.zeros((1,num_features))
        
        
        # scaled_x =  self.normalized * self.gamma + self.beta
        
        # return scaled_x
    
    
    def backward(self,dout):
        
        d_normalized = dout * self.gamma
        d_beta = dout
        d_gamma = dout * self.normalized
        
       
        
        
        print(f"The shape of d_normalized: {d_normalized.shape}")
        
        inv_std = 1/ np.sqrt(self.batch_variance + self.epsilon)
        
        print(f"The shape of batch_mean : {self.batch_mean.shape}")
        
        d_var = np.sum(d_normalized * ( self.x -self.batch_mean) * -0.5 * (inv_std**3),axis=0)
        
        print(f"The shape of d_var : {d_var.shape}")
        
        d_mean = np.sum(d_normalized* - (self.batch_variance + self.epsilon)**-0.5,axis = 0) + d_var * -2/self.x.shape[0] * np.sum(self.x - self.batch_mean,axis=0)
        
        
        print(f"The shape of d_mean :{d_mean.shape}")
        
        d_x = d_normalized* (self.batch_variance + self.epsilon)**-0.5 + d_var * 2/self.x.shape[0] * (self.x - self.batch_mean) + d_mean * 1/self.x.shape[0]
        
        dx_beta_analytical = np.sum(dout,axis=0,keepdims=True)
        
        dx_gamma_analytical = np.sum(d_gamma,axis=0,keepdims=True)
        
        #alternative best backprop one liner
        n = dout.shape[0]
        
        dx = (self.gamma/(n*np.sqrt(self.batch_variance + self.epsilon))) * ((n * dout)-(np.sum(dout,axis=0,keepdims=True))-(self.normalized * np.sum(dout*self.normalized,axis=0,keepdims=True)))
        
        return d_x,dx_beta_analytical,dx_gamma_analytical,dx
        
    