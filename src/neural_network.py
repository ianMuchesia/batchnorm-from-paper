import numpy as np


class NeuralNetwork:
    def __init__(self,input_size, hidden_size):
        
        self.W = np.random.randn(input_size,hidden_size)
        
        self.b = np.zeros((hidden_size))
        
        
    def forward(self,x,batchnorm):
        Z1 = np.dot(x,self.W) + self.b
        
        print(f"this is the shape of Z1: {Z1.shape}")
        
        Z2 = batchnorm.forward(Z1)
        
        print(f"this is the shape of Z2: {Z2.shape}")

        
        H = np.maximum(0,Z2)
        
        return H