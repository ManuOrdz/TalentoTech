import numpy as np
from scipy.special import expit

class MLP():
    def __init__(self, Weights, x, biases, activation='linear'):
        
        self.W = Weights
        self.x = x
        self.b = biases
        self.activation = activation
        
        self.output()
        self.activation_function()
        self.get_output()
        
    def output(self):
        self. y = np.dot(self.W,self.x) + self.b
    
    def activation_function(self):
        if self.activation == 'linear':
            self.y = self.y
        
        if self.activation == 'relu':
            self.y = np.maximum(0, self.y) 
        
        if self.activation == 'sigmoid':
            self.y = expit(self.y)
            self.y = self.y / self.y.sum()    
            
        if self.activation == 'tanh':
            self.y = np.tanh(self.y)
        
    def get_output(self):
        print(f'Salida MLP activacion:{self.activation}')
        print(self.y)
        
W = np.array([[1,-1,1],[1,1,0],[0,1,1],[1,0,1]])
x = np.array([2, 1, 3])
b = np.array([-5, 0, 1, -2])

# Test Linear
MLP(Weights=W, x=x, biases=b, activation='linear')
# Test Relu
MLP(Weights=W, x=x, biases=b, activation='relu')
# Test sigmoid
MLP(Weights=W, x=x, biases=b, activation='sigmoid')
# Test tanh
MLP(Weights=W, x=x, biases=b, activation='tanh')