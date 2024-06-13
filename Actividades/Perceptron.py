import numpy as np


class MLP():
    def __init__(self, W, x, b, activation='linear'):
        
        self.W = W
        self.x = x
        self.b = b
    
    def nn(self):
        
        pass
    
    def shape_verification(self):
        self.Z = np.dot(self.W,self.X) + self.b
        
        
MLP().shape_verification()