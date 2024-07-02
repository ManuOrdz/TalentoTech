import numpy as np
from collections import defaultdict
from scipy.special import expit
#Single Perceptron Layer
#Feedfoward :X->y
# To implement: Backforward, Gradients, Errors:MSE, Biases actualization
class Network():
    def __init__(self, sizes, activation):
        
        self.activation = 'sigmoid'
        self.num_layers = len(sizes)
        self.sizes = sizes        
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedfoward(self, a):
        """Return the output of the network if "a" is input.

        Args:
            a (_type_): _description_
        """
        for b,w in zip(self.biases,self.weights):
            a = np.dot(w,a)+b
            a = self.activation_function(a)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
    
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in np.arange(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in np.arange(0,n,mini_batch_size)]
        if test_data:
            print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            
        else:
            print(f"Epoch {j} complete.")
    
    def update_mini_batch(self, mini_batch, eta):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]
            
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_W = [np.zeros(w.shape) for w in self.weights]
        
        activation = x
        activations = [x]
        zs = []        
        activation = self.activation_function(z,)
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)
            
    def activation_function(self, z):
        if self.activation == 'linear':
            return z
        
        if self.activation == 'relu':
            return np.maximum(0, z) 
        
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
            
        if self.activation == 'tanh':
            return  np.tanh(z)
