#### Libraries
from __future__ import annotations
from abc import abstractmethod, ABC
import numpy as np

# Abstract Class Network
class Network(ABC):
    @abstractmethod
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
        pass
    @abstractmethod
    def forward(self, inputs):
        pass
    @abstractmethod
    def backward(self, dvalues):
        pass

# Abstract Class Activation
class Activation(ABC):
    @abstractmethod
    def forward(self, inputs):
        pass
    @abstractmethod
    def backward(self, dvalues):
        pass

# Abstract Class Loss
class Loss():
    def calulate(self, output, y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class CategoricalCrossentrotpy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples), 
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped*y_true, 
                axis=1)
        negative_log_likehoods = -np.log(correct_confidences)
        return negative_log_likehoods
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
    
#Dense Network
class Dense(Network):
    def __init__(self, n_inputs, n_neurons):
        super().__init__(n_inputs, n_neurons)
        pass

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        #Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        #Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        
#Relu activation
class Relu(Activation):
    def forward(self,inputs):
        self.inputs = inputs.copy()
        self.output = np.maximum(0,inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <=0 ] = 0
# Softmax activation
class Softmax(Activation):
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1,
                                            keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)
        self.output = probabilities
        pass

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        
        for index, (single_output, single_dvalues) in \
            enumerate(zip(self,output,dvalues)):
                
                single_output = single_output.reshape(-1,1)
                
                jacobian_matrix = np.diagflat(single_output) - \
                    np.dot(single_output, single_output.T)
                    
                self.dinputs[index] =np.dot(jacobian_matrix,
                                            single_dvalues)

class SoftmaxCategoricalCrossentropy():
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossentrotpy()
        
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calulate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true,axis=1)
            
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
    
class SGD(object):
    def __init__(self, learning_rate = 1., decay = 0., momentum = 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
                
    def update_params(self, layer):
        
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights 
            layer.weight_momentums = weight_updates
            
            
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * \
                layer.dweights
            bias_updates = -self.current_learning_rate * \
                layer.dbiases
                
        layer.weights += weight_updates
        layer.biases += bias_updates
        
    def post_update_params(self):
        self.iterations += 1