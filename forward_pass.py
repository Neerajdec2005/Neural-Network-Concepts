import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# X = [[1.3, 3.2, 4.5, 7.2],[-2.1, 4.3, 5.1, 6.8],[3.2, -5.1, 6.2, -2.1]]

X, y= spiral_data(100, 3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights= 0.1*np.random.randn( n_inputs, n_neurons)
        self.biases= np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output= np.dot(inputs, self.weights) + self.biases

class ReLU_Activation:
    def forward(self, inputs):
        self.output= np.maximum(0, inputs)

class Softmax_Activation:
    def forward(self, inputs):
        exp_values= np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        normalize= exp_values/ np.sum(exp_values, axis=1, keepdims=True)
        self.output= normalize

dense1= Layer_Dense(2, 5)
activation= ReLU_Activation()

dense2= Layer_Dense(5, 3)
activation2= Softmax_Activation()

dense1.forward(X)
activation.forward(dense1.output)

dense2.forward(activation.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

