import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# Dense layer
class Layer_Dense:
  # Layer initialization
  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))
  # Forward pass
  def forward(self, inputs):
    # Calculate output values from inputs, weights and biases
    self.output = np.dot(inputs, self.weights) + self.biases
  
  # Backward pass
  def backward(self, dvalues):
    # Gradients on parameters
    self.dweights = np.dot(self.inputs.T, dvalues)
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
    # Gradient on values
    self.dinputs = np.dot(dvalues, self.weights.T)
  
















# neg_log = -np.log(correct_confidences)
# verage_loss = np.mean(neg_log)
# print(average_loss)

# # Create dataset
# X, y = spiral_data(samples=100, classes=3)
# # Create Dense layer with 2 input features and 3 output values
# dense1 = Layer_Dense(2, 3)
# # Create ReLU activation (to be used with Dense layer):
# activation1 = Activation_ReLU()
# # Create second Dense layer with 3 input features (as we take output
# # of previous layer here) and 3 output values
# dense2 = Layer_Dense(3, 3)
# # Create Softmax activation (to be used with Dense layer):
# activation2 = Activation_Softmax()
# # Create loss function
# loss_function = lossCateCrossEntropy()
# # Perform a forward pass of our training data through this layer
# dense1.forward(X)
# # Perform a forward pass through activation function
# # it takes the output of first dense layer here
# activation1.forward(dense1.output)
# # Perform a forward pass through second Dense layer
# # it takes outputs of activation function of first layer as inputs
# dense2.forward(activation1.output)
# # Perform a forward pass through activation function
# # it takes the output of second dense layer here
# activation2.forward(dense2.output)
# # Let's see output of the first few samples:
# print(activation2.output[:5])
# # Perform a forward pass through loss function
# # it takes the output of second dense layer here and returns loss
# loss = loss_function.calculate(activation2.output, y)
# # Print loss value
# print('loss:', loss)
