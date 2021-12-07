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
  


class Activation_ReLU:
  def forward(self, inputs):
    # Calculate output values from inputs
    self.output = np.maximum(0, inputs)
  # Backward pass
  def backward(self, dvalues):
    # Since we need to modify the original variable,
    # let's make a copy of the values first
    self.dinputs = dvalues.copy()
    # Zero gradient where input values were negative
    self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
  def forward(self, inputs):
    # Get unnormalized probabilities
    expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    # Normalize them for each sample
    probabilities = expValues / np.sum(expValues, axis=1, keepdims=True)
    self.output = probabilities
  
  def backward(self, dvalues):
    self.dinputs = np.empty_like(dvalues)
    # Enumerate outputs and gradients
    for i , (singleOutput, singleDvalues) in enumerate(zip(self.output, dvalues)):
      # Flatten output array
      singleOutput = singleOutput.reshape(-1, 1)
      # Calculate Jacobian matrix of the output and
      jacobianMatrix = np.diagflat(singleOutput) - np.dot(singleOutput,single_output.T)
      # Calculate sample-wise gradient
      # and add it to the array of sample gradients
      self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
  # Calculates the data and regularization losses
  # given model output and ground truth values
  def calculate(self, output, y):
    # Calculate sample losses
    sampLosses = self.forward(output, y)
    # Calculate mean loss
    dataLoss = np.mean(sampLosses)
    # Return loss
    return dataLoss


# Cross-entropy loss
class lossCateCrossEntropy(Loss):
  #forward Pass
  def forward(self, yPred, yTrue):
    # Number of samples in a batch
    samples = len(yPred)
    # Clip data to prevent division by 0
    # Clip both sides to not drag mean towards any value
    yPredClipped = np.clip(yPred, 1e-7, 1 - 1e-7)
    # Probabilities for target values -
    # only if categorical labels
    if len(yTrue.shape) == 1:
      correctConfidence = yPredClipped[range(samples), yTrue]
    # Mask values - only for one-hot encoded labels
    elif len(yTrue.shape) == 2:
      correctConfidence = np.sum(yPredClipped * yTrue, axis=1)
    #Losses
    nigativeLog = -np.log(correctConfidence)
    return nigativeLog

  def backward(self, dvalues,  yTrue):
    # Number of samples
    samples = len(dvalues)
    # Number of labels in every sample
    # We'll use the first sample to count them
    labels = len(dvalues[0])
    # If labels are sparse, turn them into one-hot vector
    if(len(yTrue.shape) == 1):
      yTrue = np.eye(labels)[yTrue]
    # Calculate gradient
    self.dinputs = -y_true / dvalues
    # Normalize gradient
    self.dinputs = self.dinputs / samples



# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

  # create activation and loss objects
  def __init__(self):
    self.activation = Activation_Softmax()
    self.loss = lossCateCrossEntropy()
  
  def forward(self, input, yTrue):
    #output layer activation function
    self.activation.forward(input)
    #the Output
    self.output = self.activation.output
    # Calculate and return loss value
    return self.loss.calculate(self.output, yTrue)
  
  def backward(self, dvalues, yTrue):
    samples = len(dvalues)
    # If labels are one-hot encoded,
    # turn them into discrete values
    if len(y_true.shape) == 2:
      y_true = np.argmax(y_true, axis=1)
    # Copy so we can safely modify
    self.dinputs = dvalues.copy()
    # Calculate gradient
    self.dinputs[range(samples), y_true] -= 1
    # Normalize gradient
    self.dinputs = self.dinputs / samples





















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
