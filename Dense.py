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
    self.inputs = inputs
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
    self.inputs = inputs
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
      self.dinputs[index] = np.dot(jacobianMatrix, singleDvalues)


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
    if len(yTrue.shape) == 2:
      yTrue = np.argmax(yTrue, axis=1)
    # Copy so we can safely modify
    self.dinputs = dvalues.copy()
    # Calculate gradient
    self.dinputs[range(samples), yTrue] -= 1
    # Normalize gradient
    self.dinputs = self.dinputs / samples




class OptimizerGradientD:
  # Initialize optimizer - set settings,
  # learning rate of 1. is default for this optimizer
  def __init__(self,learning_rate=1.0, decay=0., momentum=0.):
    self.learning_rate = learning_rate
    self.currentLR = learning_rate
    self.decay = decay
    self.iterations = 0
    self.momentum = momentum

  # Call once before any params updates
  def preUpdateParams(self,):
    if self.decay:
      self.currentLR = self.learning_rate * (1./ (1. + self.decay * self.iterations))

  # Update Parameters
  def updateParams(self, layer):
    # If we use momentum
    if self.momentum:
      # If layer does not contain momentum arrays, create them
      # filled with zeros
      if not hasattr(layer, 'weightMomentums'):
        layer.weightMomentums = np.zeros_like(layer.weights)
        # If there is no momentum array for weights
        # The array doesn't exist for biases yet either.
        layer.biasMomentums = np.zeros_like(layer.biases)
      # Build weight updates with momentum - take previous
      # updates multiplied by retain factor and update with
      # current gradients
      weightUpdates = self.momentum * layer.weightMomentums - self.currentLR * layer.dweights
      layer.weightMomentums = weightUpdates
      # Build bias updates
      biasUpdates = self.momentum * layer.biasMomentums - self.currentLR * layer.dbiases
      layer.biasMomentums = biasUpdates
    else:
      weightUpdates = -self.currentLR * layer.dweights
      biasUpdates = - self.currentLR * layer.dbiases
    # Update weights and biases using either
    # vanilla or momentum updates
    layer.weights += weightUpdates
    layer.biases += biasUpdates

    # layer.weights += -self.currentLR * layer.dweights
    # layer.biases += -self.currentLR * layer.dbiases

  # Call once after any param updates
  def postUpdateParams(self):
    self.iterations += 1



# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 64)
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()
# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(64, 3)
# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
# Create optimizer
optimizer = OptimizerGradientD(decay=1e-3, momentum=0.9)


for epoch in range(10001):
  dense1.forward(X)
  activation1.forward(dense1.output)
  dense2.forward(activation1.output)
  loss = loss_activation.forward(dense2.output, y)
  predictions = np.argmax(loss_activation.output, axis=1)
  if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
  accuracy = np.mean(predictions == y)
  if not epoch % 100:
    print(f'Epoch :{epoch},' + f' accuracy :{accuracy:.3f},' + f' loss :{loss:.3f},' + f' lr: {optimizer.currentLR:.3f},')
  
  loss_activation.backward(loss_activation.output, y)
  dense2.backward(loss_activation.dinputs)
  activation1.backward(dense2.dinputs)
  dense1.backward(activation1.dinputs)

  optimizer.preUpdateParams() 
  optimizer.updateParams(dense1)
  optimizer.updateParams(dense2)
  optimizer.postUpdateParams()