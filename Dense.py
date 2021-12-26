import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# Dense layer
class Layer_Dense:
  # Layer initialization
  def __init__(
    self, n_inputs,
    n_neurons,
    weight_regularizer_l1=0,
    weight_regularizer_l2=0,
    bias_regularizer_l1=0,
    bias_regularizer_l2=0
  ):
    self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))
    # Set regularization strength
    self.weightRegularizerL1 = weight_regularizer_l1
    self.weightRegularizerL2 = weight_regularizer_l2
    self.biasRegularizerL1 = bias_regularizer_l1
    self.biasRegularizerL2 = bias_regularizer_l2
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

    # Gradient on regularization
    # L1 on weights
    if self.weightRegularizerL1 > 0 :
      dl1 = np.ones_like(self.weights)
      dl1[self.weights < 0] = -1
      self.dweights += self.weightRegularizerL1 * dl1
    
    # L2 on weights
    if self.weightRegularizerL2 > 0:
      self.dweights += 2 * self.weightRegularizerL2 * self.weights
    
    if self.biasRegularizerL1 > 0 :
      dl1 = np.ones_like(self.biases)
      dl1[self.biases < 0] = -1
      self.dbiases += self.biasRegularizerL1 * dl1
    
    if self.biasRegularizerL2 > 0:
      self.dbiases += 2 * self.biasRegularizerL2 * self.biases

    # Gradient on values
    self.dinputs = np.dot(dvalues, self.weights.T)
  


class LayerDropout:
  def __init__(self, rate = 0):
    # Store rate, we invert it as for example for dropout
    # of 0.1 we need success rate of 0.9
    self.rate = 1 - rate

  def forward(self, inputs):
    # Save inputs values
    self.inputs = inputs
    # Generate and save scaled mask
    self.binaryMask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
    # apply mask to output values
    self.output = inputs * self.binaryMask
  def backward(self, dvalues):
    self.dinputs = dvalues * self.binaryMask






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

  def regularizationLoss(self, layer):
    regularizationLoss = 0
    # L1 regularization - weights
    # calculate only when factor greater than 0
    if layer.weightRegularizerL1 > 0:
      regularizationLoss += layer.weightRegularizerL1 * np.sum(np.abs(layer.weights))
    # L2 regularization - weights
    if layer.weightRegularizerL2 > 0:
        regularizationLoss += layer.weightRegularizerL2 * np.sum(layer.weights * layer.weights)
    # L1 regularization - biases
    # calculate only when factor greater than 0
    if layer.biasRegularizerL1 > 0 :
      regularizationLoss += layer.biasRegularizerL1 * np.sum(np.abs(layer.biases))
    # L2 regularization - biases
    if layer.biasRegularizerL2 > 0:
      regularizationLoss += layer.biasRegularizerL2 * np.sum(layer.biases * layer.biases)
    return regularizationLoss
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


class Activation_Sigmoid:

  def forward(self, inputs):
    self.inputs = inputs
    self.output = 1 / (1 + np.exp(-inputs))
  
  def backward(self, dvalues):
    self.dinputs = dvalues * (1 - self.output) * self.output


class Loss_BinaryCrossentropy(Loss):
# Forward pass
  def forward(self, y_pred, y_true):
  # Clip data to prevent division by 0
  # Clip both sides to not drag mean towards any value
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    # Calculate sample-wise loss
    sample_losses = -(y_true * np.log(y_pred_clipped) +
    (1 - y_true) * np.log(1 - y_pred_clipped))
    sample_losses = np.mean(sample_losses, axis=-1)
    # Return losses
    return sample_losses
  # Backward pass
  def backward(self, dvalues, y_true):
  # Number of samples
    samples = len(dvalues)
    # Number of outputs in every sample
    # We'll use the first sample to count them
    outputs = len(dvalues[0])
    # Clip data to prevent division by 0
    # Clip both sides to not drag mean towards any value
    clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
    # Calculate gradient
    self.dinputs = -(y_true / clipped_dvalues -
    (1 - y_true) / (1 - clipped_dvalues)) / outputs
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





class OptimiAdagrad:
  # Initialize optimizer - set settings
  def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
    self.learning_rate = learning_rate
    self.currentLR = learning_rate
    self.decay = decay
    self.iterations = 0
    self.epsilon = epsilon
  
  def preUpdateParams(self):
    if self.decay:
      self.currentLR = self.learning_rate * (1. / (1. + self.decay * self.iterations))
  
  def updateParams(self, layer):
    # If layer does not contain cache arrays,
    # create them filled with zeros
    if not hasattr(layer, 'weightCache'):
      layer.weight_cache = np.zeros_like(layer.weights)
      layer.bias_cache = np.zeros_like(layer.biases)
    # Update cache with squared current gradients
    layer.weightCache += layer.dweights**2
    layer.biasCache += layer.dbiases**2
    # Vanilla SGD parameter update + normalization
    # with square rooted cache
    layer.weight += -self.currentLR * layer.dweights / (np.sqrt(layer.weightCache) + self.epsilon)
    layer.bias += -self.currentLR * layer.dweights / (np.sqrt(layer.biasCache) + self.epsilon)
  
  def postUpdateParams(self):
    self.iterations += 1





# Root Mean Square Propagation Optimizer
class OptimizerRMSprop:

  # init optimizer set Settings
  def __init__(self, learning_rate=0.01, decay=0, epsilon=1e-7, rho=0.9):
    self.learning_rate = learning_rate
    self.currentLR = learning_rate
    self.decay = decay
    self.iterations = 0
    self.epsilon = epsilon
    self.rho = rho

  def preUpdateParams(self):
    if self.decay:
      self.currentLR = self.learning_rate * (1. / (1. + self.decay * self.iterations))
  
  def updateParams(self, layer):

    if not hasattr(layer, 'weightCache'):
      layer.weightCache = np.zeros_like(layer.weights)
      layer.biasCache = np.zeros_like(layer.biases)
    # Update cache with squared current gradients
    layer.weightCache = self.rho * layer.weightCache + (1 - self.rho) * layer.dweights**2
    layer.biasCache = self.rho * layer.biasCache + (1 - self.rho) * layer.dbiases**2

    # Vanilla SGD parameter update + normalization
    # with square rooted cache
    layer.weights += -self.currentLR * layer.dweights / (np.sqrt(layer.weightCache) + self.epsilon)
    layer.biases += -self.currentLR * layer.dbiases / (np.sqrt(layer.biasCache) + self.epsilon)

  
  def postUpdateParams(self):
    self.iterations += 1


class Activation_linear:

  def forward(self, inputs):
    self.inputs = inputs
    self.output = inputs
  
  def backward(self, dvalues):
    self.dinputs = dvalues.copy()


class Loss_MeanSquaredError(Loss):
  def forward(self, yPred, yTrue):
    sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
    return sample_losses
  
  def backward(self, dvalues, yTrue):
    samples = len(dvalues)
    outputs = len(dvalues[0])

    self.dinputs = -2 * (yTrue - dvalues) / outputs
    self.dinputs = self.dinputs / samples

class Loss_MeanAbsoluteError(Loss):
  def forward(self, yPred, yTrue):
    sample_losses = np.mean(np.abs(yTrue, yPred), axis=-1)
    return sample_losses
  def backward(self, dvalues, yTrue):
    samples = len(dvalues)
    outputs = len(dvalues[0])
    self.dinputs = np.sign(yTrue - dvalues) / outputs
    self.dinputs = self.dinputs / samples

class OptimizerAdam:

  def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                beta_1=0.9, beta_2=0.999):
    self.learning_rate = learning_rate
    self.currentLR = learning_rate
    self.decay = decay
    self.iterations = 0
    self.epsilon = epsilon
    self.beta_1 = beta_1
    self.beta_2 = beta_2
  
  def preUpdateParams(self):
    if self.decay:
        self.currentLR = self.learning_rate * (1. / (1. + self.decay * self.iterations))
  
  def updateParams(self, layer):
    # If layer does not contain cache arrays,
    # create them filled with zeros
    if not hasattr(layer, 'weightCache'):
      layer.weightMomentums = np.zeros_like(layer.weights)
      layer.weightCache = np.zeros_like(layer.weights)
      layer.biasMomentums = np.zeros_like(layer.biases)
      layer.biasCache = np.zeros_like(layer.biases)
    # Update momentum with current gradient
    layer.weightMomentums = self.beta_1 * layer.weightMomentums + (1 - self.beta_1) * layer.dweights
    layer.biasMomentums = self.beta_1 * layer.biasMomentums + (1 - self.beta_1) * layer.dbiases
    # Get corrected momentum
    # self.iteration is 0 at first pass
    # and we need to start with 1 here
    weightMomentumsCorrected = layer.weightMomentums / (1 - self.beta_1 ** (self.iterations + 1))
    biasMomentumsCorrected = layer.biasMomentums / (1 - self.beta_1 ** (self.iterations + 1))
    # Update cache with squared current gradients
    layer.weightCache = self.beta_2 * layer.weightCache + (1 - self.beta_2) * layer.dweights ** 2
    layer.biasCache = self.beta_2 * layer.biasCache + (1 - self.beta_2) * layer.dbiases ** 2
    # Get corrected cache
    weightCacheCorrected = layer.weightCache / (1 - self.beta_2 ** (self.iterations + 1))
    biasCacheCorrected = layer.biasCache / (1 - self.beta_2 ** (self.iterations + 1))
    # Vanilla SGD parameter update + normalization
    # with square rooted cache
    layer.weights += -self.currentLR * weightMomentumsCorrected / (np.sqrt(weightCacheCorrected) + self.epsilon)
    layer.biases += -self.currentLR * biasMomentumsCorrected / (np.sqrt(biasCacheCorrected) + self.epsilon)

  def postUpdateParams(self):
    self.iterations += 1




  














# # Create dataset
# X, y = spiral_data(samples=100, classes=3)
# # Create Dense layer with 2 input features and 64 output values
# dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4,
# bias_regularizer_l2=5e-4)
# # Create ReLU activation (to be used with Dense layer):
# activation1 = Activation_ReLU()
# # Create second Dense layer with 64 input features (as we take output
# # of previous layer here) and 3 output values (output values)
# dense2 = Layer_Dense(64, 3)
# # Create Softmax classifier's combined loss and activation
# loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
# # Create optimizer
# optimizer = OptimizerAdam(decay=1e-5, learning_rate=0.02,)


# for epoch in range(10001):
#   dense1.forward(X)
#   activation1.forward(dense1.output)
#   dense2.forward(activation1.output)
#   loss = loss_activation.forward(dense2.output, y)
#   data_loss = loss_activation.forward(dense2.output, y)
#   regularization_loss = loss_activation.loss.regularizationLoss(dense1) + loss_activation.loss.regularizationLoss(dense2)
#   loss = data_loss + regularization_loss
#   predictions = np.argmax(loss_activation.output, axis=1)
#   if len(y.shape) == 2:
#     y = np.argmax(y, axis=1)
#   accuracy = np.mean(predictions == y)
#   if not epoch % 100:
#     print(f'Epoch :{epoch},' + f' accuracy :{accuracy:.3f},' + f' loss :{loss:.3f},' + f' lr: {optimizer.currentLR:.3f},'f'data_loss: {data_loss:.3f}, ' +f'reg_loss: {regularization_loss:.3f}), ')
  
#   loss_activation.backward(loss_activation.output, y)
#   dense2.backward(loss_activation.dinputs)
#   activation1.backward(dense2.dinputs)
#   dense1.backward(activation1.dinputs)

#   optimizer.preUpdateParams() 
#   optimizer.updateParams(dense1)
#   optimizer.updateParams(dense2)
#   optimizer.postUpdateParams()



# Create dataset
X, y = spiral_data(samples=100, classes=2)
# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y = y.reshape(-1, 1)
# Create Dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4,
bias_regularizer_l2=5e-4)
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()
# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 1 output value
dense2 = Layer_Dense(64, 1)
# Create Sigmoid activation:
activation2 = Activation_Sigmoid()
# Create loss function
loss_function = Loss_BinaryCrossentropy()
# Create optimizer
optimizer = OptimizerAdam(decay=5e-7)
# Train in loop
for epoch in range(10001):
# Perform a forward pass of our training data through this layer
  dense1.forward(X)
  # Perform a forward pass through activation function
  # takes the output of first dense layer here
  activation1.forward(dense1.output)
  # Perform a forward pass through second Dense layer
  # takes outputs of activation function
  # of first layer as inputs
  dense2.forward(activation1.output)
  # Perform a forward pass through activation function
  # takes the output of second dense layer here
  activation2.forward(dense2.output)
  # Calculate the data loss
  data_loss = loss_function.calculate(activation2.output, y)
  # Calculate regularization penalty
  regularization_loss = \
  loss_function.regularizationLoss(dense1) +  loss_function.regularizationLoss(dense2)
  # Calculate overall loss
  loss = data_loss + regularization_loss
  # Calculate accuracy from output of activation2 and targets
  # Part in the brackets returns a binary mask - array consisting
  # of True/False values, multiplying it by 1 changes it into array
  # of 1s and 0s
  predictions = (activation2.output > 0.5) * 1
  accuracy = np.mean(predictions==y)
  if not epoch % 100:
    print(f'epoch: {epoch}, ' +
    f'acc: {accuracy:.3f}, '+
    f'loss: {loss:.3f} (' +
    f'data_loss: {data_loss:.3f}, ' +
    f'reg_loss: {regularization_loss:.3f}), ' +
    f'lr: {optimizer.currentLR}')
  # Backward pass
  loss_function.backward(activation2.output, y)
  activation2.backward(loss_function.dinputs)
  dense2.backward(activation2.dinputs)
  activation1.backward(dense2.dinputs)
  dense1.backward(activation1.dinputs)
  # Update weights and biases
  optimizer.preUpdateParams() 
  optimizer.updateParams(dense1)
  optimizer.updateParams(dense2)
  optimizer.postUpdateParams()


# Validate the model
# Create test dataset
X_test, y_test = spiral_data(samples=100, classes=2)
# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y_test = y_test.reshape(-1, 1)
# Perform a forward pass of our testing data through this layer
dense1.forward(X_test)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Perform a forward pass through activation function
# takes the output of second dense layer here
activation2.forward(dense2.output)
# Calculate the data loss
loss = loss_function.calculate(activation2.output, y_test)
# Calculate accuracy from output of activation2 and targets
# Part in the brackets returns a binary mask - array consisting of
# True/False values, multiplying it by 1 changes it into array
# of 1s and 0s
predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions==y_test)
print('Test Data')
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
print(predictions)