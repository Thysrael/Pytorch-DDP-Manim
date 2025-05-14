import numpy as np
import pandas as pd

# Activation functions
def relu(X):
    return np.maximum(0,X)

def softmax(X):
    return np.exp(X) / sum(np.exp(X))

# Calculates the output of a given layer
def calculate_layer_output(w, prev_layer_output, b, actv_func):
    # Steps 1 & 2
    g = w @ prev_layer_output + b

    # Step 3
    return actv_func(g)

# Initialize weights & biases
def init_layer_params(row, col):
    w = np.random.randn(row, col)
    b = np.random.randn(row, 1)
    return w, b

# Calculate ReLU derivative
def relu_derivative(g):
    # g shape: (n, 1)
    derivative = g.copy()
    derivative[derivative <= 0] = 0
    derivative[derivative > 0] = 1
    # derivative.T[0] shape: (n) is the correct shape for np.diag()
    return np.diag(derivative.T[0])

# Calculate Softmax derivative
def softmax_derivative(o):
    o = o.flatten()
    return np.diag(o) - np.outer(o, o)

def layer_backprop(previous_derivative, layer_output, previous_layer_output
                   , w, actv_func):
    # 1. Calculate the derivative of the activation func
    dh_dg = None
    if actv_func is relu:
        dh_dg = relu_derivative(layer_output)
    elif actv_func is softmax:
        dh_dg = softmax_derivative(layer_output)

    # 2. Apply chain rule to get derivative of Loss function with respect to:
    dL_dg = dh_dg @ previous_derivative # activation function

    # 3. Calculate the derivative of the linear function with respect to:
    dg_dw = previous_layer_output.T     # a) weight matrix
    dg_dh = w.T                         # b) previous layer output
    dg_db = 1.0                         # c) bias vector

    # 4. Apply chain rule to get derivative of Loss function with respect to:
    dL_dw = dL_dg @ dg_dw               # a) weight matrix
    dL_dh = dg_dh @ dL_dg               # b) previous layer output
    dL_db = dL_dg * dg_db               # c) bias vector

    return dL_dw, dL_dh, dL_db

def gradient_descent(w, b, dL_dw, dL_db, learning_rate):
    w -= learning_rate * dL_dw
    b -= learning_rate * dL_db
    return w, b

def get_prediction(o):
    return np.argmax(o)

# Compute Accuracy (%) across all training data
def compute_accuracy(train, label, w1, b1, w2, b2, w3, b3):
    # Set params
    correct = 0
    total = train.shape[0]

    # Iterate through training data
    for index in range(0, train.shape[0]):
        # Select a single data point (image)
        X = train[index: index + 1,:].T

        # Forward pass: compute Output/Prediction (o)
        h1 = calculate_layer_output(w1, X, b1, relu)
        h2 = calculate_layer_output(w2, h1, b2, relu)
        o = calculate_layer_output(w3, h2, b3, softmax)

        # If prediction matches label Increment correct count
        if label[index] == get_prediction(o):
            correct += 1

    # Return Accuracy (%)
    return (correct / total) * 100

# Set hyperparameter(s)
learning_rate = 0.01
epoch = 0
accuracy = 0

# Extract MNIST csv data into train & test variables
# 28 x 28 = 784 pixel image
train = np.array(pd.read_csv('./digit-recognizer/train.csv', delimiter=','))
# just use nlp
test = np.array(pd.read_csv('./digit-recognizer/test.csv', delimiter=','))

# Check shape of train & test datasets
# (image_id, pixels)
print(f'train shape: {train.shape}')
print(f'test shape: {test.shape}')

# Extract the first column of the training dataset into a label array
label = train[:,0]

# The training dataset now becomes all columns except the first
train = train[:,1:]

# Initialise vector of all zeroes with 10 columns and the same number of
# rows as the label array
Y = np.zeros((label.shape[0], 10))

# assign a value of 1 to each column index matching the label value
# Y is a one-hot array
Y[np.arange(0,label.shape[0]), label] = 1.0

# normalize test & training dataset
# the items in train and test ranges from 0 to 255
train = train / 255
test = test / 255

# Randomly initialize weights & biases for each layer
np.random.seed(42)
w1, b1 = init_layer_params(10, 784)
w2, b2 = init_layer_params(10, 10)
w3, b3 = init_layer_params(10, 10)

# While:
#  1. Accuracy is improving by 1% or more per epoch, and
#  2. There are 20 epochs or less
while accuracy <= 85 and epoch <= 20:
    print(f'------------- Epoch {epoch} -------------')

    # Iterate through training data
    for index in range(0, train.shape[0]):
        # Select a single image and associated y vector
        X = train[index:index+1,:].T
        y = Y[index:index+1].T

        # 1. Forward pass: compute Output/Prediction (o)
        h1 = calculate_layer_output(w1, X, b1, relu)
        h2 = calculate_layer_output(w2, h1, b2, relu)
        o = calculate_layer_output(w3, h2, b3, softmax)

        # 2. Compute Loss Vector
        L = np.square(o - y)

        # 3. Backpropagation
        # Compute Loss derivative w.r.t. Output/Prediction vector (o)
        dL_do = 2.0 * (o - y)

        # Compute Output Layer derivatives
        dL3_dw3, dL3_dh2, dL3_db3 = layer_backprop(dL_do, o, h2, w3, softmax)
        # Compute Hidden Layer 2 derivatives
        dL2_dw2, dL2_dh2, dL2_db2 = layer_backprop(dL3_dh2, h2, h1, w2, relu)
        # Compute Hidden Layer 1 derivatives
        dL1_dw1, _, dL1_db1 = layer_backprop(dL2_dh2, h1, X, w1, relu)

        # 4. Update weights & biases
        w1, b1 = gradient_descent(w1, b1, dL1_dw1, dL1_db1, learning_rate)
        w2, b2 = gradient_descent(w2, b2, dL2_dw2, dL2_db2, learning_rate)
        w3, b3 = gradient_descent(w3, b3, dL3_dw3, dL3_db3, learning_rate)


    # Compute & print Accuracy (%)
    accuracy = compute_accuracy(train, label, w1, b1, w2, b2, w3, b3)
    print(f'Accuracy: {accuracy:.2f}%')

    # Increment epoch
    epoch += 1
