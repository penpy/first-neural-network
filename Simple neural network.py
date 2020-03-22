# coding: utf-8

""" Neural network with 1 hidden layer for MNIST handwritten digits recognition
===========
PRESENTATION

This Python code is an example of a simple artificial neural network
written from scratch using only :
    - the numpy package (for array manipulation)
    - the mnist module (to import the database)

Make sure these two modules (from the Pypi library) are installed.
The MNIST database of handwritten digits is used to train the network

===========
STRUCTURE OF THE NEURAL NETWORK

This is a forward propagating network with 3 layers :
    - the input layer has 784 nodes (28 x 28 input images are flattened)
    - the hidden layer has 16 nodes
    - the output layer has 10 nodes

Activation functions used:
    - Sigmoid for the hidden layer
    - Softmax for the output layer

The loss is calculated with cross entropy

===========
USAGE GUIDE

1.  Run the train() function to train the neural network
    For each iteration a batch of 32 images is processed
    At the end of each iteration the weights and biases are updated
    
2.  Run the accuracy() function to see the network's performance

3.  Feel free to change the network's parameters and try to improve accuracy

===========
NOTATIONS

x : input layer
h : hidden layer (before activation)
ha : hidden layer (after activation)
y : output layer (before activation)
ya : output layer (after activation)

w1, w2 : weight matrices 1 and 2
b1, b2 : bias vectors 1 and 2

(u represents one of the parameters above)
len_u : length of vector u
shape_u : shape of matrix u
d_u : derivative of the loss function with respect to u (for a single result)
sum_d_u : derivative of the loss function with respect to u for multiple results in a batch

batch_size : number of training samples per batch
n_iterations : total number of iterations for the training process
learn_r : learn rate
n_tests : number of image used to calculate the accuracy

t : target digit (an integer between 0 and 9)

===========
"""

import numpy as np
import mnist


# =====================
# Collecting the MNIST dataset of handwritten digits
# =====================

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()


# =====================
# Neural network
# =====================

# Length of each layers (input x, hidden h, output y)
len_x = 28 * 28
len_h = 16
len_y = 10

# Shapes of weight matrices
shape_w1 = (len_x, len_h)
shape_w2 = (len_h, len_y)

# Initialization of weight matrices (w1 and w2) with random numbers
w1 = np.random.uniform(-1, 1, shape_w1) / np.sqrt(len_x)
w2 = np.random.uniform(-1, 1, shape_w2) / np.sqrt(len_h)

# Initialization of bias vectors (b1 and b2) with zeros
b1 = np.full(len_h, 0.)
b2 = np.full(len_y, 0.)

# Training parameters
n_iterations = 200
batch_size = 32
learn_r = 0.05

# Number of test images used to calculate the accuracy
n_tests = 500


def forward_propagation(image):
    # Returns the vectors of each layers for a given image
    
    # Input layer
    x = image.flatten() / 255
    
    # Hidden layer (activation with sigmoid function)
    h = np.dot(x, w1) + b1
    ha = 1 / (1 + np.exp(-h))
    
    # Output layer (activation with softmax function)
    y = np.dot(ha, w2) + b2
    exp_y = np.exp(y)
    ya = exp_y / exp_y.sum()
    
    return x, h, ha, y, ya


def loss_function(ya, t):
    # Cross-entropy loss for a given output ya and target number t
    # This function is not used by the train() function
    # The derivatives of the loss are directly calculated in the backpropagation function
    return -np.log(ya[t])


def backpropagation(x, h, ha, ya, t):
    # Derivatives d_u of the loss with respect to each parameter u
    d_b2 = ya
    d_b2[t] -= 1
    d_w2 = np.outer(ha, d_b2)
    d_b1 = np.dot(w2, d_b2) * ha * (1 - ha)
    d_w1 = np.outer(x, d_b1)
    return d_w1, d_w2, d_b1, d_b2


def train():
    # This function updates the weights and biases to try to minimize the loss

    for k in range(n_iterations):
        
        # Initialization of the derivatives for the batch
        sum_d_w1 = np.zeros(shape_w1)
        sum_d_w2 = np.zeros(shape_w2)
        sum_d_b1 = np.zeros(len_h)
        sum_d_b2 = np.zeros(len_y)
        
        for i in range(batch_size):
            
            # index of the training image and label
            index = k * batch_size + i
            image = train_images[index]
            t = train_labels[index]
            
            x, h, ha, y, ya = forward_propagation(image)
            d_w1, d_w2, d_b1, d_b2 = backpropagation(x, h, ha, ya, t)
            
            sum_d_w1 += d_w1
            sum_d_w2 += d_w2
            sum_d_b1 += d_b1
            sum_d_b2 += d_b2

        # Updating weights and biases
        w1[:] -= learn_r * sum_d_w1
        w2[:] -= learn_r * sum_d_w2
        b1[:] -= learn_r * sum_d_b1
        b2[:] -= learn_r * sum_d_b2
        # The [:] notation is used to modify w1, w2, b1 and b2
        # Without this notation they are considered as undefined local variables
 
        
def test():
    # Takes one random image from the test dataset and checks if the
    # label and result given by the artificial network are the same
    random_number = np.random.randint(0, len(test_labels))
    image = test_images[random_number]
    label = test_labels[random_number]
    x, h, ha, y, ya = forward_propagation(image)
    result = ya.argmax()
    return result == label


def accuracy():
    # Returns the proportion of correctly guessed digits by the network
    acc = 0
    for i in range(n_tests):
        if test():
            acc += 1
    return acc / n_tests
