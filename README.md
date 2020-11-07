# Neural network built from scratch


Presentation
--------

The **[simple_nn](simple_nn.py)** Python code is an example of a simple artificial neural network 
written from scratch using only two packages from [pypi.org](https://pypi.org/):
- [numpy](https://numpy.org/) to manipulate arrays and compute matrix operations efficiently.
- [mnist](https://pypi.org/project/mnist/) to download and parse the dataset into training and testing sets with labels that can be used as numpy arrays.

The network is trained to classify handwritten digits using the MNIST database (cf. http://yann.lecun.com/exdb/mnist/).

![](MnistExamples.png)


The **[simple_nn_test.ipynb](simple_nn_test.ipynb)** contains a demo of how to use the **[simple_nn](simple_nn.py)** file to train and test the network's performance. There are also explanations of how the network works and is trained using gradient descent. In this file we use the [matplotlib](https://matplotlib.org) package to illustrate the dataset inputs and the results of the network.


Structure of the neural network
--------

This is a forward propagating network with 3 layers :
- the input layer has 784 nodes (28 x 28 input images are flattened)
- the hidden layer has 16 nodes
- the output layer has 10 nodes

Activation functions used:
- Sigmoid for the hidden layer
- Softmax for the output layer

The loss is calculated with cross entropy.


Usage guide of the [simple_nn](simple_nn.py) functions
--------

1.  Run the ***train()*** function to train the neural network.
    This function executes 200 iterations.
    For each iteration a batch of 32 images is processed.
    At the end of each iteration the weights and biases are updated.
    
2.  Run the ***accuracy()*** function to see the network's performance.

3.  Feel free to change the network's parameters and try to improve accuracy.
