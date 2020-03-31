# Neural network built from scratch


Presentation
--------

This Python code is an example of a simple artificial neural network
written from scratch using only two modules:
- [numpy](https://numpy.org/)
- [mnist](https://pypi.org/project/mnist/)

The MNIST database of handwritten digits is used to train the network: 
http://yann.lecun.com/exdb/mnist/


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


Usage guide
--------

1.  Run the ***train()*** function to train the neural network.
    This function executes 200 iterations.
    For each iteration a batch of 32 images is processed.
    At the end of each iteration the weights and biases are updated.
    
2.  Run the ***accuracy()*** function to see the network's performance.

3.  Feel free to change the network's parameters and try to improve accuracy.
