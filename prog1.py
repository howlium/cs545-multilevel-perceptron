# ————————————————————————————————————————————————————————————
# Programming Assignment #1: Multi-Level Perceptron
# CS 545, Machine Learning
# Dave Howell
# ————————————————————————————————————————————————————————————

import numpy as np
import random
import math
import operator
import datetime
import sys

# Some global parameters


# The "trial" designation drives some experimental settings
# (i.e. # hidden layers, training data file name)
trial           = "1a"  #  20 hidden nodes, full MNIST training set
trial           = "1b"  #  50 hidden nodes, full MNIST training set
trial           = "1c"  # 100 hidden nodes, full MNIST training set
trial           = "2a"  # 100 hidden nodes, half MNIST training set (named "mnist_train1.csv")
trial           = "2b"  # 100 hidden nodes, quarter training data set (named "mnist_train2.csv")

# 20 hidden layers for trial 1a, 50 for 1b, and 100 for the others (1c, 2a, and 2b)
h_count         = 20 if trial=="1a" else (50 if trial=="1b" else 100)
o_count         = 10
i_count         = 28 * 28
eta             = 0.1
alpha           = 0.9  # momentum
max_epochs      = 50

mnist = MNIST() # Read the input CSV files

# ————————————————————————————————————————————————————————————

MAX_EXP_ARG = math.log(sys.float_info.max)

def sigmoid(x):
    
    # Safety check to avoid float overflow (because exp(709) is expressible in
    # a 64-bit int, but exp(710) is not.
    if -x < MAX_EXP_ARG:
        return 1 / (1 + math.exp(-x))
    else:
        return 0.0

# ————————————————————————————————————————————————————————————

class MNIST:
    '''The MNIST data set'''

    #   Each input data set is an array of vectors x containing 784 floats
    #   i.e. x[1..784] = 28 x 28 grayscale pixel values in [0..1]
    #   labels is an array of correct Class label values in [0..9]
    #   target_outputs is one 1-hot array of outputs for each input
    #   (with 0.9 for the right output and 0.1 for each wrong one)
    # ————————————————————————————————————————————————————————————
    
    def load(self, file_path):
        '''Given a data file path, read the CSV and return the data and
        the target labels'''

        # Load test and training data from CSV files
        data = np.loadtxt(file_path, delimiter=",", skiprows=1) #, max_rows=1000)
        
        # Give that data a random shuffle for good measure
        np.random.shuffle(data)
        
        # In the input files, the first column is the target label for each row.
        num_inputs = data.shape[0]
        target_outputs = np.ndarray((num_inputs, o_count), dtype=float)
        
        # Get the target labels from the first column, delete that column...
        labels = data[:,0]
        data = np.delete(data, 0, 1)
        # ...and normalize to [0..1]
        data /= 255
        
        for n in np.arange(num_inputs):
            for k in np.arange(o_count):
                target_outputs[n,k] = .9 if labels[n] == k else .1
        
        return data, labels, target_outputs
    
    # ————————————————————————————————————————————————————————————

    def __init__(self):

        # Dataset: The directory containing this script also contains a data dir,
        # with the mnist dir in it
        DATA_PATH           = "../data/mnist/"
        TEST_CSV_PATH      = DATA_PATH + "mnist_test.csv"
        
        # Load the right training data for the trial in question
        suffix = "1" if trial=="2a" else ("2" if trial=="2b" else "")
        TRAIN_CSV_PATH     = DATA_PATH + "mnist_train"+suffix+".csv"
        
        self.test_inputs, self.test_labels, self.test_target_outputs = self.load(TEST_CSV_PATH)
        self.train_inputs, self.train_labels, self.train_target_outputs = self.load(TRAIN_CSV_PATH)

# ————————————————————————————————————————————————————————————

class MLP:
    '''
    A pretty basic MLP. Some room for potential improvement:
       1. Train in batches rather than sequentially.
       2. Shuffle data between epochs
       3. Consider the second derivatve of the error with respect to the weights.
       4. Reduce the learning rate over iterations.
    
    In this implementation we're keeping track of the biases separately from
    the other weights. We might just as well have made the biases value 0 and
    shifted the other weights rightward.
    '''
    
    # ————————————————————————————————————————————————————————————
    
    def forward_prop(self, x):
        
        # Run the perceptron for a single input vector. Return the hidden and
        # output activations. First run through the hidden nodes to feed the
        # output layer.
        h = np.ndarray(h_count, dtype = float)
        for j in np.arange(h_count):
            h[j] = sigmoid(np.dot(self.i_weights[j], x) + self.i_bias_weights[j])
        
        # Then run throught the output nodes with those hidden outputs as inputs.
        o = np.ndarray(o_count, dtype = float)
        for k in np.arange(o_count):
            o[k] = sigmoid(np.dot(self.h_weights[k], h) + self.h_bias_weights[k])
        
        return h, o
    
    # ————————————————————————————————————————————————————————————
    
    def best_output_index(self, x):
        
        _, o = self.forward_prop(x)
        return np.argmax(o)
    
    # ————————————————————————————————————————————————————————————
    
    def compute_accuracy(self, inputs, labels):
        
        # Run through the given data set to determine the proportion that are
        # predicted correctly.
        num_correct = 0
        num_inputs = inputs.shape[0]
        for n in np.arange(num_inputs):
            if self.best_output_index(inputs[n]) == labels[n]:
                num_correct += 1
        return num_correct / num_inputs
    
    # ————————————————————————————————————————————————————————————
    
    def confusion_matrix(self, inputs, labels):
        
        matrix = np.full((o_count, o_count), 0)
        num_inputs = inputs.shape[0]
        for n in np.arange(num_inputs):
            matrix[int(labels[n]), self.best_output_index(inputs[n])] += 1
        return matrix
    
    # ————————————————————————————————————————————————————————————
    
    def train(self):

        inputs = mnist.train_inputs
        labels = mnist.train_labels
        
        delta_o = np.full((o_count), 0.)
        delta_h = np.full((h_count), 0.)
        
        # Start with forward propagation
        # Step through each training example to feed the hidden nodes
        num_inputs = inputs.shape[0]
        for n in np.arange(num_inputs):
            
            # Forward-prop: compute the hidden activations (h) and the output
            # activations (o)
            x = inputs[n]
            h, o = self.forward_prop(x)
            t = mnist.train_target_outputs[n]

            best_index = np.argmax(o)
            
            # Do back-prop only if the predicted output for this input is
            # incorrect.
            if best_index != labels[n]:
            
                # Calculate error terms
                delta_o = o * (1 - o) * (t - o)
                delta_h = h * (1 - h) * np.dot(delta_o, self.h_weights)
                
                # Update the hidden-to-output layer weights and bias weights
                self.delta_h_bias = eta * delta_o + momentum * self.delta_h_bias
                self.h_bias_weights += self.delta_h_bias
                for k in np.arange(o_count):
                    self.delta_h[k] = eta * delta_o[k] * h + momentum * self.delta_h[k]
                    self.h_weights[k] += self.delta_h[k]
                
                # Update the input-to-hidden layer weights and bias weights
                self.delta_i_bias = eta * delta_h + momentum * self.delta_i_bias
                self.i_bias_weights += self.delta_i_bias
                for j in np.arange(h_count):
                    self.delta_i[j] = eta * delta_h[j] * x + momentum * self.delta_i[j]
                    self.i_weights[j] += self.delta_i[j]

    # ————————————————————————————————————————————————————————————
    
    def __init__(self):
        
        # Create an array of hidden layer weight vectors (input-to-hidden)
        # Put random values between -.05 and +.05 in each weight
        self.i_weights = np.random.rand(h_count, i_count) * .1 - 0.05
        self.i_bias_weights = np.random.rand(h_count) * .1 - 0.05
        
        # And an array of output layer weight vectors (hidden-to-output)
        self.h_weights = np.random.rand(o_count, h_count) * .1 - 0.05
        self.h_bias_weights = np.random.rand(o_count) * .1 - 0.05
        
        # Initialize the deltas, which we keep track of for use in momentum
        self.delta_i = np.full((h_count, i_count), 0.)
        self.delta_i_bias = np.full((h_count), 0.)
        self.delta_h = np.full((o_count, h_count), 0.)
        self.delta_h_bias = np.full((o_count), 0.)

# ————————————————————————————————————————————————————————————

def do_test():
    # The guts of the homework assignment. Exercises an MLP on the MNIST dataset.
    
    mlp = MLP() # Initialize our multi-level perceptron
    
    prev_acc_1 = prev_acc_2 = 0
    print(f"epoch\ttrain\ttest")
    
    # Start processing
    for epoch in np.arange(max_epochs + 1):
        
        # Compute and display accuracy
        train_accuracy = mlp.compute_accuracy(mnist.train_inputs, mnist.train_labels)
        test_accuracy = mlp.compute_accuracy(mnist.test_inputs, mnist.test_labels)
        
        print(f"{epoch}\t{'{:.3f}'.format(train_accuracy * 100)}%\t{'{:.3f}'.format(test_accuracy * 100)}%")
        
        # Train the Perceptron (one epoch)
        mlp.train()
        
        # If accuracy has stabilized, then stop training.
        if abs(train_accuracy - prev_acc_1) < 0.0001 and abs(prev_acc_2 - prev_acc_1) < 0.0001:
            break
        prev_acc_2 = prev_acc_1
        prev_acc_1 = train_accuracy
    
    # Compute and display confusion matrix for test set

    with np.printoptions(precision=3, suppress=True):
        confusion_matrix = mlp.confusion_matrix(mnist.test_inputs, mnist.test_labels)
        print("Test-set confusion matrix:")
        print(confusion_matrix)

# ————————————————————————————————————————————————————————————

do_test()
