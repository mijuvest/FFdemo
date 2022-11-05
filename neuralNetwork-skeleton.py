# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 18:41:27 2021

@author: mijuvest
"""
import numpy as np
import matplotlib.pyplot as plot
import scipy.special
import imageio
#%matplotlib inline
#plot.imshow(a, interpolation="nearest")



class neuralNetwork:
    
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5),(self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.lr = learningrate
        
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass
    
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        
        # error is the (target - actual)
        output_errors = targets - final_outputs
        
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes 
        hidden_errors = np.dot(self.who.T, output_errors)
        
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 -
        final_outputs)), np.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 -
        hidden_outputs)), np.transpose(inputs))
        
        
        pass
    
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
        
        pass
    
    
    # save neural network weights 
    def save(self):
        np.save('C:/devel/python-deeplearning-2021/saved_wih.npy', self.wih)
        np.save('C:/devel/python-deeplearning-2021/saved_who.npy', self.who)
        pass
    
    # load neural network weights 
    def load(self):
        self.wih = np.load('C:/devel/python-deeplearning-2021/saved_wih.npy')
        self.who = np.load('C:/devel/python-deeplearning-2021/saved_who.npy')
        pass
    
    
    
    
input_nodes = 784
hidden_nodes = 150
output_nodes = 10

verkko = neuralNetwork(input_nodes, hidden_nodes, output_nodes, 0.1)
#verkko.query([1.0, 0.5, -1.5])

training_datafile = open("C:/devel/python-deeplearning-2021/mnist_train.csv")
training_data_list = training_datafile.readlines()
training_datafile.close()

epochs = 7
for e in range(epochs):
        # # go through all records in the training data set
        for record in training_data_list:
        #     # split the record by the ',' commas
              all_values = record.split(',')
        
        #     # scale and shift the inputs
              inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        #     # create the target output values (all 0.01, except the desired label which is 0.99) 
              targets = np.zeros(output_nodes) + 0.01
        #     # all_values[0] is the target label for this record
              targets[int(all_values[0])] = 0.99
            
              verkko.train(inputs, targets)
              pass
        pass


# load the mnist test data CSV file into a list
test_data_file = open("C:/devel/python-deeplearning-2021/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

all_test_values = test_data_list[1].split(',')
print(all_test_values[1])

print(verkko.query((np.asfarray(all_test_values[1:]) / 255.0 * 0.99) + 0.01))

image_array = np.asfarray(all_test_values[1:]).reshape((28,28))
plot.imshow(image_array, cmap='Greys', interpolation='None')



scorecard = []
# go through all the records in the test data set
for record in test_data_list:
# split the record by the ',' commas
    all_values = record.split(',')
# correct answer is first value
    correct_label = int(all_values[0])
    #print(correct_label, "correct label")
# scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
# query the network
    outputs = verkko.query(inputs)
# the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    #print(label, "network's answer")
# append correct or incorrect to list
    if (label == correct_label):
# network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
# network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
pass

# calculate the performance score, the fraction of correct answers
scorecard_array = np.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)


img_array = imageio.imread("C:/devel/python-deeplearning-2021/testikutonen.png", as_gray=True)
img_data = 255.0 - img_array.reshape(784)
img_data = (img_data / 255.0 * 0.99) + 0.01 

print(verkko.query(np.asfarray(img_data)))


image_array = np.asfarray(img_data[0:]).reshape((28,28))
plot.imshow(image_array, cmap='Greys', interpolation='None')
























