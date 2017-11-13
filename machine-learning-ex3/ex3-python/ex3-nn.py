## Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     lrCostFunction (logistic regression cost function)
#     oneVsAll
#     predictOneVsAll
#     predict
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

import numpy as np
import matplotlib.pyplot as plt


def load_data(file1, file2):
    return np.loadtxt(file1), np.loadtxt(file2)

## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                        # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

X, y = load_data('ex3data1_X.txt', 'ex3data1_y.txt') 
m = X.shape[0]

## Randomly select 100 data points to display
#sel = randperm(size(X, 1));
#sel = sel(1:100);

#displayData(X(sel, :));

input("Press Enter to continue...")

### ================ Part 2: Loading Pameters ================
## In this part of the exercise, we load some pre-initialized 
## neural network parameters.

print('Loading Saved Neural Network Parameters ...\n')

## Load the weights into variables Theta1 and Theta2
Theta1, Theta2 = load_data('ex3data1_Th1.txt', 'ex3data1_Th2.txt') 


### ================= Part 3: Implement Predict =================
##  After training the neural network, we would like to use it to predict
##  the labels. You will now implement the "predict" function to use the
##  neural network to predict the labels of the training set. This lets
##  you compute the training set accuracy.

#pred = predict(Theta1, Theta2, X);

#fprintf('\nTraining Set Accuracy: #f\n', mean(double(pred == y)) * 100);

#fprintf('Program paused. Press enter to continue.\n');
#pause;

##  To give you an idea of the network's output, you can also run
##  through the examples one at the a time to see what it is predicting.

##  Randomly permute examples
#rp = randperm(m);

#for i = 1:m
    ## Display 
    #fprintf('\nDisplaying Example Image\n');
    #displayData(X(rp(i), :));

    #pred = predict(Theta1, Theta2, X(rp(i),:));
    #fprintf('\nNeural Network Prediction: #d (digit #d)\n', pred, mod(pred, 10));
    
    ## Pause with quit option
    #s = input('Paused - press enter to continue, q to exit:','s');
    #if s == 'q'
      #break
    #end
#end

