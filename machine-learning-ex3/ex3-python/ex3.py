## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     lrCostFunction.py (logistic regression cost function)
#     oneVsAll.py
#     predictOneVsAll.py
#     predict.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.


import numpy as np
import matplotlib.pyplot as plt


def load_data(file1, file2):
    return np.loadtxt(file1), np.loadtxt(file2)

def displayData(sel):
    m, n = sel.shape
    example_width  = int(round(np.sqrt(X.shape[1])))
    example_height = int(n/example_width)    
    display_rows   = int(np.floor(np.sqrt(m)))
    display_cols   = int(np.ceil(m / display_rows))
    
    pad = 1
    display_array = -np.ones( (pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad)), dtype=float )
    
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break 
            # Get the max value of the patch
            max_val = max(abs(X[curr_ex, :]))
            curr_pic = np.reshape( X[curr_ex, :], (example_height, example_width) )
            display_array[pad+j*(example_height+pad):pad+j*(example_height+pad)+example_height, 
                          pad+i*(example_width+pad):pad+i*(example_width+pad)+example_width] = curr_pic/max_val
            
            curr_ex = curr_ex + 1
            if curr_ex > m:
                break

    plt.imshow(display_array, cmap='gray')    
    plt.axis('off')
    plt.show()
    plt.savefig('ex3-1.png')

## Setup the parameters you will use for this part of the exercise
input_layer_size = 400  # 20x20 Input Images of Digits
num_labels = 10         # 10 labels, from 1 to 10
                        # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

X, y = load_data('ex3data1_X.txt', 'ex3data1_y.txt') # training data stored in arrays X, y
m = X.shape[0]

## Randomly select 100 data points to display
rand_indices = np.random.permutation(m)

print(rand_indices[:100])
sel = X[rand_indices[:100], :]
displayData(sel)

input("Press Enter to continue...")

### ============ Part 2a: Vectorize Logistic Regression ============
##  In this part of the exercise, you will reuse your logistic regression
##  code from the last exercise. You task here is to make sure that your
##  regularized logistic regression implementation is vectorized. After
##  that, you will implement one-vs-all classification for the handwritten
##  digit dataset.
##

## Test case for lrCostFunction
print('Testing lrCostFunction() with regularization')

theta_t = np.array([-2, -1, 1, 2])
#X_t = [ones(5,1) reshape(1:15,5,3)/10];
#y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3
#[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

#print(['Cost: #f', J])
print('Expected cost: 2.534819')
#fprintf('Gradients:\n');
#fprintf(' #f \n', grad);
#fprintf('Expected gradients:\n');
#fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

#fprintf('Program paused. Press enter to continue.\n');
#pause;
### ============ Part 2b: One-vs-All Training ============
#fprintf('\nTraining One-vs-All Logistic Regression...\n')

#lambda = 0.1;
#[all_theta] = oneVsAll(X, y, num_labels, lambda);

#fprintf('Program paused. Press enter to continue.\n');
#pause;


### ================ Part 3: Predict for One-Vs-All ================

#pred = predictOneVsAll(all_theta, X);

#fprintf('\nTraining Set Accuracy: #f\n', mean(double(pred == y)) * 100);

