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
    example_width  = int(round(np.sqrt(sel.shape[1])))
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
            max_val = max(abs(sel[curr_ex, :]))
            curr_pic = np.reshape( sel[curr_ex, :], (example_height, example_width) )
            display_array[pad+j*(example_height+pad):pad+j*(example_height+pad)+example_height, 
                          pad+i*(example_width+pad):pad+i*(example_width+pad)+example_width] = curr_pic.T/max_val
            
            curr_ex = curr_ex + 1
            if curr_ex > m:
                break

    plt.imshow(display_array, cmap='gray')    
    plt.axis('off')
    plt.show()

sigmoid = lambda z: 1.0/(1.0+np.exp(-z))

def lrCostFunction(theta, X, y, lam):
    m = len(y)
    n = len(theta)
    
    cost = 0.0
    tmp = 0.0
    print(np.dot(X,theta))
    for i in range(m):
        cost = cost -y[i]*np.log(sigmoid(np.dot(theta.T,X[i,:]))) 
        cost = cost -(1-y[i])*np.log(1-sigmoid(np.dot(theta.T,X[i,:])))
        tmp = tmp + np.dot(theta.T,X[i,:])
    print(tmp)
    return 1.0/m*cost

    
def lrCostFunction_der(theta, X, y, lam):
    return 1


## Setup the parameters you will use for this part of the exercise
input_layer_size = 400  # 20x20 Input Images of Digits
num_labels = 10         # 10 labels, from 1 to 10
                        # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data...')

X, y = load_data('ex3data1_X.txt', 'ex3data1_y.txt') # training data stored in arrays X, y
m = X.shape[0]

## Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
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
X_t = np.ones((5,4))
X_t[:,1:4] = np.reshape(np.linspace(1,15,15),(3,5)).T/10.0
y_t = np.array([1,0,1,0,1])

lambda_t = 3
J    = lrCostFunction(theta_t, X_t, y_t, lambda_t)
grad = lrCostFunction_der(theta_t, X_t, y_t, lambda_t)

print(['Cost: #f', J])
print('Expected cost: 2.534819')
print('Gradients:')
print([' #f ', grad])
print('Expected gradients:')
print(' 0.146561 -0.548558 0.724722 1.398003')

input("Press Enter to continue...")

### ============ Part 2b: One-vs-All Training ============
print('Training One-vs-All Logistic Regression...')

lam = 0.1
#[all_theta] = oneVsAll(X, y, num_labels, lambda);

input("Press Enter to continue...")


### ================ Part 3: Predict for One-Vs-All ================

#pred = predictOneVsAll(all_theta, X);

#fprintf('\nTraining Set Accuracy: #f\n', mean(double(pred == y)) * 100);

