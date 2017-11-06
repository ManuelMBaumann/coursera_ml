## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoid.py
#     costFunction.py
#     predict.py
#     costFunctionReg.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def plotData(X,y):
    # Find Indices of Positive and Negative Examples
    pos = np.where(y==1)
    neg = np.where(y==0)
    plt.plot(X[pos, 0], X[pos, 1], 'k+')
    plt.plot(X[neg, 0], X[neg, 1], 'yo')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Admitted', 'Not admitted'])
    plt.show()

def plotDecisionBoundary(theta, X, y):
    # Find Indices of Positive and Negative Examples
    pos = np.where(y==1)
    neg = np.where(y==0)
    plt.plot(X[pos, 0], X[pos, 1], 'k+')
    plt.plot(X[neg, 0], X[neg, 1], 'yo')
    
    if(X.shape[1] <= 3):
        #Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[:,1])-2,  max(X[:,1])+2])
        # Calculate the decision boundary line
        plot_y = (-1./theta[2])*(theta[1]*plot_x + theta[0])
        plt.plot(plot_x, plot_y)
    
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
    plt.show()
    
sigmoid = lambda z: 1.0/(1.0+np.exp(-z))

def costFunction(theta, X, y):
    m = len(y)
    cost = 0.0
    for i in range(m):
        cost = cost + (-y[i]*np.log(sigmoid(np.dot(theta.T,X[i,:]))) - (1-y[i])*np.log(1-sigmoid(np.dot(theta.T,X[i,:]))))
    return 1.0/m*cost

def costFunction_der(theta, X, y):
    m = len(y)
    grad = 0.0*theta
    for i in range(m):
        for j in range(len(theta)):
            grad[j] = grad[j] + ( sigmoid(np.dot(theta.T,X[i,:])) -y[i] )*X[i,j]
    return 1.0/m*grad
    

data = np.loadtxt('ex2data1.txt')
X = data[:, :2]
y = data[:, -1]

### ==================== Part 1: Plotting ====================
##  We start the exercise by first plotting the data to understand the 
##  the problem we are working with.
print(['Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.'])
plotData(X, y)
input("Press Enter to continue...")

### ============ Part 2: Compute Cost and Gradient ============
##  In this part of the exercise, you will implement the cost and gradient
##  for logistic regression. You need to complete the code in 
##  costFunction.py

##  Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

## Add intercept term to x and X_test
X = np.ones((m,n+1), dtype=float)
X[:,1:] = data[:, :2]

## Initialize fitting parameters
initial_theta = np.zeros((n+1,1))

## Compute and display initial cost and gradient
cost = costFunction(initial_theta, X, y)
grad = costFunction_der(initial_theta, X, y)

print(['Cost at initial theta (zeros): ', cost])
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n')
print([' #f \n', grad])
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

## Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost = costFunction(test_theta, X, y)
grad = costFunction_der(test_theta, X, y)

print(['Cost at test theta: ', cost])
print('Expected cost (approx): 0.218')
print('Gradient at test theta: ')
print(' #f ', grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

input("Press Enter to continue...")


### ============= Part 3: Optimizing using fminunc  =============
##  In this exercise, you will use a built-in function (fminunc) to find the
##  optimal parameters theta.

##  Run fminunc to obtain the optimal theta
##  This function will return theta and the cost 

res = minimize(lambda t: costFunction(t, X, y), initial_theta, method='BFGS', jac=lambda t:costFunction_der(t, X, y), 
               options={'disp':True, 'maxiter':400})
cost = res.fun
theta = res.x

## Print theta to screen
print(['Cost at theta found by fminunc: ', cost])
print('Expected cost (approx): 0.203')
print('theta:')
print([' #f ', theta])
print('Expected theta (approx):')
print('-25.161 0.206 0.201')

## Plot Boundary
plotDecisionBoundary(theta, X, y)

input("Press Enter to continue...")

### ============== Part 4: Predict and Accuracies ==============
##  After learning the parameters, you'll like to use it to predict the outcomes
##  on unseen data. In this part, you will use the logistic regression model
##  to predict the probability that a student with score 45 on exam 1 and 
##  score 85 on exam 2 will be admitted.
##
##  Furthermore, you will compute the training and test set accuracies of 
##  our model.
##
##  Your task is to complete the code in predict.m

##  Predict probability for a student with score 45 on exam 1 
##  and score 85 on exam 2 

#prob = sigmoid([1 45 85] * theta);
#fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         #'probability of #f\n'], prob);
#fprintf('Expected value: 0.775 +/- 0.002\n\n');

## Compute accuracy on our training set
#p = predict(theta, X);

#fprintf('Train Accuracy: #f\n', mean(double(p == y)) * 100);
#fprintf('Expected accuracy (approx): 89.0\n');
#fprintf('\n');


