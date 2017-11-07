## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.py
#     costFunction.py
#     predict.py
#     costFunctionReg.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def plotData(X,y):
    # Find Indices of Positive and Negative Examples
    pos = np.where(y==1)
    neg = np.where(y==0)
    plt.plot(X[pos, 0], X[pos, 1], 'k+')
    plt.plot(X[neg, 0], X[neg, 1], 'yo')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(['y = 1', 'y = 0'])
    plt.show()

def mapFeature(X1, X2, degree=6):
    m = int((degree+1)*(degree+2)/2)
    if (X1.ndim == 0):
        n = 1
    else:
        n = X1.shape[0]

    out = np.ones((n,m))
    k = 0
    for i in range(1,degree+1):
        for j in range(i+1):
            out[:,k+1] = np.multiply(X1**(i-j), X2**j)
            k = k + 1
    return out

def plotDecisionBoundary(theta, X, y, lam):
    # Find Indices of Positive and Negative Examples
    pos = np.where(y==1)
    neg = np.where(y==0)
    plt.plot(X[pos, 1], X[pos, 2], 'k+')
    plt.plot(X[neg, 1], X[neg, 2], 'yo')
    
    if(X.shape[1] <= 3):
        #Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[:,1])-2,  max(X[:,1])+2])
        # Calculate the decision boundary line
        plot_y = (-1./theta[2])*(theta[1]*plot_x + theta[0])
        plt.plot(plot_x, plot_y)
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u),len(v)))
        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = np.dot(mapFeature(u[i], v[j]),theta)
         
        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        U, V = np.meshgrid(u,v)
        levels = [0.0]
        plt.contour(U, V, z.T, levels)
    
    #plt.title(['lambda = ', lam])    
    #plt.xlabel('Microchip Test 1')
    #plt.ylabel('Microchip Test 2')
    #plt.legend(['y = 1', 'y = 0', 'Decision boundary'])
    plt.show()
    
sigmoid = lambda z: 1.0/(1.0+np.exp(-z))

def costFunction(theta, X, y, lam):
    m = len(y)
    n = len(theta)
    
    cost = 0.0
    for i in range(m):
        cost = cost + (-y[i]*np.log(sigmoid(np.dot(theta.T,X[i,:]))) - (1-y[i])*np.log(1-sigmoid(np.dot(theta.T,X[i,:]))))
    reg = 0.0
    for j in range(1,n):
        reg = reg + theta[j]**2
    return 1.0/m*cost + lam/(2*m)*reg

def costFunction_der(theta, X, y, lam):
    m = len(y)
    n = len(theta)
    grad = 0.0*theta
    for j in range(n):
        for i in range(m):
            grad[j] = grad[j] + ( sigmoid(np.dot(theta.T,X[i,:])) -y[i] )*X[i,j]
        if (j > 0):
            grad[j] = grad[j]+lam*theta[j]
    return 1.0/m*grad

def costFunctionReg(theta, X, y, lam):
    cost = costFunction(theta, X, y, lam)
    grad = costFunction_der(theta, X, y, lam)
    return cost, grad

## Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

data = np.loadtxt('ex2data2.txt')
X = data[:, :2]
y = data[:, -1]

plotData(X, y)


### =========== Part 1: Regularized Logistic Regression ============
##  In this part, you are given a dataset with data points that are not
##  linearly separable. However, you would still like to use logistic
##  regression to classify the data points.
##
##  To do so, you introduce more features to use -- in particular, you add
##  polynomial features to our data matrix (similar to polynomial
##  regression).
##

## Add Polynomial Features

## Note that mapFeature also adds a column of ones for us, so the intercept
## term is handled
X = mapFeature(X[:,0], X[:,1])

## Initialize fitting parameters
initial_theta = np.zeros((X.shape[1],1), dtype=float)

## Set regularization parameter lambda to 1
lam = 1

## Compute and display initial cost and gradient for regularized logistic
## regression
cost, grad = costFunctionReg(initial_theta, X, y, lam)

print(['Cost at initial theta (zeros): ', cost])
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros) - first five values only:')
print([' #f ', grad[:5]])
print('Expected gradients (approx) - first five values only:')
print(' 0.0085, 0.0188, 0.0001, 0.0503, 0.0115,')

input("Press Enter to continue...")

## Compute and display cost and gradient
## with all-ones theta and lambda = 10
test_theta = np.ones((X.shape[1],1))

cost, grad = costFunctionReg(test_theta, X, y, 10.0)

print(['Cost at test theta (with lambda = 10): ', cost])
print('Expected cost (approx): 3.16')
print('Gradient at test theta - first five values only:')
print([' #f ', grad[:5]])
print('Expected gradients (approx) - first five values only:')
print(' 0.3460, 0.1614, 0.1948, 0.2269, 0.0922')

input("Press Enter to continue...")

### ============= Part 2: Regularization and Accuracies =============
##  Optional Exercise:
##  In this part, you will get to try different values of lambda and
##  see how regularization affects the decision coundart
##
##  Try the following values of lambda (0, 1, 10, 100).
##
##  How does the decision boundary change when you vary lambda? How does
##  the training set accuracy vary?
##

## Initialize fitting parameters
initial_theta = np.zeros((X.shape[1],1))

## Set regularization parameter lambda to 1 (you should vary this)
lam = 0.0

res = minimize(lambda t: costFunction(t, X, y, lam), initial_theta, method='BFGS', jac=lambda t:costFunction_der(t, X, y, lam), 
               options={'disp':True, 'maxiter':400})

theta = res.x

## Plot Boundary
plotDecisionBoundary(theta, X, y, lam)

## Compute accuracy on our training set
#p = predict(theta, X);

#fprintf('Train Accuracy: #f\n', mean(double(p == y)) * 100);
#fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');

