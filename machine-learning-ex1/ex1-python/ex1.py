## Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     warmUpExercise.py
#     plotData.py
#     gradientDescent.py
#     computeCost.py
#     gradientDescentMulti.py
#     computeCostMulti.py
#     featureNormalize.py
#     normalEqn.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#

import numpy as np
import matplotlib.pyplot as plt

def warmUpExercise():
    return np.identity(5)

def plotData(X,y):
    plt.plot(X,y,'rx')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()
    
def computeCost(X, y, theta):
    m = len(y) # number of training examples
    J = 0.0
    for i in range(m):
        J = J + (np.dot(theta.T,X[:,i]) - y[i])**2
    return J/(2.0*m)
    
def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y) # number of training examples
    for it in range(num_iters):
        tmp = 0*theta
        for i in range(m):
            for j in range(len(theta)):
                tmp[j] = tmp[j] + (np.dot(theta.T,X[:,i])-y[i])*X[j,i]
        theta = theta - alpha/m*tmp
        #print(computeCost(X, y, theta))
    return theta

## ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = np.loadtxt('ex1data1.txt')
X = data[:, 0]
y = data[:, 1]
m = len(y)  # number of training examples

# Plot Data
plotData(X, y)

input("Press Enter to continue...")

## =================== Part 3: Cost and Gradient descent ===================

X = np.ones((2,m), dtype=float)
X[1,:] = data[:,0]              # Add a column of ones to x
theta = np.zeros((2,1))         # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')
# compute and display initial cost
J = computeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = ', J)
print('Expected cost value (approx) 32.07\n')

input("Press Enter to continue...")

print('\nRunning Gradient Descent ...\n')
# run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:\n')
print('\n', theta)
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

## Plot the linear fit
plt.plot(X[1,:],y, 'rx')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[1,:], np.dot(X.T,theta))
plt.legend(['Training data', 'Linear regression'], loc='lower right')
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5])*theta
print('For population = 35,000, we predict a profit of \n', predict1*10000)
predict2 = np.aray([1, 7])*theta
print('For population = 70,000, we predict a profit of \n', predict2*10000)

input("Press Enter to continue...")

#%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
#fprintf('Visualizing J(theta_0, theta_1) ...\n')

#% Grid over which we will calculate J
#theta0_vals = linspace(-10, 10, 100);
#theta1_vals = linspace(-1, 4, 100);

#% initialize J_vals to a matrix of 0's
#J_vals = zeros(length(theta0_vals), length(theta1_vals));

#% Fill out J_vals
#for i = 1:length(theta0_vals)
    #for j = 1:length(theta1_vals)
	  #t = [theta0_vals(i); theta1_vals(j)];
	  #J_vals(i,j) = computeCost(X, y, t);
    #end
#end


#% Because of the way meshgrids work in the surf command, we need to
#% transpose J_vals before calling surf, or else the axes will be flipped
#J_vals = J_vals';
#% Surface plot
#figure;
#surf(theta0_vals, theta1_vals, J_vals)
#xlabel('\theta_0'); ylabel('\theta_1');

#% Contour plot
#figure;
#% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
#contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
#xlabel('\theta_0'); ylabel('\theta_1');
#hold on;
#plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
