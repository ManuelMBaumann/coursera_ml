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
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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
predict1 = np.dot(np.array([1, 3.5]),theta)
print('For population = 35,000, we predict a profit of \n', predict1*10000)
predict2 = np.dot(np.array([1, 7]),theta)
print('For population = 70,000, we predict a profit of \n', predict2*10000)

input("Press Enter to continue...")

## ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

#Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

TH0_vals, TH1_vals = np.meshgrid(theta0_vals, theta1_vals)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros( (len(theta0_vals), len(theta1_vals)) )

# Fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i,j] = computeCost(X, y, t)

J_vals = J_vals.T
# Surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(TH0_vals, TH1_vals, J_vals, cmap=cm.coolwarm,)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.title('Objective function')
plt.show()

# Contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.title('Location of minimum')
plt.plot(theta[0], theta[1], 'rx', mew=2, ms=8)
plt.show()
