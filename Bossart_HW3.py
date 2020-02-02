import pandas as pd
import numpy as np
from numpy.linalg import inv, multi_dot


# import the data
data = pd.read_csv('facerecognition.csv', delimiter = ' ')
y = data['match'].to_numpy().reshape((1042,1))
z = data['eyediff'].to_numpy().reshape((1042,1))
ones = np.ones((1042, 1))
z = np.hstack((ones, z))
n = y.size
beta = np.asarray([0.95913, 0]).reshape(2,1)

def update_pi(beta):
    return (1/(1 + np.exp(-z.dot(beta))))

def newton_beta(beta):
    # this if our first value for everything\
    iteration = 0

    W = np.zeros((n,n))
    pi= update_pi(beta)
    np.fill_diagonal(W, pi*(1-pi))

    for i in range(5):

        Hessian = inv(multi_dot((z.T, W, z)))
        print(iteration, beta, Hessian)
        iteration = iteration + 1
        beta = beta + Hessian.dot(z.T.dot(y - pi))
        pi = update_pi(beta)
        np.fill_diagonal(W, pi*(1-pi))
  
newton_beta(beta)

# number 1b
beta = np.asarray([0, 0]).reshape(2,1)
newton_beta(beta)