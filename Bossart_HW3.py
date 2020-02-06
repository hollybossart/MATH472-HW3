import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, multi_dot
## %matplotlib inline


# import the data
data = pd.read_csv('facerecognition.csv', delimiter = ' ')
y = data['match'].to_numpy().reshape((1042,1))
z = data['eyediff'].to_numpy().reshape((1042,1))
ones = np.ones((1042, 1))
z = np.hstack((ones, z))
n = y.size
beta = np.asarray([0.95913, 0]).reshape(2,1)
abeta0s = []
abeta1s = []
bbeta0s = []
bbeta1s = []



def update_pi(beta):
    pi = (1/(1 + np.exp(-z.dot(beta))))
    return pi

def get_b(pi):
    return -np.log(pi)

def newton_beta(beta, opt):
    # this if our first value for everything
    iteration = 0

    W = np.zeros((n,n))
    pi= update_pi(beta)
    np.fill_diagonal(W, pi*(1-pi))

    for i in range(5):
        Hessian = inv(multi_dot((z.T, W, z)))
        print(iteration, beta, Hessian)
        iteration = iteration + 1
        
        if opt == 0:
            abeta0s.append(beta[0])
            abeta1s.append(beta[1])
            
        if opt == 1:
            bbeta0s.append(beta[0])
            bbeta1s.append(beta[1])
            
        beta = beta + Hessian.dot(z.T.dot(y - pi))
        pi = update_pi(beta)
        np.fill_diagonal(W, pi*(1-pi))
    return;

newton_beta(beta, 0)

# number 1b
beta = np.asarray([0, 0]).reshape(2,1)
newton_beta(beta, 1)
# TODO: make the tables look nice

# number 1c - contour plot
num_pts = 200;
b0_vals = np.linspace(-2, 2, num_pts)
b1_vals = np.linspace(-15, 15, num_pts)
B0, B1 = np.meshgrid(b0_vals, b1_vals)


# we need to iterate through the mesh now and calculate the loglikehood

def loglikelihood(y, z, beta0, beta1):
    ones = np.ones((y.size, 1))
    beta = np.array([beta0, beta1]).reshape(2,1)
    pi = (1/(1 + np.exp(-z.dot(beta)))).reshape(y.size, 1)
    b = -np.log(1-pi)
    return multi_dot((y.T, z, beta)) -b.T.dot(ones)

ll = np.zeros((B0.shape[0], B1.shape[0]))    
for i in range(B0.shape[0]):
    for j in range (B1.shape[0]):
        b0 = B0[i,j]
        b1 = B1[i,j]
        ll[i,j] = loglikelihood(y,z,b0,b1)

plt.figure()
plt.contourf(B0, B1, ll, 40, cmap = 'ocean')
plt.plot(abeta0s, abeta1s, '*m-')
plt.title('Contour plot of the log-likelihood function with inputs, B0 and B1')
plt.xlabel('B0 values')
plt.ylabel('B1 values')
plt.annotate('B0_b, B1_b', (-0.2,2))
plt.annotate('B0_a, B1_a', (1,3))
plt.plot(bbeta0s, bbeta1s, '*r-')

# number 2
from scipy.stats import gamma

a = 2 #given from gamma
thetas = np.linspace(gamma.ppf(0, a), gamma.ppf(0.99, a), 100)
pdf_vals = gamma.pdf(thetas, a) #f
plt.figure()
plt.plot(thetas, pdf_vals,'r-', lw=5, alpha=0.6, label='gamma pdf')
plt.title('The Gamma PDF')
maxy = pdf_vals.max()
indexy = pdf_vals.argmax()

num_iterations = 0
max_iterations = 10000
area = 0
step_size = maxy / 10000

while (area <= 0.95) & (num_iterations < max_iterations):
    y = maxy - num_iterations*(step_size) 
    num_iterations += 1
    deltas = np.abs(pdf_vals - y)
    left_array = deltas[0:indexy]
    left_idx = left_array.argmin()
    right_array = deltas[indexy: -1]
    right_idx = right_array.argmin()
    
    left_intercept = thetas[left_idx]
    right_intercept = thetas[right_idx]
    area = gamma.cdf(right_intercept,a) - gamma.cdf(left_intercept,a)

plt.axvline(left_intercept)
plt.annotate('Θ_a: the lower interval bound', (0.2, 0.025))
plt.annotate('Θ_b: the upper \ninterval bound', (5, 0.3))
plt.ylabel('The PDF f(Θ)')
plt.xlabel('Θ values')
plt.axvline(right_intercept)
