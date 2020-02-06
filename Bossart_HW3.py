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
cauchy_vals = np.asarray([1.77,-0.23,2.76,3.8,3.47,56.75,-1.34,4.24,-2.44,3.29,3.71,-2.4,4.53,-0.07,-1.05,-13.87,-2.53,-1.75,0.27,43.21])
n = cauchy_vals.size

def loglike(theta):
    foo = -np.log([1+(cauchy_vals[i]-theta)**2 for i in range(n)]).sum(axis=0)
    return  -n*np.log(np.pi) + foo

def llprime(theta):
    temp = []
    #iterate through all observations
    for i in range(n):
        temp.append(2*(cauchy_vals[i]-theta)/((cauchy_vals[i] - theta)**2 + 1))   
    return sum(temp)

def llprime2(theta):
    temp = []
    for i in range(n):
        temp.append((-2*(theta**2-2*theta*cauchy_vals[i]+cauchy_vals[i]**2-1))/((theta**2-2*theta*cauchy_vals[i]+cauchy_vals[i]**2+1)**2))
    return sum(temp)

def step(theta):
    return llprime(theta) / llprime2(theta)

def newton_method(theta, tol, max_iterations, print_option):
    theta_vals = [theta]
    num_iterations = 0
    
    while abs(llprime(theta)) > tol and num_iterations < max_iterations:
        old_theta = theta_vals[num_iterations]
        new_theta = old_theta + step(old_theta)
        num_iterations += 1
        theta_vals.append(new_theta)
        theta = new_theta
        
        
    # at this point we have broken out of the while loop    
    if num_iterations == max_iterations:
        print("Exceeded maximum number of iterations.")
        return
        
    # this is where we hope to be if newtons went well    
    if print_option == 1:
        print('\n')
        print("Starting value: " + str(theta_vals[0]))
        sol = theta_vals[-1]
        print("Number of iterations: " + str(num_iterations) + " \nTolerance: " + str(tol))
        print("Final solution: " + str(sol))
        return
    
    if print_option == 0:
        return theta_vals[-1]

# plotting the log likelihood function
thetas = np.linspace(-12,12, 1000)
plt.figure()
plt.plot(thetas, loglike(thetas))

plt.title('Graph of the log-likelihood function')
plt.xlabel('Theta values')
plt.ylabel('Log-likehood function')

# find the MLE for theta using NR method
starting_pts = np.asarray([-11, -1, 0, 1.5, 4, 4.7, 7, 8, 38])
tol = 1e-6
max_iterations = 2000
printopt = 1
for val in starting_pts:
    newton_method(val, tol, max_iterations, printopt)



## find MLE using bisection method with new starting pts

def bisection(x, a, b, tol):
    num_iterations = 0
    max_iterations = 100
    a0 = a
    b0 = b
    
    # creating lists for these values for graphing purposes later
    a_list = [a]
    b_list = [b]
    x_list = [x]
    
    # check the condition holds to continue the loop
    while ((abs(llprime(x)) > tol) and (num_iterations < max_iterations)):
        
        # check that condition holds for intermediate value theorem
        if llprime(a)*llprime(b) > 0:
            print("Bisection method fails.")
            return
        
        # if this does not fail, then we can test other conditions given in book
        if llprime(a)*llprime(x) <= 0:
            a = a
            b = x
        elif llprime(a)*llprime(x) > 0:
            a = x
            b = b
        else:
            print("Something went wrong!")
            return
        
        # updating new x values in new interval
        x = (a+b)/2;
        num_iterations+=1;
        
        # storing our new a, b, and x values at the end of a list for graphing later
        a_list.append(a)
        b_list.append(b)
        x_list.append(x)
    
    # while loop has broken at this point in the code
    # we are either within the appropriate tolerance, or reached max iterations
    if num_iterations == max_iterations:
        print("Exceeded maximum number of iterations.")
        return
        
    # this is where we hope to be if bisection went well    
    else:
        print("Number of iterations: " + str(num_iterations) + " with tolerance: " + str(tol))
        print("Final solution: " + str(x))

 


bisection(0, -1, 1, tol)


# fixed point iteration
alphas = [1,.64,.25] # scaling factor 
iterations = 0
for alpha in alphas:
    theta = -1 
    while llprime(theta)>tol:
        theta = theta + alpha*llprime(theta) 
        k+=1    
    print('Fixed-point estimate for alpha = %.2f is: ' %alpha,theta)
    print('Iterations: ',iterations)
    print('Tolerance: ', tol)
    print()


# number 3
from scipy.stats import gamma

a = 2 #given from gamma
thetas = np.linspace(gamma.ppf(0, a), gamma.ppf(0.99, a), 100)
pdf_vals = gamma.pdf(thetas, a) #f
plt.figure()
plt.plot(thetas, pdf_vals,'r-', lw=2, alpha=0.6, label='gamma pdf', color = 'k')
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
    right_intercept = thetas[right_idx + left_array.size]
    area = gamma.cdf(right_intercept,a) - gamma.cdf(left_intercept,a)

plt.axvline(left_intercept)
plt.annotate('Θ_a: the lower interval bound', (0.2, 0.025))
plt.annotate('Θ_b: the upper \ninterval bound', (5, 0.3))
plt.ylabel('The PDF f(Θ)')
plt.xlabel('Θ values')
plt.axvline(right_intercept)

x = np.arange(left_intercept, right_intercept, 0.01)
plt.fill_between(x, gamma.pdf(x, a), color = 'c')
plt.annotate('95% posterior \ndensity interval', (0.8, 0.17))


