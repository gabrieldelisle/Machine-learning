import numpy as np 
import matplotlib.pyplot as plt
from random import random 
from numpy.linalg import inv


def polynomialFit(x,y,d,l=0) :
	#d is the degree
	X = np.array([[u**i for i in range(d+1)] for u in x])
	theta = inv(X.T.dot(X)+l*np.eye(d+1)).dot(X.T).dot(y)
	print("theta :",theta)
	return lambda u: theta.dot(np.array([u**i for i in range(d+1)]))

if __name__ == '__main__':
	
	#Generating data with noise
	f = lambda u: 2*np.log(u+2)
	f_noise = lambda u: f(u)+random()-0.5

	#training data
	x = np.array(list(range(50)))
	y = np.array([f_noise(u) for u in x])

	#Fit with a degree 3 polynom
	g = polynomialFit(x,y,3)

	#Plot fitting function and data
	y2 = np.array([g(u) for u in x])
	plt.plot(x,y,'.')
	plt.plot(x,y2)
	plt.show()
