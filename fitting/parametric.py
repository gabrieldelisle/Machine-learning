import numpy as np 
import matplotlib.pyplot as plt
from random import random 

def error(x,y,g) :
	return sum([(g(u)-y[i])**2 for i,u in enumerate(x)])/len(x)


#for general cases, we use gradiant descent to minimize the error
def parametricFit(test, x, y, params, e=1e-5, a=0.05, Nmax=1e9) :
	#test is a fitting function which takes as parameters x and params 
	#params are the starting parameters
	#e is the maximal difference between 2 steps
	#a is the step of gradiant descent
	#Nmax is the maximum number of iteration

	def grad(f,params,h=1e-10) :
		return [(f(params[:i]+[p+h]+params[i+1:])-f(params[:i]+[p-h]+params[i+1:]))/2/h for i,p in enumerate(params)]

	#square erro
	def energy(params) :
		return error(x,y,lambda u: test(u,params))

	i=0
	eOld = 0
	eNew = energy(params)
	while abs(eNew-eOld)>e and i<Nmax :
		i+=1
		g = grad(energy, params)
		params = [p-a*g[i] for  i,p in enumerate(params)]
		eOld = eNew
		eNew = energy(params)

	print("params:",params)
	return lambda x: test(x,params)


if __name__ == '__main__':
	#Generating data with noise
	f = lambda u: 2*np.log(u+2)
	f_noise = lambda u: f(u)+random()-0.5

	#training data
	x = np.array(list(range(50)))
	y = np.array([f_noise(u) for u in x])


	#test function
	def test(x,params) :
		a,b,c = params
		return a*np.log(x+b)+c

	#Fit with a test function
	g = parametricFit(test, x, y, [1,1,1] )

	#Plot fitting function and data
	y2 = np.array([g(u) for u in x])
	plt.plot(x,y,'.')
	plt.plot(x,y2)
	plt.show()