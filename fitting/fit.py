import numpy as np 
from numpy.linalg import inv

#k nearest neighbours
def knn(x,y,k) :
	def g(u) :
		N = x.size
		i0 = np.argmax((x>=u))
		A = i0-k//2
		B = i0+k-k//2
		if A<0 or u<x[0] :
			A=0
			B=k
		if B>N or u>x[N-1] :
			A=N-k
			B=N

		return np.sum(y[A:B])/k
	return g

#square error
def error(x,y,g) :
	return sum([(g(u)-y[i])**2 for i,u in enumerate(x)])/len(x)

#least square method

#for polynoms, solution is known
def polynomialFit(x,y,d,l=0) :
	#d is the degree
	X = np.array([[u**i for i in range(d+1)] for u in x])
	theta = inv(X.T.dot(X)+l*np.eye(d+1)).dot(X.T).dot(y)
	print("theta :",theta)
	return lambda u: theta.dot(np.array([u**i for i in range(d+1)]))



#for general cases, we use gradiant descent to minimize the error
def parametricFit(test, x, y, params, e=1e-5, a=0.05, Nmax=1e9) :
	#test is a test function which takes as parameters x and params 
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
