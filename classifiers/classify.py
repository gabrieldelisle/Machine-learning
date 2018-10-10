import random as rd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np


class GaussianClassifier:
	#model data as followin a gaussian law, and use the most likely class to classify
	def __init__(self, X, labels):
		#check if data rae same size
		assert(X.shape[0]==labels.shape[0])

		Npts,Ndims = X.shape
		classes = np.unique(labels)
		Nclasses = np.size(classes)

		#compute the a priori probability of being in each
		self.proba = np.zeros(classes.shape)
		for l in labels :
			self.proba[l]+=1
		self.proba/=Npts

		#data is modeled as following a gaussian law
		#compute mean and covariance matrices
		self.mu = np.zeros((Nclasses,Ndims))
		self.sigma = np.zeros((Nclasses,Ndims,Ndims))

		for c in classes :
			Xc = X[labels==c,:]
			Nc = Xc.shape[0]
			self.mu[c,:] = np.sum(Xc, axis=0)/Nc
			v = (Xc - self.mu[c,:])
			self.sigma[c,:,:] = v.T.dot(v)/Nc


	def classify(self, x):
		Nclasses,Ndims = np.shape(self.mu)
		#compute the logarithmic likelihood probability of x being in class c 
		logProb = np.log(self.proba) - 0.5 * np.array([
			(x-self.mu[c,:]).dot(np.linalg.inv(self.sigma[c,:,:])).dot((x-self.mu[c,:]).T)
			+ np.log(np.linalg.det(self.sigma[c,:,:]))
			for c in range(Nclasses) ])

		#return the most likely class
		h = np.argmax(logProb)
		return h

#=============

#Kernels for the support vector machines
def linearKernel(x,y) :
	return x.dot(y)

def quadraticKernel(x,y) :
	return (x.dot(y)+1)**10

def rbfKernel(x,y) :
	sigma = 1
	d = x-y
	return np.exp(-d.dot(d)/2/sigma**2)


class SupportVectorMachine:
	# SVM maximize margins around a separating line which can be a straight line or more complex.
	# to do so, we transform the problem in another called dual problem, in which we minimize an objective.
	# We can also allow some points to be inside margins adding slack quantity
	
	def __init__(self, X, labels, kernel=linearKernel, slack=None) :
		#check if data rae same size
		assert(X.shape[0]==labels.shape[0])

		self.X = X
		self.kernel = kernel
		self.labels = labels
		N = X.shape[0]

		#pre-computation of a later used matrix
		P = np.array([[kernel(x,y)*ti*tj for ti, x in zip(labels, X)] for tj,y in zip(labels, X)])

		#define functions used in dual problem
		def objective(alpha) :
			return 1/2*alpha.dot(P).dot(alpha) - np.sum(alpha)

		def zerofun(alpha) :
			return alpha.dot(labels)

		start = np.ones(N)
		B = [(0,slack) for b in range(N)]
		XC = {'type':'eq', 'fun':zerofun}

		#use scipy minimize to solve the dual problem
		res = minimize( objective, start, bounds=B, constraints=XC )
		self.alpha = res['x']

		nonZeroAlpha = self.alpha[abs(self.alpha)>1e-5]
		nonZeroTargets = labels[abs(self.alpha)>1e-5]
		nonZeroInputs = X[abs(self.alpha)>1e-5]

		#compute treshold value b
		s = nonZeroInputs[0]
		ts = nonZeroTargets[0]
		Ks = np.array([kernel(s,x) for x in X])
		self.b = np.sum(self.alpha*labels*Ks) - ts 
		self.a = self.alpha*labels

	def classify(self, s) :
		Ks = np.array([self.kernel(s,x) for x in self.X])
		return np.sum(self.alpha*self.labels*Ks) - self.b
