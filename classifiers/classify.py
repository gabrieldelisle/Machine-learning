import random as rd
import matplotlib.pyplot as plt
import numpy as np


class GaussianClassifier:
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
