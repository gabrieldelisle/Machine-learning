import numpy as np 
from numpy.linalg import inv


class linearFit :
	# linear fitting: we assume a gaussian distribution for the points, with a priori the same mean and variance
	# N i the dimension of the input
	# the noice over the data is a hyper-parameter, part of the asumptions.
	def __init__(self, N, noice, mu0=0, sigma0=1) :
		self.N = N
		self.beta = 1/noice**2
		self.mu = mu0*np.ones(N)
		self.Sigma = sigma0*np.eye(N)

	def train(self, phi, t) :
		S = inv(inv(self.Sigma) + self.beta*phi.T.dot(phi))
		m = S.dot(inv(self.Sigma).dot(self.mu) + self.beta*phi.T.dot(t))
		self.Sigma = S
		self.mu = m

	def sample(self, phi) :
		W = np.random.multivariate_normal(self.mu, self.Sigma)
		return W.dot(phi)
