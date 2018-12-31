from bayesFit import linearFit
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, det

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

if __name__ == '__main__':
	
	#generate the data
	sigma = 0.3
	x = np.arange(-2,2.02,0.02)
	t = np.array([-1.5 * xi - 0.5 + np.random.normal(0,sigma**2) for xi in x])

	indices = np.arange(0,x.shape[0])
	np.random.shuffle(indices)
	x = x[indices]
	t = t[indices]

	phi = np.array([[xi,1] for xi in x])

	fit = linearFit(2, sigma)


	#plot parameters distribution
	def plot() :
		def gauss(W,m,S) : 
			u = W-m
			return np.exp(-u.T.dot(inv(S)).dot(u)/2) / np.sqrt(2*np.pi*det(S))

		p = lambda u: gauss(u, fit.mu, fit.Sigma)

		xx = np.arange(-2,2.02,0.02)
		grid = np.array([[p(np.array([i,j])) for i in xx] for j in xx])
		plt.matshow(grid, extent=[-2,2,2,-2])
		
		plt.xlabel('w0')
		plt.ylabel('w1')
		plt.colorbar()
		plt.pause(1)
		plt.close()

	# distribution over w0 and w1
	plot()

	fit.train(phi[:1], t[:1])

	plot()

	fit.train(phi[1:5], t[1:5])

	plot()

	fit.train(phi[5:], t[5:])

	plot()


	x2 = np.array([-2,2])
	phi2 = np.array([[xi,1] for xi in x2])

	#samples
	plt.plot(x,t,'.')
	for i in range(5) :
		y = fit.sample(phi2.T)
		plt.plot(x2,y)
	plt.xlabel('x')
	plt.ylabel('t')
	plt.pause(1)
	plt.close()

	#mean
	print(fit.mu)
	plt.plot(x,t,'.')
	y = fit.mu.dot(phi2.T)
	plt.plot(x2,y)
	plt.pause(1)
	plt.close()