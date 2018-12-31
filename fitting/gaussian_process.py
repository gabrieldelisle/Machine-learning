import numpy as np
from numpy.linalg import inv



def rbf_kernel(x,y) :
	sigma = 2
	return np.exp(-(x-y)**2/2/sigma**2)

class GaussianProcessClassifier(object):
	# gaussian processes are infinite set of gaussian random variables
	# we set their covariance with a kernel matrix, such as two close points have a low covariance 
	# while two distant points have a low correlation
	def __init__(self, x, t, kernel=rbf_kernel):
		self.kernel = kernel
		self.x = x
		self.t = t


	def mean(self, x) :
		K11 = np.array([[self.kernel(xi,xj) for xi in self.x] for xj in self.x])
		K21 = np.array([[self.kernel(xi,xj) for xi in x] for xj in self.x])
		K22 = np.array([[self.kernel(xi,xj) for xi in x] for xj in x])
		return K21.T.dot(inv(K11)).dot(self.t) 

	def deviation(self, x) :
		K11 = np.array([[self.kernel(xi,xj) for xi in self.x] for xj in self.x])
		K21 = np.array([[self.kernel(xi,xj) for xi in x] for xj in self.x])
		K22 = np.array([[self.kernel(xi,xj) for xi in x] for xj in x])
		return np.sqrt(np.diag(K22 - K21.T.dot(inv(K11)).dot(K21)))

	def sample(self, x) :
		K11 = np.array([[k(xi,xj) for xi in self.x] for xj in self.x])
		K12 = np.array([[k(xi,xj) for xi in self.x] for xj in x])
		K22 = np.array([[k(xi,xj) for xi in x] for xj in x])
		invK11 = inv(K11)
		mu = K21.T.dot(invK11).dot(T)
		K = K22 - K21.T.dot(invK11).dot(K21) 
		return np.random.multivariate_normal(mu, K)
		

if __name__ == '__main__':
	import matplotlib.pyplot as plt 

	#generate data
	def f(xi) :
		return (2 + (0.5 * xi -1)**2 ) * np.sin(3*xi)

	x = np.linspace(-4,6,9)
	t = np.array([ f(xi) + np.random.normal(0,3.5) for xi in x])

	indices = np.arange(0,x.shape[0])
	np.random.shuffle(indices)
	x = x[indices]
	t = t[indices]

	#noisy kernel
	sigma_f = 10
	l = 1
	noice = 0
	def noisy_rbf(xi,xj) :
		u = xi-xj
		return sigma_f * np.exp(- u**2 / l**2 ) + (u==0)*noice

	#create gau
	GP = GaussianProcessClassifier(x,t, kernel = noisy_rbf)

	#new data 
	x_test = np.linspace(-5,7,200)

	# plot without noice
	mean = GP.mean(x_test)
	deviation = GP.deviation(x_test)
	plt.plot(x_test,mean,'r')
	plt.plot(x_test,mean-deviation,':r')
	plt.plot(x_test,mean+deviation,':r')
	plt.plot(x,t,'.', 'b')
	plt.ylim(-15,15)
	plt.xlim(-5,7)
	plt.xlabel('x')
	plt.ylabel('t')
	plt.show()

	# plot with noice
	noice=3.5

	mean = GP.mean(x_test)
	deviation = GP.deviation(x_test)
	plt.plot(x_test,mean,'r')
	plt.plot(x_test,mean-deviation,':r')
	plt.plot(x_test,mean+deviation,':r')
	plt.plot(x,t,'.', 'b')
	plt.ylim(-15,15)
	plt.xlim(-5,7)
	plt.xlabel('x')
	plt.ylabel('t')
	plt.show()
