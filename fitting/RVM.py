import numpy as np
from numpy.linalg import inv, det

def rbf_kernel(x,y) :
	sigma = 2
	return np.exp(-(x-y)**2/2/sigma**2)

class RelevanceVectorMachine(object):
	def __init__(self, x, t, eps=1e-2, kernel = rbf_kernel, relevance = 1e-10):
		self.eps = eps
		self.relevance = relevance
		self.kernel = kernel

		self.N = x.shape[0]
		self.x = x
		self.phi = np.array([[1]+[kernel(xi,xj) for xi in x] for xj in x])
		self.support = np.arange(self.N)
		self.t = t
		self.constant_relevant = 1 # 1 if the constant is relevant and 0 else

		#initial hyper-parameters
		self.beta = 1
		self.A = np.eye(self.N+1)
		self.A_new = self.A

		self.train()
		
		

	def E_step(self) :
		# compute posterior
		self.Sigma =  inv(self.beta*self.phi.T.dot(self.phi) + self.A)
		self.mu =  self.beta*self.Sigma.dot(self.phi.T.dot(self.t))

	def M_step(self) :
		# compute new A and beta
		gamma = 1 - np.diag(self.A)*np.diag(self.Sigma)
		self.A = np.diag(gamma/self.mu**2)
		vec = self.t - self.phi.dot(self.mu)
		
		self.beta = (self.N - np.sum(gamma)) / (vec.dot(vec))

	def update_relevant(self) :
		#relevant points to keep
		keep = np.abs(self.mu) > self.relevance * np.mean(np.abs(self.mu))

		self.mu = self.mu[keep]
		self.phi = self.phi[:,keep]
		self.Sigma = self.Sigma[:,keep][keep,:]
		self.A = self.A[:,keep][keep,:]
		self.N = keep.shape[0]-1
		self.A_new = self.A_new[:,keep][keep,:]

		self.support = self.support[keep[self.constant_relevant:]]
		if self.constant_relevant :
			self.constant_relevant = int(keep[0])

	# EM algorithm to maximize the likelihood of A and beta
	def train(self) :
		self.E_step()

		while True : 
			self.A_new = self.A
			self.beta_new = self.beta
			self.M_step()
			self.update_relevant()
			self.E_step()
			print(np.mean((np.abs(np.diag(self.A_new) - np.diag(self.A)))/np.diag(self.A)), self.eps)
			if np.mean((np.abs(np.diag(self.A_new) - np.diag(self.A)))/np.diag(self.A)) < self.eps :
				break
		
	def fit(self, x_new) :
		if self.constant_relevant :
			phi_new = np.array([[1]+[self.kernel(xi,xj) for xi in self.x[self.support]] for xj in x_new])  
		else :
			phi_new = np.array([[self.kernel(xi,xj) for xi in self.x[self.support]] for xj in x_new])

		return phi_new.dot(self.mu)


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	# generate data
	sigma = 0.1
	f = lambda x: np.sin(x)/x
	x = np.linspace(-10,10,100)
	t = np.array([ f(xi) + np.random.normal(0,sigma**2) for xi in x])

	# train model
	RVM = RelevanceVectorMachine(x,t,eps=5e-2, kernel = rbf_kernel)

	# fit new dataset
	x_test = np.linspace(-10,10,1000)
	t_test = f(x_test)
	t_predict = RVM.fit(x_test)

	# compute error
	err = np.sqrt(np.mean((t_predict-t_test)**2))
	print("error:", err)
	print("support:", RVM.support.shape[0], "vectors")

	# plot 
	plt.plot(x_test, t_test, label="real")
	plt.plot(x_test, t_predict, label="predicted")
	plt.plot(x[RVM.support], t[RVM.support], '.', label="support vectors")
	plt.xlabel('x')
	plt.ylabel('t')
	plt.legend()
	plt.show()
