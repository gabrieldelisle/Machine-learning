import numpy as np
from scipy.optimize import minimize



def splineKernel(xi,xj) :
	return (1 + xi*xj + xi*xj * min(xi,xj) - 1/2 * (xi + xj) * min(xi,xj)**2 + 1/3 * min(xi,xj)**3)/5


def rbfKernel(x,y) :
	sigma = 1
	return np.exp(-(x-y)**2/2/sigma**2)


class SupportVectorRegression:
	# SVR minimize margins around the fitted curve, where all points should be
	# to do so, we transform the problem in another called dual problem, in which we minimize an objective.
	# We can also allow some points to be outside margins adding slack quantity C
	
	def __init__(self, X, T, kernel=rbfKernel, eps=1, C=None) :
		assert(X.shape[0]==T.shape[0])

		self.X = X
		self.kernel = kernel
		self.T = T
		N = X.shape[0]

		#pre-computation of kernel matrix
		K = np.array([[kernel(x,y) for x in X] for y in X])

		#define functions used in dual problem
		def objective(alpha) :
			N = alpha.shape[0]//2
			alphap = alpha[:N]
			alphan = alpha[N:]
			aa = (alphap-alphan)
			return 1/2*aa.dot(K).dot(aa) - aa.dot(T) + eps * np.sum(alphap+alphan)

		def zerofun(alpha) :
			N = alpha.shape[0]//2
			alphap = alpha[:N]
			alphan = alpha[N:]
			return np.sum(alphap-alphan)

		start = np.ones(2*N)
		B = [(0,C) for b in range(2*N)]
		XC = {'type':'eq', 'fun':zerofun}

		#use scipy minimize to solve the dual problem
		res = minimize( objective, start, bounds=B, constraints=XC )
		
		alpha = res['x']
		N = alpha.shape[0]//2
		alphap = alpha[:N]
		alphan = alpha[N:]
		self.alpha = (alphap-alphan)
		print("alpha: ", self.alpha)

		# use only support vectors
		self.T = T[abs(self.alpha)>1e-10]
		self.X = X[abs(self.alpha)>1e-10]
		self.alpha = self.alpha[abs(self.alpha)>1e-10]
		

		#compute treshold value b
		s = self.X[0]
		ts = self.T[0]
		Ks = np.array([kernel(s,x) for x in self.X])
		self.b = self.alpha.dot(Ks) - ts 

	def fit(self, s) :
		#fit a new set of data s
		Ks = np.array([self.kernel(s,x) for x in self.X])
		return self.alpha.dot(Ks) - self.b

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	# generate data
	f = lambda x: np.sin(x)/x
	x = np.linspace(-10,10,100)
	sigma = 0.1
	t = np.array([f(xi) + np.random.normal(0,sigma**2) for xi in x])

	# create support vector
	SVR = SupportVectorRegression(x,t, eps = 0.001, kernel=rbfKernel, C=1 )


	# fit different data
	x_test = np.linspace(-10,10,1000)
	T = np.array([SVR.fit(u) for u in x_test])

	# compute error
	t_test = f(x_test)
	err = np.sqrt(np.mean((T-t_test)**2))
	print("error:", err)
	print("support:", SVR.X.shape[0], "vectors")

	# plot 
	plt.plot(x_test,t_test, label='real')
	plt.plot(x_test,T, label='predicted')
	plt.plot(SVR.X, SVR.T, '.', label='support vectors')
	plt.xlabel('x')
	plt.ylabel('t')
	plt.legend()
	plt.show()


