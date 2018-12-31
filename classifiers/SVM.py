import random as rd
import numpy as np
from scipy.optimize import minimize

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
	
	def __init__(self, X, labels, kernel=rbfKernel, slack=None) :
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


if __name__ == '__main__':
	
	import matplotlib.pyplot as plt

	#Generating 2 sets of data
	NA=100
	NB=100
	meanA = np.array([1,1])
	meanB = np.array([-1,-1])
	covA = np.array([[0.2, 0.1],[0.1, 0.2]])
	covB = np.array([[0.2, 0.1],[0.1, 0.2]])
	classA = np.random.multivariate_normal(meanA, covA, NA)
	classB = np.random.multivariate_normal(meanB, covB, NB)

	#Ploting the data
	xA,yA = classA.T
	plt.plot(xA, yA,'.')
	xB,yB = classB.T
	plt.plot(xB, yB,'.')

	#Create the training set as corresponding input/target
	inputs = np.concatenate(( classA , classB ))
	targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

	#Shuffle
	permute = list(range(NA+NB)) 
	rd.shuffle(permute)
	inputs = inputs[permute,: ]
	targets = targets [permute ]



	#Train the classifier
	SVM = SupportVectorMachine(inputs, targets, kernel=linearKernel, slack=2)



	#Plot separating line
	xgrid=np.linspace(-4, 4, 50) 
	ygrid=np.linspace(-3, 3, 50)

	grid=np.array([[SVM.classify(np.array([x, y])) for x in xgrid ] for y in ygrid])
	plt.contour( xgrid , ygrid , grid , (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

	plt.axis('equal')

	plt.show()

	#==========
	# Example2


	#Generating 2 sets of data

	classA = np.concatenate ((
		np.random.randn(20, 2) * 0.4 + [-1, 0], 
		np.random.randn(20, 2) * 0.4 + [1, 0]
	)) 
	classB = np.concatenate ((
		np.random.randn(20, 2) * 0.4 + [0, 1], 
		np.random.randn(20, 2) * 0.4 + [0, -1]
	)) 

	#Ploting the data
	xA,yA = classA.T
	plt.plot(xA, yA,'.')
	xB,yB = classB.T
	plt.plot(xB, yB,'.')

	#Create the training set as corresponding input/target
	inputs = np.concatenate(( classA , classB ))
	targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

	#Shuffle
	permute = list(range(targets.shape[0])) 
	rd.shuffle(permute)
	inputs = inputs[permute,: ]
	targets = targets [permute ]



	#Train the classifier 
	SVM = SupportVectorMachine(inputs, targets, kernel=rbfKernel, slack=10)



	#Plot separating line
	xgrid=np.linspace(-4, 4, 100) 
	ygrid=np.linspace(-3, 3, 100)
	grid=np.array([[SVM.classify(np.array([x, y])) for x in xgrid ] for y in ygrid])
	plt.contour( xgrid , ygrid , grid , (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

	plt.axis('equal')

	plt.show()
