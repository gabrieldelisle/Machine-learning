import random as rd
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

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	#Generating 2 sets of data
	NA=100
	NB=200
	meanA = np.array([1,1])
	meanB = np.array([-1,-1])
	covA = np.array([[0.5, 0.1],[0.1, 0.2]])
	covB = np.array([[0.3, 0.1],[0.1, 0.6]])
	classA = np.random.multivariate_normal(meanA, covA, NA)
	classB = np.random.multivariate_normal(meanB, covB, NB)

	#Ploting the data
	xA,yA = classA.T
	plt.plot(xA, yA,'.')
	xB,yB = classB.T
	plt.plot(xB, yB,'.')

	#Create the training set as corresponding input/target
	inputs = np.concatenate(( classA , classB ))
	targets = np.concatenate((np.zeros(classA.shape[0], dtype=int), np.ones(classB.shape[0], dtype=int)))

	#Shuffle
	permute = list(range(NA+NB)) 
	rd.shuffle(permute)
	inputs = inputs[permute,: ]
	targets = targets [permute ]



	#Train the classifier 
	C = GaussianClassifier(inputs, targets)


	#Plot separating line
	xgrid=np.linspace(-4, 4, 100) 
	ygrid=np.linspace(-3, 3, 100)
	grid=np.array([[C.classify(np.array([x, y])) for x in xgrid ] for y in ygrid])
	print(grid)
	plt.contour( xgrid , ygrid , grid , (0.4,0.5,0.6), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
	plt.axis('equal')
	plt.show()