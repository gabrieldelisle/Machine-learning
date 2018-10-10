import random as rd
import matplotlib.pyplot as plt
import numpy as np
from classify import GaussianClassifier


#Generating 2 sets of data
NA=100
NB=200
meanA = np.array([1,1])
meanB = np.array([-1,-1])
covA = np.array([[0.5, 0.1],[0.1, 0.3]])
covB = np.array([[0.3, 0.1],[0.1, 0.5]])
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
xgrid=np.linspace(-3, 3, 100) 
ygrid=np.linspace(-3, 3, 100)
grid=np.array([[C.classify(np.array([x, y])) for x in xgrid ] for y in ygrid])
print(grid.shape)
plt.contour( xgrid , ygrid , grid , (0.5), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
plt.axis('equal')
plt.show()