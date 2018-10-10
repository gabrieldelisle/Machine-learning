import random as rd
import matplotlib.pyplot as plt
import numpy as np
from classify import SupportVectorMachine


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
SVM = SupportVectorMachine(inputs, targets, slack=2)



#Plot separating line
xgrid=np.linspace(-4, 4, 50) 
ygrid=np.linspace(-3, 3, 50)

grid=np.array([[SVM.classify(np.array([x, y])) for x in xgrid ] for y in ygrid])
plt.contour( xgrid , ygrid , grid , (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

plt.axis('equal')

plt.show()
