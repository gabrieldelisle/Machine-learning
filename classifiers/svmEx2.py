import random as rd
import matplotlib.pyplot as plt
import numpy as np
from classify import SupportVectorMachine, linearKernel, rbfKernel, quadraticKernel


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
