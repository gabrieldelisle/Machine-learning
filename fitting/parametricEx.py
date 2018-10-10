import numpy as np 
import matplotlib.pyplot as plt
from random import random 
from fit import parametricFit, error

#Generating data with noise
f = lambda u: 2*np.log(u+2)
f_noise = lambda u: f(u)+random()-0.5

#training data
x = np.array(list(range(50)))
y = np.array([f_noise(u) for u in x])

#test data
x_test = np.array(list(range(50)))
y_test = np.array([f(u) for u in x])


#test function
def test(x,params) :
	a,b,c = params
	return a*np.log(x+b)+c


#Fit with a degree 3 polynom
g = parametricFit(test, x, y, [1,1,1] )



#Print error
err = error(x_test,y_test,g)
print("error :", err)

#Plot fitting function and data
y2 = np.array([g(u) for u in x])
plt.plot(x,y,'+')
plt.plot(x,y2)
plt.show()