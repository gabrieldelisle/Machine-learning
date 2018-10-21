import matplotlib.pyplot as plt
import numpy as np
from fit import knn

f = lambda x: np.sin(x)

N=100
a=0
b=2

x = np.arange(a,b,(b-a)/N)
y = f(x)+0.2*(np.random.rand(N)-0.5)

plt.plot(x,y,".")

g = knn(x,y,9)
x2 = np.arange(a,b,(b-a)/N/4)


plt.plot(x2,[f(u) for u in x2])
plt.plot(x2,[g(u) for u in x2])
plt.show()


