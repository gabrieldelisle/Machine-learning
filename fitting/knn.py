import matplotlib.pyplot as plt
import numpy as np

#k nearest neighbours
def knn(x,y,k) :
	def g(u) :
		N = x.size
		i0 = np.argmax((x>=u))
		A = i0-k//2
		B = i0+k-k//2
		if A<0 or u<x[0] :
			A=0
			B=k
		if B>N or u>x[N-1] :
			A=N-k
			B=N

		return np.sum(y[A:B])/k
	return g

if __name__ == '__main__':
	f = lambda x: np.sin(x)

	N=1000
	a=0
	b=2

	x = np.arange(a,b,(b-a)/N)
	y = f(x)+0.2*(np.random.rand(N)-0.5)


	g = knn(x,y,51)
	x2 = np.arange(a,b,(b-a)/N/4)

	plt.plot(x,y,".")
	plt.plot(x2,[f(u) for u in x2])
	plt.plot(x2,[g(u) for u in x2])
	plt.show()


