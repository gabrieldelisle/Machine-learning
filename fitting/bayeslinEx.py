from bayesFit import linearFit
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, det


#generate the data
sigma = 0.3
x = np.arange(-2,2.02,0.02)
t = np.array([-1.5 * xi - 0.5 + np.random.normal(0,sigma**2) for xi in x])

indices = np.arange(0,x.shape[0])
np.random.shuffle(indices)
x = x[indices]
t = t[indices]

phi = np.array([[xi,1] for xi in x])

fit = linearFit(2, sigma)


#plot parameters distribution
def plot() :
	def gauss(W,m,S) : 
		u = W-m
		return np.exp(-u.T.dot(inv(S)).dot(u)/2) / np.sqrt(2*np.pi*det(S))

	p = lambda u: gauss(u, fit.mu, fit.Sigma)

	xx = np.arange(-2,2.02,0.02)
	grid = np.array([[p(np.array([i,j])) for i in xx] for j in xx])
	plt.matshow(grid, extent=[-2,2,2,-2])
	
	plt.xlabel('w0')
	plt.ylabel('w1')
	plt.colorbar()
	plt.pause(1)
	plt.close()


plot()

fit.train(phi[:1], t[:1])

plot()

fit.train(phi[1:5], t[1:5])

plot()

fit.train(phi[5:], t[5:])

plot()


x2 = np.array([-2,2])
phi2 = np.array([[xi,1] for xi in x2])

plt.plot(x,t,'.')
for i in range(5) :
	y = fit.sample(phi2.T)
	plt.plot(x2,y)
plt.xlabel('x')
plt.ylabel('t')
plt.pause(1)
plt.close()

print(fit.mu)
plt.plot(x,t,'.')
y = fit.mu.dot(phi2.T)
plt.plot(x2,y)
plt.pause(1)
plt.close()