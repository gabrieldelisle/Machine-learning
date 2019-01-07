import numpy as np
import random as rd

# GENERATOR FUNCTIONS _____________________________________________________________________________________________________________________________

def define_HMMs(K,R,M):
	# Class probabilities - one class is much more probable than the rest
	pi = np.zeros((K))
	class1 = rd.randint(0,K-1)
	pi[class1] = 0.5
	for k in range(K):
		if pi[k]==0.0:
			pi[k] = 0.5/(K-1)

	# Start probabilities - UNIFORM
	init = 1/R*np.ones((R))

	# Transition probabilities - from each row, there are only two possible next states, with varying probabilities
	A = np.zeros((K, R, R))
	for k in range(K):
		for r in range(R):
			rand = rd.randint(10,20)
			row1 = rd.randint(0,R-1)
			row2 = rd.randint(0,R-1)
			while(row2 == row1):
				row2 = rd.randint(0,R-1)

			A[k,r,row1] = rand/20
			A[k,r,row2] = (20-rand)/20

	# Emission probabilities - different noise for different classes, but same noise for all rows within that class
	E = np.zeros((K, R, R))
	for k in range(K):
		rand = rd.randint(10,20)
		for r in range(R):
			E[k,r,r] = rand/20
			E[k,r,(r+1)%K] = (20-rand)/40
			E[k,r,(r-1)%K] = (20-rand)/40

	return pi, init, A, E


def generate_states(k,R,M):
	init = start_prob
	X = np.zeros((M), dtype=int)

	rand = rd.random()
	sum_steps = 0.0
	for r in range(R):
		if rand>=sum_steps and rand<sum_steps+init[r]:
			X[0] = r
			break
		sum_steps += init[r]

	for m in range(1,M):
		A = transition_prob[k,X[m-1],:]
		rand = rd.random()
		sum_steps = 0.0
		for r in range(R):
			if rand>=sum_steps and rand<sum_steps+A[r]:
				X[m] = r;
				break
			sum_steps += A[r]
	
	return X

		
def generate_observations(k,R,M,X):
	Z =  np.zeros((M), dtype=int)
	for m in range(M):
		E = emission_prob[k,X[m],:]
		rand = rd.random()
		sum_steps = 0.0
		for r in range(R):
			if rand>=sum_steps and rand<sum_steps+E[r]:
				Z[m] = r
				break
			sum_steps += E[r]
	
	return Z


def generate_data(N,K,R,M):
	classes = np.zeros((N), dtype=int)
	observations = np.zeros((N,M), dtype=int)

	for n in range(N):
		rand = rd.random()
		sum_steps = 0.0
		for k in range(K):
			if rand>=sum_steps and rand<sum_steps+class_prob[k]:
				k_n = k;
				break
			sum_steps += class_prob[k]

		classes[n] = k_n
		observations[n,:] = generate_observations(k_n, R, M, generate_states(k_n, R, M))

	return classes, observations



# SOLUTION FUNCTIONS _______________________________________________________________________________________________________________________

R = 10 #rows
M = 10 #columns
N = 10 #tries
K = 10 #classes

def most_likely(pi) :
	index = list(range(K))
	l = []
	for n in range(N) :
		l.append(max(index, key = lambda i: pi[n,i]))
	return l


class Model(): 
	# Mixture of K HMM of length M with R states
	def __init__(self, start, transition, emission):
		self.start = start
		self.A = transition
		self.E = emission

		self.gamma = np.zeros((N, K, M, R))
		self.pi = np.ones((N,K))/K
		self.phi = np.zeros((N,K))

	def E_step(self, x) :
		# forward-backward algotrithm to update gamma

		alpha = np.zeros((N, K, M, R))
		beta = np.zeros((N, K, M, R))

		# scaling factor to prevent values from being lower than precision limit
		scaling = np.zeros((N, K, M))

		for n in range(N) :
			for c in range(K) :

				# alpha initialisation
				alpha[n,c,0,:] = self.start * self.E[c,:,x[n,0]]
				scaling[n,c,0] = np.sum(alpha[n,c,0,:])
				alpha[n,c,0,:] /= scaling[n,c,0]


				# forward computation of alpha
				for m in range(1,M) :
					alpha[n,c,m,:] = self.E[c,:,x[n,m]] * alpha[n,c,m-1,:].dot(self.A[c,:,:])
					scaling[n,c,m] = np.sum(alpha[n,c,m,:])
					if scaling[n,c,m]==0 :
						scaling[n,c,m:]=1.
						break
					alpha[n,c,m,:] /= scaling[n,c,m]
				# beta initalisation
				beta[n,c,M-1,:] = 1.

				# backward computation of beta
				for m in range(M-2,-1,-1) :
					beta[n,c,m,:] = ( self.A[c,:,:] ).dot( beta[n,c,m+1,:] * self.E[c,:,x[n,m+1]] ) / scaling[n,c,m+1]

				# gamma computation
				for m in range(M) :
					self.gamma[n,c,m,:] = alpha[n,c,m,:]*beta[n,c,m,:]

		# phi computation
		self.phi = self.pi * np.sum(np.sum(self.gamma, axis=2), axis=2)

		self.loglikelihood = np.sum(np.log(scaling))


	def M_step(self, x) :
		self.pi = self.phi
		for n in range(N) :
			self.pi[n,:]/=np.sum(self.pi[n,:])

	def EM(self, x, eps) :
		old = 0.
		new = 1.
		while abs(old-new)/abs(new) > eps :
			self.E_step(x)
			self.M_step(x)
			old = new
			new = self.loglikelihood
		return most_likely(self.pi)



# _____________________________________________________________________________________________________________________________________


Nite = 10

right = 0

for _ in range(Nite) :

	class_prob, start_prob, transition_prob, emission_prob = define_HMMs(K, R, M)
	targets, data = generate_data(N, K, R, M)
	print("\nObserved sequences\n",data)
	print("\nTrue classes\n", targets)

	model = Model(start_prob, transition_prob, emission_prob)
	classes = model.EM(data, 1e-3)
	print("most likely:", classes)
	print("with proba:", [model.pi[i,c] for i,c in enumerate(classes)])

	right += sum(targets == classes)

print("success rate:", right/(N)/Nite)


		
