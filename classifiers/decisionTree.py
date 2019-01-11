import numpy as np
from numpy.random import shuffle, normal, random
import matplotlib.pyplot as plt


# entropy is used to measure the information we have on the data set
def entropy(S) :
	N = len(S)
	classes = {}
	for u in S :
		if u in classes :
			classes[u]+=1
		else :
			classes[u]=1
	return -sum(p/N*np.log(p/N) for p in classes.values())


def gain(data, target, feature, treshold) :
	S1 = target[data[:,feature] <= treshold]
	S2 = target[data[:,feature] > treshold]
	return entropy(target) - ( len(S1)*entropy(S1) + len(S2)*entropy(S2) ) /len(target)


# choose the tuple feature treshold which minimize entropy
def choose_feature(data, target) :
	best_E = entropy(target)
	best_feature = None
	best_treshold = None
	for feature in range(data.shape[1]) :
		for treshold in data[:-1,feature] :
			S1 = target[data[:,feature] <= treshold]
			S2 = target[data[:,feature] > treshold]
			# entropy of separated subsets
			E = ( len(S1)*entropy(S1) + len(S2)*entropy(S2) ) /len(target)
			if E<best_E :
				best_E = E
				best_feature = feature
				best_treshold = treshold
	return best_feature, best_treshold


def major(S) :
	classes = {}
	for u in S :
		if u in classes :
			classes[u]+=1
		else :
			classes[u]=1
	return max(classes.keys(), key = lambda x: classes[x])



class DecisionTree() :
	# decision trees are efficient to separate data with many features

	def __init__(self, max_depth = float('inf')) :
		self.feature = {}
		self.treshold = {}
		self.major = {"": None}
		self.leaves = {}
		self.root = "" #the different choices will be stored in a string
		self.max_depth = max_depth

	def copy(self) :
		tree = DecisionTree(max_depth = self.max_depth)
		tree.feature = self.feature.copy()
		tree.treshold = self.treshold.copy()
		tree.major = self.major.copy()
		tree.leaves = self.leaves.copy()
		return tree

	def train(self, data, target) :
		pile = [(data, target, 0, self.root)]
		while pile :
			d, t, depth, way = pile.pop()
			self.major[way] = major(t)

			if depth < self.max_depth and entropy(t)>0 :
				feature, treshold = choose_feature(d, t)
				# use best feature and treshold to separate the data at each node of the tree
				if feature :
					self.feature[way] = feature
					self.treshold[way] = treshold
					self.leaves[way] = True
					self.leaves[way[:-1]] = False
					
					#separate the data
					choice1 = d[:,feature] <= treshold
					choice2 = d[:,feature] > treshold

					pile.append((d[choice1], t[choice1], depth+1, way+'1'))
					pile.append((d[choice2], t[choice2], depth+1, way+'2'))


	def classify(self, point) :
		way = self.root
		# follow every test until the end
		while way in self.feature :
			if point[self.feature[way]] <= self.treshold[way] :
				way+='1'
			else :
				way+='2'
		# return the most probable class knowing the tests
		return self.major[way]


	def error(self, data, target) :
		predict = np.array([self.classify(u) for u in data])
		return 1-np.mean(predict == target)


	def allPruned(self) :
		#list of all possible pruning of this tree
		l = []
		for way in self.leaves :
			if self.leaves[way] :
				tree = self.copy()
				del tree.feature[way]
				del tree.treshold[way]
				del tree.major[way+'1']
				del tree.major[way+'2']
				tree.leaves[way] = False
				tree.leaves[way[:-1]] = True
				l.append(tree)
		return l

	def prune(self, data, target) :
		# some tests at the leaves of the tree might be not representative
		# the goal of pruning is to remove these tests
		while 1 :
			old_err = self.error(data, target)
			l = [(t.error(data, target), t) for t in self.allPruned()]
			if not l :
				break
			new_err, new_t = max(l, key= lambda x:x[0])
			if new_err < old_err :
				# use the best pruned tree if it reduces error
				self = new_t
			else :
				break
			

if __name__ == '__main__':

	# generate dataset
	N=400
	K=10
	dataA = normal(1,1,(N,K))
	dataB = normal(-1,1,(N,K))
	targetA = np.zeros(N)
	targetB = np.ones(N)
	for i in range(N) :
		if random()<0.1 :
			targetA[i], targetB[i] = targetB[i], targetA[i]
	data = np.concatenate((dataA, dataB))
	target = np.concatenate((targetA, targetB))

	# shuffle data set
	indexes = np.arange(2*N)
	shuffle(indexes)
	data = data[indexes]
	target = target[indexes]


	# split data set in training/validation/test datasets
	N = 2*N
	N1 = N//2-N//6
	N2 = N//2
	training_data = data[:N1]
	training_target = target[:N1]
	validation_data = data[N1:N2]
	validation_target = target[N1:N2]
	test_data = data[N2:]
	test_target = target[N2:]

	# use decision tree on dataset
	tree = DecisionTree()
	tree.train(training_data, training_target)
	print("error:",tree.error(test_data, test_target))

	# Pruning is a good way to reduce overfitting 
	# Here, the data set is very simple and it has no effect
	tree.prune(validation_data, validation_target)
	print("after pruning error:",tree.error(test_data, test_target))


