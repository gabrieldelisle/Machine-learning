import numpy as np

# boosting is a method that use several simple classifiers
# at each step, we give a weight to each point
# this weight is higher if the point has been often classified right 
# then, we train the weighted points with a new base classifier, and so on
# we also give a weight to each classifier depending of their accuracy

# finally, to classify a new data, we classify it with each classifier
# the result is the class of highest sum of weighted votes 

class BoostClassifier(object):
    def __init__(self, base_classifier, T=10, params = []):
        self.base_classifier = base_classifier
        self.T = T
        self.params = params

    def train(self, X, labels):
        Npts,Ndims = np.shape(X)
        self.classes = np.unique(labels)

        self.classifiers = [] # list of base classifiers
        self.alphas = np.array([]) # vote weights of the classifiers

        # initial weights
        w = np.ones(Npts)/float(Npts)

        for _ in range(0, self.T):
            self.classifiers.append(self.base_classifier(*self.params))
            self.classifiers[-1].train(X, labels, w)
            vote = self.classifiers[-1].classify(X) != labels
            eps = max(w.dot(vote),1e-100)
            self.alphas = np.append(self.alphas, 1/2 * (np.log(1-eps) - np.log(eps)))
            w = np.exp(self.alphas[-1] * ((vote) * 2 - 1))
            w /= np.sum(w)


    def classify(self, X):
        #classify X with each classifiers
        votes = np.array([c.classify(X) for c in self.classifiers])
        #compute the weighted means of votes
        weighted_vote = np.array([self.alphas.dot(votes == i) for i in self.classes])
        #return the most likely classes
        return np.argmax(weighted_vote, axis=0)


if __name__ == '__main__':
    # In this example, we use boosting with decision trees
    from decisionTree import DecisionTree
    from numpy.random import shuffle, normal, random


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


    # split data set in training/test subsets
    training_data = data[:N]
    training_target = target[:N]
    test_data = data[N:]
    test_target = target[N:]

    # train and use boosted classifier
    model = BoostClassifier(DecisionTree, T=10)
    model.train(training_data, training_target)
    print("model trained")

    predicted = model.classify(test_data)
    err = 1-np.mean(predicted == test_target)
    print("error: ", err)

