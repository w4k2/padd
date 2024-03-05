import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from strlearn.streams import StreamGenerator
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from vapor.methods import certaintyEns, get_real_drifts
from skmultiflow.meta import LeverageBagging
from Protocol import Protocol
from sklearn.base import clone

class MLPwrap:
    def __init__(self, clf, n_epochs=150):
        self.clf = clone(clf)
        self.n_epochs = n_epochs
        
    def partial_fit(self, X, y, classes):
        [self.clf.partial_fit(X, y, classes) for i in range(self.n_epochs)]
        return self
    
    def predict(self, X):
        return self.clf.predict(X)

n_chunks = 250
stream = StreamGenerator(n_chunks=n_chunks,
                         chunk_size=200,
                         n_drifts=5,
                         n_classes=3,
                         n_features=30,
                         n_informative=30,
                         n_redundant=0,
                         n_repeated=0, 
                         concept_sigmoid_spacing=(5))

protocol = Protocol(metrics=balanced_accuracy_score, n_test=50)

clfs = [
    GaussianNB(),
    MLPwrap(MLPClassifier()),
    LeverageBagging()]

protocol.process(stream, clfs)

print(protocol.scores.shape)
res = protocol.scores

train_chunks = protocol.train_chunks

real_drifts = np.linspace(0,n_chunks,6)[:-1]
real_drifts += (real_drifts[1]/2)

cols = plt.cm.twilight(np.linspace(0.2,0.8,res.shape[0]))

fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.plot(res[0,:,0], label='GNB', color=cols[0])
ax.plot(res[1,:,0], label='MLP', color=cols[1])
ax.plot(res[2,:,0], label='LBC', color=cols[2])
ax.vlines(real_drifts,0,1, color='red', ls=':')
ax.set_xticks(train_chunks)
ax.set_xlabel('train_chunks')

ax.legend()
ax.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')