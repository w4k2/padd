import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

class Protocol:
    def __init__(self, 
                 n_test = 10,
                 metrics=(accuracy_score, balanced_accuracy_score), 
                 verbose=True):

        self.n_test = n_test
        
        if isinstance(metrics, (list, tuple)):
            self.metrics = metrics
        else:
            self.metrics = [metrics]
        self.verbose = verbose
        
        self.train_chunks = []

    def process(self, stream, clfs):
        
        counter = 0
        
        # Verify if pool of classifiers or one
        if isinstance(clfs, ClassifierMixin):
            self.clfs_ = [clfs]
        else:
            self.clfs_ = clfs

        # Assign parameters
        self.stream_ = stream

        # Prepare scores table
        self.scores = np.zeros(
            (len(self.clfs_), ((self.stream_.n_chunks - 1)), len(self.metrics))
        )

        if self.verbose:
            pbar = tqdm(total=stream.n_chunks)
            
        while True:
            X, y = stream.get_chunk()
            if self.verbose:
                pbar.update(1)

            # Test
            if stream.previous_chunk is not None:
                for clfid, clf in enumerate(self.clfs_):
                    y_pred = clf.predict(X)

                    self.scores[clfid, stream.chunk_id - 1] = [
                        metric(y, y_pred) for metric in self.metrics
                    ]

            if counter%self.n_test==0:
                # Train
                [clf.partial_fit(X, y, self.stream_.classes_) for clf in self.clfs_]
                self.train_chunks.append(counter)
                
            counter += 1

            if stream.is_dry():
                break