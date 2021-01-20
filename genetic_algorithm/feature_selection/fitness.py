import numpy as np
import tqdm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

class FeatureFitness:
    def __init__(self, features: np.array, labels: np.array, svm):
        self.features = features
        self.labels = labels
        self.svm = svm

    def calc_features(self, features):
        result = list()
        pbar = tqdm.tqdm(features, desc="Collecting features")
        for feature in pbar:
            cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
            scores = cross_val_score(self.svm, feature, self.labels, cv=cv, n_jobs=-1, verbose=False)            
            result.append((scores.mean() -  (scores.std() * 2)))
        return np.array(result)

    def __call__(self, genomes: np.array):
        try:
            results = np.array([genome.accuracy for genome in genomes])
            recalculate = (results == None).any() # pylint is wrong :( it's not the same
        except AttributeError:
            recalculate = True
        if recalculate:
            features = [self.features[:, genome.feature_mask] for genome in genomes]
            results = self.calc_features(features)
            for result, genome in zip(results, genomes):
                genome.accuracy = result
        return results

class FeatureTransformFitness(FeatureFitness):
    def __init__(self, features: np.array, labels: np.array, transformer, svm):
        super().__init__(features, labels, svm)
        self.transformer = transformer

    def __call__(self, genomes: np.array):
        try:
            results = np.array([genome.accuracy for genome in genomes])
            recalculate = (results == None).any() # pylint is wrong :( it's not the same
        except AttributeError:
            recalculate = True
        if recalculate:
            # This is for testing only
            #for i,genome in enumerate(genomes):
            #    feat = self.features[:, genome.feature_mask]
            #    feat_transform = self.transformer.fit_transform(feat)
            features = np.array(
                [
                self.transformer.fit_transform(self.features[:, genome.feature_mask])
                for genome in genomes
                ]
            )
            results = self.calc_features(features)
            for result, genome in zip(results, genomes):
                genome.accuracy = result
        return results