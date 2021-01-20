import numpy as np

class FeatureGenome:
    def __init__(self, feature_mask: np.array, parents = None):
        self.feature_mask = feature_mask
        self.accuracy = None
        self.parents = parents

    def __getitem__(self, key):
        return self.feature_mask[key]

    def __len__(self):
        return len(self.feature_mask)

    def __repr__(self):
        return "Feature Genome with: " + self.feature_mask.__repr__()

    def create_children(self, feature_mask, mother):
        return FeatureGenome(feature_mask, parents=[self, mother])