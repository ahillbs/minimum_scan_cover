import numpy as np

from FeatureSelection.genome import FeatureGenome
from FeatureSelection.mutation import at_least_two_features


def create_feature_genomes(num_features, n=200):
    # Intermediate population with values ranging from 0 to 100
    # Will be used to generate instances with varying amounts of connections
    intermediate_pop = np.array(
        [np.random.randint(0, 100, num_features) for i in range(n)])
    population = np.zeros(intermediate_pop.shape, dtype=np.bool)
    for i in range(intermediate_pop.shape[0]):
        # Restults in 0-1 instances where value is 1 if intermediate instance value is under i
        population[i] = intermediate_pop[i] < (i % 100)
    genomes = np.zeros(n, dtype=object)
    genomes[:] = [FeatureGenome(individual) for individual in population]
    # Repair entities with less than two feature
    at_least_two_features(genomes)
    return genomes


