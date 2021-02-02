import numpy as np
import math
from genetic_algorithm import GeneticAlgorithm


def _get_selection(fitness_indices, size, amount_saved, p):
    returned_size = (math.ceil((size-amount_saved)/2), 2)
    selection = np.random.choice(fitness_indices[:size-amount_saved], returned_size, p=p)
    # Check if some individual of the selection have the same father and mother (sic!)
    mask = selection[:, 0] == selection[:, 1]
    while mask.any():
        selection[mask] = np.random.choice(
            fitness_indices[:size-amount_saved],
            (mask.sum(), 2), p=p
        )
        mask = selection[:, 0] == selection[:, 1]
    return selection

def uniform_wheel_selection(evo_alg: GeneticAlgorithm, genomes, fitness, elitism_factor=0.1):
    size = fitness.shape[0]
    fitness_indices = np.argsort(fitness)
    # elitism
    amount_saved = int(size * elitism_factor) if elitism_factor < 1 else elitism_factor

    p = np.array([fitness[i] for i in range(size)], dtype=np.float)
    # if there are negative values add it minimum to all values (setting everything above(equal) to zero)
    p += np.min([0, p.min()])

    p /= p.sum()
    # size can happen to not match for uneven sizes
    
    selection = _get_selection(fitness_indices, size, 0, p)
    return (fitness_indices[size-amount_saved:], selection)


def linear_rank_selection(evo_alg: GeneticAlgorithm, genomes, fitness, elitism_factor=0.1, pressureFactor=2):
    fitness_indices = np.argsort(fitness)
    size = fitness.shape[0]

    # Carry over the elite instances without changing them
    amount_saved = int(size * elitism_factor) if elitism_factor < 1 else elitism_factor
    p = np.array(
        [(1+i) * pressureFactor for i in range(size - amount_saved)], dtype=np.float)
    # Due to small numbers, the sum tend to not be near 1
    # Therefore, divive by value by sum of values, to get closer to 1
    p /= p.sum()
    # size can happen to not match for uneven sizes
    selection = _get_selection(fitness_indices, size, amount_saved, p)

    return (fitness_indices[size-amount_saved:], selection)
