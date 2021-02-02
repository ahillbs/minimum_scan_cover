import numpy as np
from typing import List
from .genome import Genome

def one_point_crossover(father: Genome, mother: Genome) -> List[Genome]:
    """Performs a one point crossover for parents

    Arguments:
        father {Genome} -- Parent one
        mother {Genome} -- Parent two

    Returns:
        List[Genome] -- Two offsprings
    """
    cross_point = np.random.randint(0, len(father))

    offspring1 = list(father[:cross_point]) + list(mother[cross_point:])
    offspring2 = list(mother[:cross_point]) + list(father[cross_point:])
    offsprings = [
        father.create_children(np.array(offspring1), mother=mother),
        father.create_children(np.array(offspring2), mother=mother)
        ]

    return offsprings


def k_point_crossover(father: Genome, mother: Genome, k=None) -> List[Genome]:
    """Performs a k point crossover for parents

    Arguments:
        father {[type]} -- Parent one
        mother {[type]} -- Parent two

    Keyword Arguments:
        k {int/None} -- how many crossover points exists (default: None = inner k)

    Returns:
        List[Genome] -- Two offsprings
    """
    try:
        if k is None:
            k = k_point_crossover.k
    except AttributeError:
        pass # If k_points_crossover does not has k  set, do nothing
    finally:
        if k is None: # If k is not set, standard value of 2 will be set
            k = 2
    cross_points = np.random.randint(0, len(father), k)
    cross_points = np.append(cross_points, 0)
    cross_points = np.append(cross_points, len(father))
    cross_points = np.sort(cross_points)

    parents = [father, mother]
    offspring1 = []
    offspring2 = []

    for i in range(k+1):
        offspring1 = offspring1 + list(parents[i % 2][cross_points[i]:cross_points[i+1]])
        offspring2 = offspring2 + list(parents[i+1 % 2][cross_points[i]:cross_points[i+1]])

    offsprings = [
        father.create_children(np.array(offspring1), mother=mother),
        father.create_children(np.array(offspring2), mother=mother)
        ]

    return offsprings

def uniform_crossover(father: Genome, mother: Genome) -> List[Genome]:
    """Performs a uniform random crossover for parents

    Arguments:
        father {Genome} -- Parent one
        mother {Genome} -- Parent two

    Returns:
        List[Genome] -- Two offsprings
    """
    parents = (father, mother)
    point_size = len(parents[0])

    point_rand = np.random.randint(0, 2, point_size)
    offspring1 = [parents[point_rand[i]][i] for i in range(point_size)]
    offspring2 = [parents[1-point_rand[i]][i] for i in range(point_size)]

    return [
        father.create_children(np.array(offspring1), mother=mother),
        father.create_children(np.array(offspring2), mother=mother)
        ]
