import numpy as np
from typing import List
from genetic_algorithm.genome import Genome

def multi_uniform_crossover(father: Genome, mother: Genome) -> List[Genome]:
    """Performs a uniform random crossover for parents

    Arguments:
        father {Genome} -- Parent one
        mother {Genome} -- Parent two

    Returns:
        List[Genome] -- Two offsprings
    """
    parents = (father, mother)
    genome_size = len(parents[0])
    offspring1 = []
    offspring2 = []
    for i in range(genome_size):
        point_size = len(parents[0][i])

        point_rand = np.random.randint(0, 2, point_size)
        offspring1.append([parents[point_rand[i]][i] for i in range(point_size)])
        offspring2.append([parents[1-point_rand[i]][i] for i in range(point_size)])

    return [
        father.create_children(np.array(offspring1), mother=mother),
        father.create_children(np.array(offspring2), mother=mother)
        ]

def multi_k_point_crossover(father: Genome, mother: Genome, k=None) -> List[Genome]:
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
            k = multi_k_point_crossover.k
    except AttributeError:
        pass # If k_points_crossover does not has k  set, do nothing
    finally:
        if k is None: # If k is not set, standard value of 2 will be set
            k = 2
    parents = [father, mother]
    offspring1_genomes = []
    offspring2_genomes = []
    for useless_dna in father:
        cross_points = np.random.randint(0, len(useless_dna), k)
        cross_points = np.append(cross_points, 0)
        cross_points = np.append(cross_points, len(useless_dna))
        cross_points = np.sort(cross_points)

        offspring1_dna = []
        offspring2_dna = []

        for i in range(k+1):
            offspring1_dna = offspring1_dna + list(parents[i % 2][cross_points[i]:cross_points[i+1]])
            offspring2_dna = offspring2_dna + list(parents[i+1 % 2][cross_points[i]:cross_points[i+1]])
        offspring1_genomes.append(offspring1_dna)
        offspring2_genomes.append(offspring2_dna)

    offsprings = [
        father.create_children(np.array(offspring1_genomes), mother=mother),
        father.create_children(np.array(offspring2_genomes), mother=mother)
        ]

    return offsprings

class OrderUniformCrossover():
    def __init__(self, father_taken_chance=0.5):
        self.father_taken_chance = father_taken_chance

    def __call__(self, father: Genome, mother: Genome):
        order_numbers_father = self._get_order_numbers(father)
        order_numbers_mother = self._get_order_numbers(mother)
        
        point_size = len(father)
        point_rand = np.random.random(point_size)
            
        parents = [order_numbers_father, order_numbers_mother]
        parent_order = [0 if i < self.father_taken_chance else 1 for i in point_rand]

        offspring1 = [parents[parent][i] for i, parent in enumerate(parent_order)]
        offspring2 = [parents[1-parent][i] for i, parent in enumerate(parent_order)]

        offsprings = [
            father.create_children(np.array(offspring1), mother=mother),
            father.create_children(np.array(offspring2), mother=mother)
        ]
        return offsprings

    def _get_order_numbers(self, parent):
        if hasattr(parent, "order_numbers"):
            return parent.order_numbers
        else:
            return np.random.randint(0, size=len(parent))
        