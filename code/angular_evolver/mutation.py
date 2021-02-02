from typing import List
import numpy as np

from .graph_genome import GraphGenome

def check_duplicates(genomes: np.array, correct=True):
    """Checks if any genome graph has duplicate points

    Arguments:
        genomes {np.array} -- Array of genomes
    """
    for genome in genomes:
        unique = np.unique(genome.graph.vertices, return_counts=True,
                           return_index=True, axis=0)
        while unique[2].max() > 1 and correct:
            #print("Found duplicate points:", unique)
            # index of the most often occuring points
            uq_index = unique[2].argmax()
            # index in the original array
            index = unique[1][uq_index]
            genome.graph.vertices[index] = np.random.randint(
                genome.bounds[0], genome.bounds[1], 2)
            unique = np.unique(
                genome.graph.vertices, return_counts=True, return_index=True, axis=0)
        return unique[2].max() > 1

def mutate_2d_points(genomes: np.array, genome_mutation_chance=0.03, dna_mutation_chance=0.03):
    """Mutation function for graph instances

    Arguments:
        genomes {np.array} -- Genomes to mutate

    Keyword Arguments:
        genome_mutation_chance {float} -- Overall genome mutation chance (default: {0.03})
        dna_mutation_chance {float} -- Chance a 'dna strain' mutates (default: {0.03})
    """
    genome_size = genomes.shape[0]
    dna_size = int(len(genomes[0])/2)
    # random choice of chosen genome via random indices
    genomes_chance = np.random.random(genome_size)
    mutation_indices = np.where(genomes_chance < genome_mutation_chance)[0]
    for mutation_index in mutation_indices:
        genome = genomes[mutation_index]
        dna_chances = np.random.random(dna_size)
        mutated_dna = np.where(dna_chances < dna_mutation_chance)[0]
        for dna in mutated_dna:
            genome.graph.vertices[dna] = np.random.randint(
                genome.bounds[0], genome.bounds[1], 2
            )

    # check for points that are duplicate points in an instance and change them
    check_duplicates(genomes)

def mutate_vertex_edge_genomes(genomes: List[GraphGenome], genome_mutation_chance=0.03, dna_mutation_chance=0.03):
    # Vertex stuff

    genome_size = len(genomes)
    vert_size = genomes[0].graph.vert_amount
    # random choice of chosen genome via random indices
    genomes_chance = np.random.random(genome_size)
    mutation_indices = np.where(genomes_chance < genome_mutation_chance)[0]
    for mutation_index in mutation_indices:
        genome = genomes[mutation_index]
        dna_chances = np.random.random(vert_size)
        mutated_dna = np.where(dna_chances < dna_mutation_chance)[0]
        for dna in mutated_dna:
            genome.graph.vertices[dna] = np.random.randint(
                genome.bounds[0], genome.bounds[1], 2
            )

    # check for points that are duplicate points in an instance and change them
    check_duplicates(genomes)

    # Edge stuff
    ad_matrix_size = len(genomes[0]) - vert_size
    # random choice of chosen genome via random indices
    genomes_chance = np.random.random(genome_size)
    mutation_indices = np.where(genomes_chance < genome_mutation_chance)[0]
    for mutation_index in mutation_indices:
        genome = genomes[mutation_index]
        first_iteration = False
        while not first_iteration or genome[vert_size:].max() == 0:
            dna_chances = np.random.random(ad_matrix_size)
            mutated_dna = np.where(dna_chances < dna_mutation_chance)[0] + vert_size
            if len(mutated_dna) > 0:
                genome.chromosome[mutated_dna] = 1 - genome.chromosome[mutated_dna]
                genome.update_graph()
            first_iteration = True

    # check_edges:
    for genome in genomes:
        while genome[vert_size:].max() == 0:
            dna_chances = np.random.random(ad_matrix_size)
            mutated_dna = np.where(dna_chances < dna_mutation_chance)[0] + vert_size
            if len(mutated_dna) > 0:
                genome.chromosome[mutated_dna] = 1 - genome.chromosome[mutated_dna]
                genome.update_graph()
