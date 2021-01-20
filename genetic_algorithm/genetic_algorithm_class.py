from functools import reduce
from typing import List
import operator
import numpy as np
from .genome import Genome


class GeneticAlgorithm:
    def __init__(self, **kwargs):
        """Creates a genetic algorithm.
        Keyword Arguments:
            selection {callable} -- Selection function used
            fitness {callable} -- Fitness function used
            mutation {callable} -- Mutation function used
            genomes {Genomes} -- First generation of genomes
            termCon {callable} -- Termination condition
            callback {callable} -- Callback at the end of an iteration
            elitism {float} -- Elitism percentage
            pressure {float} -- EXPERIMENTAL: Pressure factor for fitness values (default: {1})
            mutationChance {float} -- Chance a genome is selected for mutation (default: {0.03})
            mutationChanceGene {float} -- Chance a gene is mutated (default: {0.03})
        """
        if(kwargs is not None):
            self.selection = kwargs.pop("selection", None)
            self.fitness = kwargs.pop("fitness", None)
            self.mutation = kwargs.pop("mutation", None)
            self.crossover = kwargs.pop("crossover", None)
            self.genomes = kwargs.pop("genomes", None)
            self.termCon = kwargs.pop("termCon", None)
            self.callback = kwargs.pop("callback", None)
            self.elitism = kwargs.pop("elitism", 0.1)
            self.pressureFactor = kwargs.pop("pressure", 1)
            self.genomeMutationChance = kwargs.pop("mutationChance", 0.03)
            self.geneMutationChance = kwargs.pop("mutationChanceGene", 0.03)
        self.evolving = False
        self.generation = 0
        self.fitness_val = None

    def evolve(self, generation=0) -> np.ndarray:
        """Evolve until the termination condition is met
        
        Keyword Arguments:
            generation {int} -- Starting generation (default: {0})
        
        Returns:
            numpy.ndarray[Genome] -- Last iteration genomes
        """
        self.evolving = True
        self.generation = generation
        local_genomes = self.genomes
        while not self.termCon(self):
            local_genomes = self.evoStep(local_genomes)
        self.evolving = False
        self.fitness_val = None
        return local_genomes

    def evoStep(self, local_genomes=None) -> List[Genome]:
        """Single evolution step
        
        Keyword Arguments:
            local_genomes {List[Genome]} -- Genomes that will be evolved. If None self.genomes will be used (default: {None})
        
        Returns:
            numpy.ndarray[Genome] -- Genomes evolved in this evolution step
        """     
        # evolutionary steps
        if local_genomes is not None:
            self.genomes = local_genomes
        arr = np.empty(len(self.genomes), dtype=object)
        arr[:] = self.genomes[:]
        self.fitness_val = self.fitness(self.genomes)


        elite, crossover_candidates = self.selection(
            self, self.genomes, self.fitness_val, self.elitism
            )

        children = [
            self.crossover(self.genomes[parent1], self.genomes[parent2])
            for parent1, parent2 in crossover_candidates
            ]
        children = reduce(operator.concat, children)

        arr_children = np.empty(len(children), dtype=object)
        arr_children[:] = children[:]
        elites = np.empty(len(elite), dtype=object)
        elites[:] = [genome.carry_in_next() for genome in arr[elite]]
        if self.mutation is not None:
            self.mutation(arr_children, self.genomeMutationChance, self.geneMutationChance)
        local_genomes = np.hstack([elites, arr_children])
        until = np.min([local_genomes.size, self.genomes.size])
        self.genomes = local_genomes[:until]

        self.fitness_val = self.fitness(self.genomes)
        if callable(self.callback):
            self.callback(self)
        self.generation += 1
        return self.genomes
