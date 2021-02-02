import numpy as np
from genetic_algorithm import (Genome,
                               IterationTerminationConditionMet, NoImprovementsTermination, TerminationCombination,
                               uniform_wheel_selection, update_callback)
from database import Graph
from .fitness import AngularSolverFitness
from .crossovers import OrderUniformCrossover
from solver.greedy import AngularMinSumGreedySolver
from instance_generation import create_circle_n_k

class EdgeOrderGraphGenome(Genome):
    """Genome of graph edge order. This genome does not encode solution orders!
    """
    def __init__(self, orig_edge_graph: Graph, bounds=(0, 500), generation=0, solution=None, skip_graph_calc=False, **kwargs):
        self.orig_graph = orig_edge_graph
        self.bounds = bounds
        self.order_numbers = kwargs.pop("order_numbers", self._create_order_numbers())
        if not skip_graph_calc:
            self.calculate_graph()
        self.generation = generation
        self.solution = solution
        self.skip_graph_calc = skip_graph_calc
        super().__init__(**kwargs)

    def calculate_graph(self):
        """(Re-)calculate the graph with new sorted order_numbers
        """
        sorted_indices = np.argsort(self.order_numbers)
        edges = self.orig_graph.edges[sorted_indices]
        self.graph = Graph(self.orig_graph.vertices, edges)

    def carry_in_next(self):
        return EdgeOrderGraphGenome(
            self.orig_graph,
            parents=[self],
            #task=self.task,
            solution=self.solution,
            generation=self.generation+1,
            bounds=self.bounds,
            order_numbers=self.order_numbers,
            skip_graph_calc=self.skip_graph_calc
            )

    def create_children(self, data, mother=None, **kwargs):
        return EdgeOrderGraphGenome(
            self.orig_graph,
            self.bounds,
            order_numbers=data,
            generation=self.generation+1,
            skip_graph_calc=self.skip_graph_calc
            )

    def __getitem__(self, key):
        return self.order_numbers[key]

    def __len__(self):
        return self.orig_graph.edge_amount

    def _create_order_numbers(self):
        random_order = np.random.randint(np.iinfo(np.int).max, size=self.orig_graph.edge_amount)
        while len(np.unique(random_order)) != self.orig_graph.edge_amount:
            random_order = np.random.randint(np.iinfo(np.int).max, size=self.orig_graph.edge_amount)
        return random_order

class EdgeOrderMutation():
    def __init__(self, **kwargs):
        pass

    def __call__(self, genomes: np.array, genome_mutation_chance=0.03, dna_mutation_chance=0.03):
        genome_size = genomes.shape[0]
        dna_size = len(genomes[0])
        # random choice of chosen genome via random indices
        genomes_chance = np.random.random(genome_size)
        mutation_indices = np.where(genomes_chance < genome_mutation_chance)[0]
        for mutation_index in mutation_indices:
            genome = genomes[mutation_index]
            dna_chances = np.random.random(dna_size)
            mutated_dna = np.where(dna_chances < dna_mutation_chance)[0]
            for strain in mutated_dna:
                genome.order_numbers[strain] = np.random.randint(np.iinfo(np.int).max)
                genome.solution = None
            if not genome.skip_graph_calc:
                genome.calculate_graph()