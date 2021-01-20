import numpy as np

from sqlalchemy import Column, Integer, DECIMAL, ForeignKey
from sqlalchemy.orm import reconstructor

from database import DatabaseGraphGenome, AngularGraphSolution, Graph
from genetic_algorithm import Genome

class CompleteGraphGenome(DatabaseGraphGenome):
    """Genome for graph class
    """
    __tablename__ = 'CompleteGraphGenomes'
    id = Column(Integer, ForeignKey('genomes.id'), primary_key=True)
    lower_bound = Column(DECIMAL)
    upper_bound = Column(DECIMAL)

    __mapper_args__ = {
        'polymorphic_identity':'CompleteGraphGenomes',
    }
    def __init__(self, graph: Graph, bounds=(0, 500), graph_creation_func=None, **kwargs):
        kwargs["bounds"] = bounds
        kwargs["graph"] = graph
        DatabaseGraphGenome.__init__(self, **kwargs)
        try:
            self.solution = kwargs.pop("solution", None)
            self.generation = kwargs.pop("generation")
        except AttributeError:
            pass

        self.bounds = bounds
        self.lower_bound = bounds[0]
        self.upper_bound = bounds[1]
        self.graph_creation_func = graph_creation_func
        

    @reconstructor
    def reconstruct(self):
        self.bounds = (self.lower_bound, self.upper_bound)
        self.graph_creation_func = None

    def __getitem__(self, key):
        return self.graph.vertices[key]

    def __len__(self):
        return len(self.graph.vertices)

    def carry_in_next(self):
        return CompleteGraphGenome(
            self.graph,
            parents=[self],
            task=self.task,
            solution=self.solution,
            generation=self.generation+1,
            bounds=self.bounds,
            )

    def create_children(self, data, mother=None, **kwargs):
        # If no graph creation function is passed just create a fully connected graph
        generation = self.generation
        try:
            generation = max(generation, mother.generation)
        except AttributeError:
            pass
        generation = kwargs.pop("generation", generation+1)

        if self.graph_creation_func is None:
            ad_matrix = np.ones((len(data), len(data)))
            ad_matrix[np.diag_indices_from(ad_matrix)] = 0
            self.graph_creation_func = None
            graph = Graph(data, ad_matrix=ad_matrix)
        else:
            graph = self.graph_creation_func(data)

        return CompleteGraphGenome(
            graph,
            parents=[self, mother],
            graph_creation_func=self.graph_creation_func,
            generation=generation,
            bounds=self.bounds,
            task=self.task
            )

    @classmethod
    def create_random_instance(cls, point_amount, task, bounds=(0, 500), generation=0):
        points = np.random.random_integers(bounds[0], bounds[1], size=(point_amount, 2))
        ad_matrix = np.ones((len(points), len(points)))
        ad_matrix[np.diag_indices_from(ad_matrix)] = 0
        graph = Graph(points, ad_matrix=ad_matrix)
        return CompleteGraphGenome(graph, bounds=bounds, task=task, generation=generation)

class GraphGenome(DatabaseGraphGenome):
    """Genome for graph class
    """
    __tablename__ = 'GraphGenomes'
    id = Column(Integer, ForeignKey('genomes.id'), primary_key=True)
    lower_bound = Column(DECIMAL)
    upper_bound = Column(DECIMAL)
    __mapper_args__ = {
        'polymorphic_identity':'GraphGenomes',
    }

    @property
    def chromosome(self):
        return self._chromosome
    @chromosome.setter
    def set_chromosome(self, value):
        self._chromosome = value
        self.update_graph()

    def update_graph(self):
        vertices, ad_matrix = self._translate_chromosome(self._chromosome)
        self.graph.vertices = vertices
        self.graph.ad_matrix = ad_matrix
        self.graph.update()

    def __init__(self, graph: Graph, bounds=(0, 500), **kwargs):
        kwargs["graph"] = graph
        DatabaseGraphGenome.__init__(self, **kwargs)
        try:
            self.solution = kwargs.pop("solution", None)
            self.generation = kwargs.pop("generation")
        except AttributeError:
            pass

        self.bounds = bounds
        self.lower_bound = bounds[0]
        self.upper_bound = bounds[1]
        self._chromosome = self._generate_chromosome()
    
    @reconstructor
    def reconstruct(self):
        self.bounds = (self.lower_bound, self.upper_bound)

        self._chromosome = self._generate_chromosome()

    def _generate_chromosome(self):
        if self.graph.simple:
            ad_flat = self.graph.ad_matrix[np.tril_indices(self.graph.vert_amount, -1)]
        else:
            array = np.ones((self.graph.vert_amount, self.graph.vert_amount)) - \
                np.diag([1 for i in range(self.graph.vert_amount)])
            nonzeros = np.nonzero(array)
            ad_flat = self.graph.ad_matrix[nonzeros]
        return np.array(self.graph.vertices.tolist() + ad_flat.tolist())

    def __getitem__(self, key):
        return self.chromosome[key]

    def __len__(self):
        return len(self.chromosome)

    def carry_in_next(self):
        return GraphGenome(
            self.graph,
            parents=[self],
            task=self.task,
            solution=self.solution,
            generation=self.generation+1,
            bounds=self.bounds,
            )

    def create_children(self, data, mother=None, **kwargs):
        generation = self.generation
        try:
            generation = max(generation, mother.generation)
        except AttributeError:
            pass
        generation = kwargs.pop("generation", generation+1)

        
        vertices, ad_matrix = self._translate_chromosome(data)

        graph = Graph(V=vertices, ad_matrix=ad_matrix, i_type=self.graph.i_type)

        return GraphGenome(
            graph,
            parents=[self, mother],
            generation=generation,
            bounds=self.bounds,
            task=self.task
            )

    def _translate_chromosome(self, data):
        vertices = np.array(data[:self.graph.vert_amount])
        if self.graph.simple:
            ad_flat = data[self.graph.vert_amount:]
            tril_indices = np.tril_indices(self.graph.vert_amount, -1)
            ad_matrix = np.zeros((self.graph.vert_amount, self.graph.vert_amount))
            ad_matrix[tril_indices] = ad_flat
            ad_matrix = ad_matrix + ad_matrix.T
        else:
            array = np.ones((self.graph.vert_amount, self.graph.vert_amount)) - \
                np.diag([1 for i in range(self.graph.vert_amount)])
            nonzeros = np.nonzero(array)
            ad_matrix = np.zeros((self.graph.vert_amount, self.graph.vert_amount))
            ad_matrix[nonzeros] = data[self.graph.vert_amount:]
        return vertices, ad_matrix

    @classmethod
    def create_random_instance(cls, point_amount, task, bounds=(0, 500), generation=0, edge_chance=0.5):
        points = np.random.random_integers(bounds[0], bounds[1], size=(point_amount, 2))
        ad_matrix = np.zeros((len(points), len(points)))
        tril_indices = np.tril_indices_from(ad_matrix, -1)
        tril_edges = np.array([1 if i < edge_chance else 0 for i in np.random.random(size=len(tril_indices[0]))])
        while tril_edges.max() == 0:
            tril_edges = np.array([1 if i < edge_chance else 0 for i in np.random.random(size=len(tril_indices[0]))])
        ad_matrix[tril_indices] = tril_edges
        ad_matrix = ad_matrix + ad_matrix.T
        graph = Graph(points, ad_matrix=ad_matrix)
        return GraphGenome(graph, bounds=bounds, task=task, generation=generation)