"""Evolver functions and classes for angular problem variations
"""
from .graph_genome import CompleteGraphGenome, GraphGenome
from .fitness import AngularSolverFitness
from .mutation import mutate_2d_points, check_duplicates, mutate_vertex_edge_genomes
from .instance_factory import CompleteGraphGenomeCreator, GraphGenomeCreator
