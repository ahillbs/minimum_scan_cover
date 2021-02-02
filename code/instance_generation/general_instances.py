import numpy as np
from database import Graph

def create_random_instance_fixed_edges(point_amount, edge_amount, bounds=(0, 500), seed=None):
    gen = np.random.default_rng(seed)
    points = gen.integers(bounds[0], bounds[1], size=(point_amount, 2))
    ad_matrix = np.zeros((len(points), len(points)))
    tril_indices = np.tril_indices_from(ad_matrix, -1)
    l = len(tril_indices[0])
    chosen = gen.choice(range(l), size=edge_amount, replace=False)
    chosen_indices = np.zeros(l)
    chosen_indices[chosen] = 1
    ad_matrix[tril_indices] = chosen_indices
    ad_matrix = ad_matrix + ad_matrix.T
    graph = Graph(points, ad_matrix=ad_matrix)
    return graph

def create_random_instance(point_amount, bounds=(0, 500), edge_chance=0.5, seed=None):
    gen = np.random.default_rng(seed)
    points = gen.integers(bounds[0], bounds[1], size=(point_amount, 2))
    ad_matrix = np.zeros((len(points), len(points)))
    tril_indices = np.tril_indices_from(ad_matrix, -1)
    tril_edges = np.array([1 if i < edge_chance else 0 for i in gen.random(size=len(tril_indices[0]))])
    while tril_edges.max() == 0:
        tril_edges = np.array([1 if i < edge_chance else 0 for i in gen.random(size=len(tril_indices[0]))])
    ad_matrix[tril_indices] = tril_edges
    ad_matrix = ad_matrix + ad_matrix.T
    graph = Graph(points, ad_matrix=ad_matrix)
    return graph