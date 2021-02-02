import numpy as np
import itertools
from typing import List
from database import Graph, CelestialGraph, CelestialBody
from utils import get_vertex_sectors

def get_lower_bounds(vertices, ad_matrix):
    v_LB = []
    for i in range(len(vertices)):
        sectors = get_vertex_sectors(
            vertices[i],
            vertices[np.nonzero(ad_matrix[i])],
            degrees=True)
        if sectors:
            v_LB.append(360 - max(s[2] for s in sectors))
    if v_LB:
        return v_LB
    return [0]

def create_random_bipartite(point_amount, bounds=(0, 500), edge_chance=0.5, max_cone=np.inf):
    points = np.random.random_integers(bounds[0], bounds[1], size=(point_amount, 2))
    association = np.random.random_integers(0,1, size=(point_amount))
    while association.max() == 0 or association.min() == 1:
        association = np.random.random_integers(0,1, size=(point_amount))
    v_1 = []
    v_2 = []
    for i, a in enumerate(association):
        v_1.append(i) if not a else v_2.append(i)

    ad_matrix = np.zeros((len(points), len(points)))
    
    while ad_matrix.max() == 0:
        for i, j in itertools.product(v_1, v_2):
            if np.random.random() < edge_chance:
                ad_matrix[i, j] = ad_matrix[j, i] = 1
                lb = max(get_lower_bounds(points, ad_matrix))
                if lb > max_cone:
                    ad_matrix[i, j] = ad_matrix[j, i] = 0
    
    graph = Graph(points, ad_matrix=ad_matrix)
    return graph, association
