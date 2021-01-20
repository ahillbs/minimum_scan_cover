"""Converter module for angular graphs"""
from typing import Dict, Tuple
import numpy as np
from . import get_angle
from database import Graph

def convert_graph_to_angular_abstract_graph(graph: Graph, simple_graph=True, return_tripel_edges=False) -> Graph:
    """Converts a graph into an abstract angular graph
    Can be used to calculate a path tsp
    
    Arguments:
        graph {Graph} -- Graph to be converted
        simple_graph {bool} -- Indicates if graph is simple
        return_tripel_edges {bool} -- Also return translation for original edges to abstract 
    
    Returns:
        Graph -- Converted abstract graph
    """
    # create a vertex for every edge in the original graph
    # For geometric instances, only one direction of edges is needed
    vertices = np.array([[u, v] for u, v in graph.edges if u < v])
    edges = {}
    tripel_edges = {}
    for i, vertex in enumerate(vertices):
        ran = range(i+1, len(vertices)) if simple_graph else range(len(vertices))
        for j in ran:
            if j == i:
                continue
            other = vertices[j]
            if np.intersect1d(vertex, other).size > 0:
                shared_vertex = np.intersect1d(vertex, other)
                non_shared = np.setdiff1d(np.hstack([vertex, other]), shared_vertex)
                edges[(i, j)] = get_angle(
                    graph.vertices[shared_vertex],
                    graph.vertices[non_shared[0]],
                    graph.vertices[non_shared[1]]
                )
                if return_tripel_edges:
                    from_vertex = np.intersect1d(vertex, non_shared)
                    to_vertex = np.intersect1d(other, non_shared)
                    edge = (*from_vertex, *to_vertex)
                    tripel_edges[(*shared_vertex, *edge)] = (i, j)
    graph = Graph(vertices, edges.keys(), c=edges)
    if return_tripel_edges:
        return (tripel_edges, graph)
    return graph

def get_tripeledges_from_abs_graph(abstract_graph: Graph) -> Dict[Tuple, Tuple]:
    tripel_edges = {}
    for edge in abstract_graph.edges:
        vert1, vert2 = abstract_graph.vertices[tuple(edge), :]
        
        shared_vertex = np.intersect1d(vert1, vert2)
        non_shared = np.setdiff1d(np.hstack([vert1, vert2]), shared_vertex)

        
        from_vertex = np.intersect1d(vert1, non_shared)
        to_vertex = np.intersect1d(vert2, non_shared)
        from_to = (*from_vertex, *to_vertex)
        tripel_edges[(*shared_vertex, *from_to)] = tuple(edge)
    return tripel_edges
