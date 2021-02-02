import numpy as np
from scipy.spatial import ConvexHull
from database import Graph, AngularGraphSolution
from utils import DependencyNode, convert_graph_to_angular_abstract_graph, calculate_order, get_dep_graph, CircularDependencyException
from .solver_class import Solver

class SnkMirrorSolver(Solver):
    def __init__(self, **kwargs):
        pass

    def is_multicore(self):
        return True

    def solve(self, graph: Graph, **kwargs):
        # Sort'em
        hull = ConvexHull(graph.vertices)
        # Indices are in counter clockwise order, therefore reverse them
        indices = [vertex for vertex in reversed(hull.vertices)]
        length = len(indices)
        assert length == graph.vert_amount
        k = graph.ad_matrix[0].sum() / 2
        assert k.is_integer()
        k = int(k)
        neighbors = {indices[i]: tuple(indices[int(j % length)] for j in range(i-k, i+k+1) if i != j) for i in range(length)}
        # Just check if every node has 2k neighbors
        for index in indices:
            #neighbors = tuple(indices[int(j % length)] for j in range(i-k, i+k+1) if i != j)
            assert graph.ad_matrix[index, neighbors[index]].sum() == 2*k
        tripel_edges, abs_graph = convert_graph_to_angular_abstract_graph(graph, simple_graph=False, return_tripel_edges=True)
        edge_vert = {i: {} for i in range(length)}
        for edge in tripel_edges:
            edge_vert[edge[0]][edge[1:]] = tripel_edges[edge]
            #edge_vert[edge[0]][tuple(reversed(edge[1:]))] = tripel_edges[edge]
        taken_edges = []
        
        # Phase one
        self._phase_one(indices, k, neighbors, taken_edges, edge_vert)
        # Phase two
        split_vertex_index = round(length / 2)
        self._phase_two(split_vertex_index, k, indices, neighbors, taken_edges, edge_vert, length)
        # Clockwise part
        self._phase_three(k, indices, split_vertex_index, neighbors, taken_edges, edge_vert)        
        dep_graph = get_dep_graph(taken_edges, abs_graph)
        order = calculate_order(dep_graph, calculate_circle_dep=True)
        returned_order = [tuple(abs_graph.vertices[i]) for i in order]

        return AngularGraphSolution(graph, 0, self.__class__.__name__, 'MinSum', False, order=reversed(returned_order))

    def _phase_three(self, k, indices, split_vertex_index, neighbors, taken_edges, edge_vert):
        # Clockwise part
        for i in range(k - 1):
            index = indices[split_vertex_index - k + i]
            prev = neighbors[index][0]
            for j in range(1, k):
                edge = (prev, neighbors[index][j])
                taken_edges.append(edge_vert[index][edge])
                prev = edge[1]

            for j in range(2+i, k+1):
                edge = (prev, neighbors[index][-j])
                taken_edges.append(edge_vert[index][edge])
                prev = edge[1]

            for j in range(2*k-i-1, 2*k):
                edge = (prev, neighbors[index][j])
                taken_edges.append(edge_vert[index][edge])
                prev = edge[1]

        # Counterclockwise part
        for i in range(k - 1):
            index = indices[split_vertex_index + k - 1 - i]
            prev = neighbors[index][-1]
            for j in range(2, k+1):
                edge = (prev, neighbors[index][-j])
                taken_edges.append(edge_vert[index][edge])
                prev = edge[1]

            for j in range(1+i, k):
                edge = (prev, neighbors[index][j])
                taken_edges.append(edge_vert[index][edge])
                prev = edge[1]

            for j in range(2*k-i, 2*k+1):
                edge = (prev, neighbors[index][-j])
                taken_edges.append(edge_vert[index][edge])
                prev = edge[1]

        # Clockwise part
        index = indices[split_vertex_index-1]
        prev = neighbors[index][0]
        for j in range(1, k):
            edge = (prev, neighbors[index][j])
            taken_edges.append(edge_vert[index][edge])
            prev = edge[1]
        for j in range(1, k+1):
            edge = (prev, neighbors[index][-j])
            taken_edges.append(edge_vert[index][edge])
            prev = edge[1]

        # Counterclockwise part
        index = indices[split_vertex_index]
        prev = neighbors[index][-1]
        for j in range(2, k+1):
            edge = (prev, neighbors[index][-j])
            taken_edges.append(edge_vert[index][edge])
            prev = edge[1]
        for j in range(k):
            edge = (prev, neighbors[index][j])
            taken_edges.append(edge_vert[index][edge])
            prev = edge[1]

    def _phase_two(self, split_vertex_index, k, indices, neighbors, taken_edges, edge_vert, length):
        # Clockwise part
        for i in range(1, split_vertex_index - k):
            index = indices[i]
            prev = neighbors[index][0]
            for j in range(1, k):
                edge = (prev, neighbors[index][j])
                taken_edges.append(edge_vert[index][edge])
                prev = edge[1]
            for j in range(1, k+1):
                edge = (prev, neighbors[index][-j])
                taken_edges.append(edge_vert[index][edge])
                prev = edge[1]

        # Counterclockwise part
        for i in range(2, length - k - split_vertex_index + 1):
            index = indices[-i]
            prev = neighbors[index][-1]
            for j in range(2, k+1):
                edge = (prev, neighbors[index][-j])
                taken_edges.append(edge_vert[index][edge])
                prev = edge[1]
            for j in range(k):
                edge = (prev, neighbors[index][j])
                taken_edges.append(edge_vert[index][edge])
                prev = edge[1]

    def _phase_one(self, indices, k, neighbors, taken_edges, edge_vert):
        start_positive_order = indices[0]     
        start_reverse_order = indices[-1]

        for j in range(2*k - 1):
            edge = (neighbors[start_positive_order][(k-j-1) % (2*k)], neighbors[start_positive_order][(k-j-2) % (2*k)])
            taken_edges.append(edge_vert[start_positive_order][edge])

        for j in range(2*k - 1):
            edge = (neighbors[start_reverse_order][(k+j) % (2*k)], neighbors[start_reverse_order][(k+j+1) % (2*k)])
            taken_edges.append(edge_vert[start_reverse_order][edge])
        # Go around graph clockwise and add edges for every vertex
        

        
        
