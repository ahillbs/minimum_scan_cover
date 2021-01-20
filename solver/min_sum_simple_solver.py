import numpy as np

from utils import get_angle
from database import Graph, AngularGraphSolution


def solve_min_sum_simple_n_gon(graph: Graph):
    #ToDo: Check if graph is a simple n-gon
    ordered_vertices = _get_ordered_vertices(graph.vertices)
    length = len(graph.vertices)
    obj_val = 0
    time = 0
    time_dict = {}
    v = graph.vertices[ordered_vertices[0]]
    w = graph.vertices[ordered_vertices[1]]
    u = graph.vertices[ordered_vertices[2]]
    angle = get_angle(v, w, u)
    for i in range(length-1):
        time_dict[(ordered_vertices[i], ordered_vertices[i+1])] = time
        for j in range(i+2, length):
            obj_val += angle
            time += angle
            time_dict[(ordered_vertices[i], ordered_vertices[j])] = time
        obj_val += angle * (length-i-1) # turn all not used vertices by one slice
        if i != length-1: # turn next vertex to his next neighbour
            obj_val += angle * (length-i-1)
            time += angle * (length-i-1)
    print("Objective value:", obj_val)
    return AngularGraphSolution(graph,
                                times=time_dict,
                                runtime=0,
                                obj_val=obj_val,
                                solver=solve_min_sum_simple_n_gon.__name__,
                                solution_type="min_sum")



            

def _get_ordered_vertices(vertices):
    #ToDo: Bring vertices in some order
    return range(len(vertices))