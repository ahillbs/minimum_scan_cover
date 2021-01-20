import random
import math

import pytest
import numpy as np
import gurobipy

from database import Graph
from solver.mip.angular_graph_scan_hamilton import AngularGraphScanMakespanHamilton
from solver.mip.angular_graph_scan_absolute import AngularGraphScanMakespanAbsolute, AngularGraphScanMakespanAbsoluteReduced

def test_angles():
    ip = AngularGraphScanMakespanHamilton()
    v = (1, 1)
    n = 3
    ad_vert = np.zeros([n*n-1], dtype=tuple)
    ad_vert[:] = [(i, j) for i in range(3) for j in range(3) if (i, j) != (v[0], v[1])]
    arc, md = ip._get_angles(v, ad_vert)
    

def test_build_paths():
    ip = AngularGraphScanMakespanHamilton()
    n = 3
    ad_vert = np.zeros([n * n], dtype=tuple)
    ad_vert[:] = [(i, j) for i in range(3) for j in range(3)]
    edges = set([(v1, v2) for v1 in ad_vert for v2 in ad_vert if v1 != v2])
    g = Graph(ad_vert, edges)
    model = gurobipy.Model()
    ip._build_hamilton_path(0)

def test_ip_solver():
    ip = AngularGraphScanMakespanAbsoluteReduced()
    n = 3
    ad_vert = np.zeros([n * n], dtype=tuple)
    ad_vert[:] = [(i, j) for i in range(3) for j in range(3)]
    e_arr = np.triu_indices(n*n, 1)
    edges = {(ad_vert[e_arr[0][i]], ad_vert[e_arr[1][i]]) for i in range(len(e_arr[0]))}

    g = Graph(ad_vert, edges)
    solution = ip.build_ip_and_optimize(g)
    from utils.visualization import visualize_solution_2d, visualize_graph_2d
    #visualize_graph_2d(solution.graph)
    visualize_solution_2d(solution)

def test_ip_solver_not_fully_connected():
    ip = AngularGraphScanMakespanAbsoluteReduced()
    n = 3
    ad_vert = np.zeros([n * n - 1], dtype=tuple)
    ad_vert[:] = [(i, j) for i in range(3) for j in range(3) if i != 1 or i != j]
    e_arr = np.triu_indices(n*n-1, 1)
    edges = [(ad_vert[e_arr[0][i]], ad_vert[e_arr[1][i]]) for i in range(len(e_arr[0]))]
    random.seed(0)

    chosen_edges = random.sample(edges, k=math.ceil(len(edges)/2))
    g = Graph(ad_vert, chosen_edges)
    solution = ip.build_ip_and_optimize(g)
    from utils.visualization import visualize_solution_2d, visualize_graph_2d
    #visualize_graph_2d(solution.graph)
    visualize_solution_2d(solution)

