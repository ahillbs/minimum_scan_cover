import pytest
import numpy as np
from instance_generation import create_circle_n_k, create_random_instance_fixed_edges
from solver import GraphN2Solver
from solver.s_n_2_solver import PointSweep
from utils import DependencyNode

def test_dependency_graph():
    circle = create_circle_n_k(6, 2)
    solver = GraphN2Solver()
    sweep_types = [0, -1, 0, 0, 0, 1]
    directions = [-1, -1, 1, -1, 1, 1]
    orientations = [
        PointSweep(direction, sweep_type, index)
        for direction, sweep_type, index
        in zip(directions, sweep_types, range(len(directions)))
        ]

    dep_graph = solver._calculate_dependecy_graph(circle, orientations)

    orig_dep_dict = {
        (0, 1): [(0, 2), (1, 5)],
        (0, 2): [(0, 4), (2, 4)],
        (1, 2): [(0, 2), (1, 3)],
        (1, 3): [(2, 3), (0, 1)],
        (2, 3): [],
        (2, 4): [(2, 3), (0, 4)],
        (3, 4): [(3, 5), (2, 4)],
        (3, 5): [(1, 3), (4, 5)],
        (4, 5): [(1, 5)],
        (0, 4): [(4, 5), (0, 5)],
        (0, 5): [],
        (1, 5): [(0, 5)]
    }
    orig_dep_graph = {edge: DependencyNode(edge) for edge in orig_dep_dict}
    for edge in orig_dep_dict:
        for dependency in orig_dep_dict[edge]:
            orig_dep_graph[edge].add_dependency(orig_dep_graph[dependency])
    
    assert orig_dep_graph == dep_graph

def test_generate_possible_sweeps():
    circle = create_circle_n_k(6, 2)
    solver = GraphN2Solver()
    possibilities = solver._generate_possible_start_ends(circle, 2)    
    sweep = solver._calculate_orientations(circle, possibilities[0])
    sweep_types = [-1, -1, 0, 0, 0, 0]
    directions = [1, 1, -1, 1, -1, 1]
    for sweep_point, sweep_type, direction in zip(sweep, sweep_types, directions):
        assert sweep_point.direction == direction
        assert sweep_point.type == sweep_type
    
def test_order():
    circle = create_circle_n_k(6, 2)
    solver = GraphN2Solver()
    sweep_types = [0, -1, 0, 0, 0, 1]
    directions = [-1, -1, 1, -1, 1, 1]
    orientations = [
        PointSweep(direction, sweep_type, index)
        for direction, sweep_type, index
        in zip(directions, sweep_types, range(len(directions)))
        ]

    dep_graph = solver._calculate_dependecy_graph(circle, orientations)
    order = solver._calculate_order(dep_graph)
    
def test_solver():
    circles = [create_circle_n_k(n, 2) for n in range(7, 35)]
    solver = GraphN2Solver()
    sols = [solver.solve(circle) for circle in circles]
    for sol in sols:
        assert sol, "Did not find a solution for a graph"

def test_fixed_edges_instance_generation():
    n = 10
    m = 35
    graph = create_random_instance_fixed_edges(n, m)
    assert graph.edge_amount == m