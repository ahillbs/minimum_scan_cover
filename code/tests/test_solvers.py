import pytest
import math
import numpy as np

from instance_generation import create_circle_n_k
from database import Graph

from solver.s_n_k_mirror_solver import SnkMirrorSolver
from solver.cp import ConstraintAbsSolver, ConstraintDependencyLocalMinSumSolver, ConstraintDependencySolver
from create_gen_instances_script import ConfigHolder, _fixed_generation

def test_snk_mirror_solver():
    circle = create_circle_n_k(9, 3)
    solver = SnkMirrorSolver()
    sol = solver.solve(circle)
    circle = create_circle_n_k(12, 3)
    sol1 = solver.solve(circle)
    circle = create_circle_n_k(14, 4)
    sol2 = solver.solve(circle)
    circle = create_circle_n_k(20, 5)
    sol3 = solver.solve(circle)
    print("\\o/")

def test_cp_lcm_gcd():
    solver_classes = [ConstraintAbsSolver, ConstraintDependencySolver, ConstraintDependencyLocalMinSumSolver]
    solvers_lcm = [sol(use_lcm=True) for sol in solver_classes]
    solvers_gcd = [sol() for sol in solver_classes]
    gen = np.random.default_rng(0)
    config = ConfigHolder(None)
    config.edge_min = 10
    config.edge_max = 25
    config.min_n = 5
    config.max_n = 9
    config.max_amount = 50
    graphs = _fixed_generation(config, gen)
    for g in graphs:
        for sv_lcm, sv_gcd in zip(solvers_lcm, solvers_gcd):
            s_lcm = sv_lcm.solve(g)
            s_gcd = sv_gcd.solve(g)
            if s_lcm.is_optimal and s_gcd.is_optimal:
                if s_lcm.solution_type == "makespan":
                    assert math.isclose(s_lcm.makespan, s_gcd.makespan, abs_tol=10**-5)
                elif s_lcm.solution_type == "min_sum":
                    assert math.isclose(s_lcm.min_sum, s_gcd.min_sum)
                elif s_lcm.solution_type == "local_min_sum":
                    assert math.isclose(s_lcm.local_min_sum, s_gcd.local_min_sum)
