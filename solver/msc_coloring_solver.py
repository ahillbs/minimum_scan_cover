import math
import time
from itertools import combinations, product

import numpy as np
from pyclustering.gcolor.dsatur import dsatur

from solver.mip import AngularGraphScanMakespanAbsolute, AngularGraphScanMinSumHamilton
from solver.cp import ConstraintAbsSolver
from solver import AngularBipartiteMinSumSolver, AngularBipartiteMakespanSolver
from utils import get_angles, calculate_times
from .coloring_solver import Coloring_IP_Solver, Coloring_CP_Solver
from . import Solver

from database import Graph, AngularGraphSolution

# For the dsatur algorithm, use adjacency matrix as parameter
def solve_dsatur(graph: Graph):
    dsatur_instance = dsatur(graph.ad_matrix)
    dsatur_instance.process()
    coloring = dsatur_instance.get_colors()
    return coloring

def solve_color_ip(graph: Graph, pre_solution=True):
    pre_sol = solve_dsatur(graph) if pre_solution else None
    ip_solver = Coloring_IP_Solver()
    return ip_solver.solve(graph, start_solution=pre_sol)

def solve_color_cp(graph: Graph):
    cp_solver = Coloring_CP_Solver()
    return cp_solver.solve(graph)

DEFAULT_SUBSOLVER = AngularBipartiteMakespanSolver()
ALLOWED_SOLVER = {
    "AngularGraphScanMakespanAbsolute": AngularGraphScanMakespanAbsolute,
    "ConstraintAbsSolver": ConstraintAbsSolver,
    "AngularGraphScanMinSumHamilton": AngularGraphScanMinSumHamilton,
    "AngularBipartiteMinSumSolver": AngularBipartiteMinSumSolver,
    "AngularBipartiteMakespanSolver": AngularBipartiteMakespanSolver
}

class MscColoringSolver(Solver):
    def __init__(self, coloring_solver=solve_dsatur, sub_solver=DEFAULT_SUBSOLVER, no_used_edges=True, greedy_glue=True):
        self.coloring_solver = coloring_solver
        self.sub_solver = sub_solver if isinstance(sub_solver, Solver) else\
            self._instanciate_sub_solver(sub_solver)
        self.no_used_edges = no_used_edges
        self.greedy_glue = greedy_glue
        super().__init__()

    def is_multicore(self):
        return True
    
    def _instanciate_sub_solver(self, sub_solver_str):
        return ALLOWED_SOLVER[sub_solver_str]()

    def solve(self, graph: Graph, **kwargs):
        start_time = time.time()
        times = None
        error_message = None
        try:
            coloring = self.coloring_solver(graph)
            max_color = max(coloring)
            index_colorings = []

            log_max_color = math.ceil(math.log2(max_color))
            color_bit_vector = np.array([
                [(color >> (log_max_color-1 - i)) & 1 for i in range(log_max_color)]
                for color in range(max_color)
            ])

            for vec in color_bit_vector.T:
                # Get indices of the vertices where in one of the sets
                to_color = np.arange(len(vec))[vec == 1]
                isin = np.isin(coloring, to_color)
                both_sets_indices = (*np.where(isin == 0), *np.where(isin == 1))
                index_colorings.append(both_sets_indices)

            solutions = []
            used_edges = set()
            for set_one, set_two in index_colorings:
                subgraph = graph.get_bipartite_subgraph(set_one, set_two, forbidden_edges=used_edges)
                if subgraph.edge_amount > 0:
                    sol = self.sub_solver.solve(subgraph, colors=[int(i in set_one) for i in range(graph.vert_amount)])
                    solutions.append(sol)
                    # If wanted collect the already used edges to maybe get a better solution
                    if self.no_used_edges:
                        used_edges.update({(u) for u in product(set_one, set_two)})

            times = self._calculate_order(solutions, graph)
            if len(times) != graph.edge_amount:
                times = None
                raise Exception("Edge order does not contain all edges!")
        except Exception as e:
            error_message = str(e)
            raise e
        return AngularGraphSolution(graph,
                                    time.time() - start_time,
                                    self.__class__.__name__,
                                    self.sub_solver.solution_type,
                                    times=times,
                                    error_message=error_message)

    def _calculate_order(self, solutions, graph):
        order = []
        had_duplicates = False
        if not self.greedy_glue:
            for sol in solutions:
                for edge in sol.order:
                    # ToDo: DELETE THIS DEBUG LINES
                    duplicate_edge = edge in order
                    had_duplicates = had_duplicates or duplicate_edge
                    # END DEBUG LINES
                order.extend([edge for edge in sol.order if edge not in order])
            times = calculate_times(order, graph)
        else:
            best_order = None
            best_angles = None
            best_times = None
            for sol in solutions:
                order = [edge for edge in sol.order]
                times = calculate_times(order, graph)
                angles = [0 for i in range(graph.vert_amount)]
                remaining_solutions = {sol2 for sol2 in solutions if sol2 != sol}
                while remaining_solutions:
                    next_times = None
                    next_angles = None
                    next_sol = None

                    for sol2 in remaining_solutions:
                        second_order = [edge for edge in sol2.order if edge not in order]
                        orders = [second_order,[i for i in reversed(second_order)]]
                        for new_order in orders:
                            curr_times, angles = calculate_times(new_order, graph, times=times.copy(), return_angle_sum=True)
                            if next_times is None or sum(angles) < sum(next_angles):
                                next_times = curr_times
                                next_angles = angles
                                next_sol = sol2
                                next_order = new_order
                    
                    angles = next_angles

                    times = next_times
                    order = order + next_order
                    remaining_solutions.remove(next_sol)
                if best_order is None or sum(angles) < sum(best_angles):
                    best_angles = angles
                    best_order = order
                    best_times = times
                return best_times
        return times
