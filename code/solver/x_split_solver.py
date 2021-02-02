import math
import time
from itertools import combinations, product
import numpy as np

from solver import Solver, AngularBipartiteMinSumSolver, AngularBipartiteMakespanSolver
from utils import get_angles, calculate_times

from database import Graph, AngularGraphSolution

DEFAULT_SUBSOLVER = AngularBipartiteMakespanSolver()
ALLOWED_SOLVER = {
    "AngularBipartiteMinSumSolver": AngularBipartiteMinSumSolver,
    "AngularBipartiteMakespanSolver": AngularBipartiteMakespanSolver
}

class SplitXSolver(Solver):
    def __init__(self, sub_solver=DEFAULT_SUBSOLVER, greedy_glue=True):
        self.sub_solver = sub_solver if isinstance(sub_solver, Solver) else\
            self._instanciate_sub_solver(sub_solver)
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
            # Split edges into log n subgraphs along the x-axis
            n = graph.vert_amount
            split_array = np.zeros((math.ceil(math.log2(n)), n))
            self._calculate_split(split_array, 0, 0, n)
            # Calculate subgraphs
            used_edges = set()
            argsorted = np.argsort(graph.vertices[:, 0])
            solutions = []
            self._solve_subgraphs(split_array, argsorted, graph, used_edges, solutions)
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

    def _solve_subgraphs(self, split_array, argsorted, graph, used_edges, solutions):
        for row in split_array:
            bindices_zero = (row == 0)
            set_one = argsorted[bindices_zero]
            set_two = argsorted[~bindices_zero]
            subgraph = graph.get_bipartite_subgraph(set_one, set_two, forbidden_edges=used_edges)
            if subgraph.edge_amount > 0:
                sol = self.sub_solver.solve(subgraph, colors=[int(i in set_one) for i in range(graph.vert_amount)])
                solutions.append(sol)
                # If wanted collect the already used edges to maybe get a better solution
                used_edges.update({(u) for u in product(set_one, set_two)})
        
        
        
    def _calculate_split(self, array, depth, min_x, max_x):
        if max_x - min_x <= 1:
            return
        split = math.ceil((max_x - min_x)/2)+min_x
        array[depth, min_x: split] = 0
        array[depth, split: max_x] = 1
        self._calculate_split(array, depth+1, min_x, split)
        self._calculate_split(array, depth+1, split, max_x)
    

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