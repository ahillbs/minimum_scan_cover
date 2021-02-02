from typing import List, Union, Optional
import time
import math
import numpy as np
import itertools
import tqdm
from os import cpu_count
import concurrent.futures
from solver import Solver, AngularMinSumGreedySolver, AngularMinSumGreedyConnectedSolver
from solver.s_n_k_mirror_solver import SnkMirrorSolver
from database import Graph, AngularGraphSolution
from utils import calculate_times, get_graph_angles



class AngularIteratedMinSumSolver(Solver):
    solution_type = "min_sum"
    allowed_sub_solver = {
        "AngularMinSumGreedyConnectedSolver": AngularMinSumGreedyConnectedSolver, "AngularMinSumGreedySolver": AngularMinSumGreedySolver,
        "SnkMirrorSolver": SnkMirrorSolver
    }
    def is_multicore(self):
        return True
    
    def __init__(self, sub_solver=None, sub_solver_args={}, time_limit=900, max_iterations: Optional[int] = None):
        assert sub_solver is None or sub_solver in self.allowed_sub_solver,\
            "No allowed subsolver chosen!"
        self.sub_solver = self.allowed_sub_solver[sub_solver](**sub_solver_args).solve if sub_solver is not None else self._get_random_sol
        self.time_limit = time_limit
        self.max_iterations = max_iterations

    def _get_random_sol(self, graph: Graph) -> AngularGraphSolution:
        order = np.random.permutation(graph.edges).tolist()
        sol = AngularGraphSolution(
            graph,
            0,
            self.__class__.__name__,
            self.solution_type,
            is_optimal=False,
            order=order
        )
        return sol

    def solve(self, graph: Graph, start_solution=None, **kwargs):
        time_limit = float(kwargs.pop("time_limit", self.time_limit))
        start_time = time.time()
        
        if graph.costs is None:
            angles = get_graph_angles(graph)
            graph.costs = angles
        else:
            angles = [i.copy() for i in graph.costs]

        solution = self.sub_solver(graph) if start_solution is None else start_solution
        improvement = True
        # solution can be AngularGraphSolution or simple edge order
        best_order = np.array(solution.order) if isinstance(solution, AngularGraphSolution) else np.array(solution)
        best_cost = self._get_cost(solution, angles)
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            try:
                error_message = None
                with tqdm.tqdm(total=self.max_iterations, desc=f"Iterating local neighborhood. Best Sol: {best_cost}") as progressbar:
                    iteration = 0
                    while (self.max_iterations is None or iteration < self.max_iterations) and\
                          (time_limit or time.time() - start_time < time_limit) and\
                          (improvement):
                        improvement = False
                        candidate_order = None
                        candidate_cost = None
                        futures = [
                            executor.submit(
                                self._inner_solve, [i.copy() for i in angles], i, best_order, best_cost, time_limit, start_time
                            ) for i in range(graph.edge_amount)
                        ]
                        timeout = time_limit - (time.time() - start_time) + 10
                        concurrent.futures.wait(futures, timeout=timeout)
                        for future in futures:
                            if future.done():
                                order, cost = future.result()
                                if candidate_cost is None or cost < candidate_cost:
                                    candidate_cost = cost
                                    candidate_order = order

                        if candidate_cost is not None and candidate_cost < best_cost:
                            improvement = True
                            best_cost = candidate_cost
                            best_order = candidate_order
                            progressbar.set_description(f"Iterating local neighborhood. Best Sol: {best_cost}")
                        if not improvement:
                            break
                        progressbar.update()
                        iteration += 1
            except concurrent.futures.TimeoutError as time_error:
                error_message = str(time_error) + " in iteration " + str(iteration)
        sol = AngularGraphSolution(
            graph,
            time.time() - start_time,
            self.__class__.__name__,
            self.solution_type,
            is_optimal=False,
            order=best_order.tolist(),
            error_message=error_message
        )
        return sol

    def _inner_solve(self, angles, i, curr_order, curr_cost, time_limit, start_time):
        if (time_limit or time.time() - start_time < time_limit):
            return None, None
        best_cost = curr_cost
        best_order = curr_order
        for j in range(len(curr_order)):
            if j == i:
                continue
            swap = (i, j)
            order, cost = self._calculate_swap(angles, swap, curr_order)
            if cost < best_cost:
                best_cost = cost
                best_order = order
        return best_order, best_cost


    def _get_cost(self, solution: Union[AngularGraphSolution, List], angles: Optional = None):
        if isinstance(solution, AngularGraphSolution):
            return solution.min_sum
        times, heads = calculate_times(solution, angles=angles, return_angle_sum=True)
        return sum(heads)
    
    def _calculate_swap(self, angles, swap: List[int], best_order: np.ndarray):
        new_order = np.array(best_order)
        new_order[list(swap)] = new_order[list(reversed(swap))]
        min_sum = self._get_cost(new_order, angles=angles)
        return new_order, min_sum

class AngularIteratedLocalMinSumSolver(AngularIteratedMinSumSolver):
    solution_type = "local_min_sum"

    def _get_cost(self, solution: Union[AngularGraphSolution, List], angles: Optional = None):
        if isinstance(solution, AngularGraphSolution):
            return solution.local_min_sum
        times, heads = calculate_times(solution, angles=angles, return_angle_sum=True)
        return max(heads)
    

class AngularIteratedMakespanSolver(AngularIteratedMinSumSolver):
    solution_type = "makespan"

    def _get_cost(self, solution: Union[AngularGraphSolution, List], angles: Optional = None):
        if isinstance(solution, AngularGraphSolution):
            return solution.makespan
        times = calculate_times(solution, angles=angles)
        return sum([times[key] for key in times])
