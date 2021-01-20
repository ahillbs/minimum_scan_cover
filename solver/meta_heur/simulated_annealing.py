from typing import List, Union, Optional
import time
import math
import numpy as np
import itertools
import tqdm
from os import cpu_count
import concurrent.futures
from solver import Solver
from solver.s_n_k_mirror_solver import SnkMirrorSolver
from database import Graph, AngularGraphSolution
from utils import calculate_times, get_graph_angles

class BolzmanHeatWithReheat():
    def __init__(self, start_temperature=1000, when_reheat=500, reheat_amount=400):
        self.start_temperature = start_temperature
        self.when_reheat = when_reheat
        self.reheat_amount = reheat_amount
        self.temperature = start_temperature

    def __call__(self, best_cost, curr_cost, iteration, no_improvement):
        chances = self.calculate_chances(no_improvement, curr_cost, best_cost)
        return np.random.random() < chances

    def calculate_chances(self, no_improvement, curr_cost, best_cost):
        self.temperature = max(1, self.temperature-1)
        if no_improvement % self.when_reheat == 0:
            self.temperature += self.reheat_amount
        chances = np.e**(-(abs(curr_cost - best_cost))/self.temperature)
        return chances

class AngularSimulatedAnnealingMinSumSolver(Solver):
    solution_type = "min_sum"
    
    def is_multicore(self):
        return True
    
    def __init__(self, sub_solver=None, sub_solver_args={}, time_limit=900, max_iterations: Optional[int] = None, max_no_improve: Optional[int] = 6000, heat_function=None):
        from solver import ALL_SOLVER
        assert sub_solver is None or sub_solver in ALL_SOLVER,\
            "No allowed subsolver chosen!"
        self.sub_solver = ALL_SOLVER[sub_solver](**sub_solver_args).solve if sub_solver is not None else self._get_random_sol
        self.time_limit = time_limit
        self.max_time = time_limit
        self.max_iterations = max_iterations
        self.max_no_improve = max_no_improve
        self.heat_function = heat_function

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
    
    def _get_cost(self, solution: Union[AngularGraphSolution, List], angles: Optional = None):
        if isinstance(solution, AngularGraphSolution):
            return solution.order, solution.min_sum
        times, heads = calculate_times(solution, return_angle_sum=True, angles=angles)
        return solution, sum(heads)

    def solve(self, graph: Graph, start_solution=None, **kwargs):
        self.time_limit = float(kwargs.pop("time_limit", self.max_time))
        start_time = time.time()
        
        #    graph.costs = get_graph_angles(graph)
        solution = self.sub_solver(graph) if start_solution is None else start_solution
        angles = None
        if not isinstance(solution, AngularGraphSolution):
            angles = get_graph_angles(graph)
        best_order, best_cost = self._get_cost(solution, angles)
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            try:
                error_message = None
                futures = [
                    executor.submit(
                        self._inner_solve, graph, best_order
                    ) for swap in range(cpu_count())
                ]
                timeout = self.time_limit - (time.time() - start_time) + 10
                concurrent.futures.wait(futures, timeout=timeout)
            except TimeoutError as time_error:
                error_message = str(time_error)
            for future in futures:
                if future.done():
                    order, cost = future.result()
                    if cost < best_cost:
                        best_cost = cost
                        best_order = order
            if isinstance(best_order, np.ndarray):
                best_order = best_order.tolist()
            sol = AngularGraphSolution(
                graph,
                time.time() - start_time,
                self.__class__.__name__,
                self.solution_type,
                is_optimal=False,
                order=best_order,
                error_message=error_message
            )
            return sol
    
    

    def _inner_solve(self, graph: Graph, first_sol: Union[AngularGraphSolution, List]):
        angles = get_graph_angles(graph)
        start_time = time.time()
        time_limit = self.time_limit
        max_iter = self.max_iterations
        max_no_impr = self.max_no_improve
        best_order, best_cost = self._get_cost(first_sol, angles=angles)
        curr_cost = best_cost
        curr_order = best_order
        heat_function = self.heat_function if self.heat_function is not None else\
            BolzmanHeatWithReheat()

        neighborhood = np.array(
            [neighbor for neighbor in itertools.permutations(range(len(best_order)), 2)]
            )
        i = 0
        improved_in_step = 0
        while (max_iter is None or i < max_iter) and\
              (max_no_impr is None or i - improved_in_step < max_no_impr) and\
              (time.time() - start_time < time_limit):
            swap_index = np.random.randint(0, len(neighborhood), size=1)
            swap = neighborhood[swap_index]
            order, cost = self._calculate_swap(angles, swap, curr_order)
            if cost < curr_cost or heat_function(best_cost, cost, i, i-improved_in_step):
                improved_in_step = i if cost < best_cost else improved_in_step
                curr_cost = cost
                curr_order = order
                if cost < best_cost:
                    best_cost = cost
                    best_order = order
            i += 1
        return best_order, best_cost

    def _calculate_swap(self, angles, swap: List[int], best_order: np.ndarray):
        new_order = np.array(best_order)
        new_order[swap[0]] = new_order[[i for i in reversed(swap[0])]]
        order, cost = self._get_cost(new_order, angles=angles)
        return new_order, cost
    
class AngularSimulatedAnnealingLocalMinSumSolver(AngularSimulatedAnnealingMinSumSolver):
    solution_type = "local_min_sum"

    def _get_cost(self, solution: Union[AngularGraphSolution, List], angles: Optional = None):
        if isinstance(solution, AngularGraphSolution):
            return solution.order, solution.local_min_sum
        times, heads = calculate_times(solution, return_angle_sum=True, angles=angles)
        return solution, max(heads)

class AngularSimulatedAnnealingMakespanSolver(AngularSimulatedAnnealingMinSumSolver):
    solution_type = "makespan"

    def _get_cost(self, solution: Union[AngularGraphSolution, List], angles: Optional = None):
        if isinstance(solution, AngularGraphSolution):
            return solution.order, solution.makespan
        times, heads = calculate_times(solution, return_angle_sum=True, angles=angles)
        return solution, max([times[key] for key in times])