import concurrent.futures
import sys
import math
from os import cpu_count
from typing import List, Union

import gurobipy as grb
import numpy as np
import tqdm
from celery import group


from database import AngularGraphSolution
from solver import Solver
from .graph_genome import CompleteGraphGenome



class AngularSolverFitness():
    RUNTIME = 0
    MAKESPAN = 1
    MIN_SUM = 2
    LOCAL_MIN_SUM = 3
    FITNESS_GOALS = [RUNTIME, MAKESPAN, MIN_SUM, LOCAL_MIN_SUM]

    def __init__(self, solver, solver_params, maximize=True,
                 remote_computation=True, concurrent_jobs=None, fitness_goal=0,
                 slicing='auto', custom_result_func=None, zero_fitness_non_optimal=False):
        self.solver = solver
        self.solver_params = solver_params if solver_params is not None else {}
        self.maximize = maximize
        self.tqdm = None
        self.remote_computation = remote_computation
        self.concurrent_jobs = concurrent_jobs
        assert fitness_goal in self.FITNESS_GOALS
        self.fitness_goal = fitness_goal
        self.slicing = slicing
        self.zero_fitness_non_optimal = zero_fitness_non_optimal
        self.custom_result_func = custom_result_func

    def _on_message(self, body):
        if body["status"] in ['SUCCESS', 'FAILURE']:
            if body["status"] == 'FAILURE':
                print("Found an error:", body)
            try:
                if self.concurrent_jobs is None and self.slicing == 'auto':
                    self.tqdm.update()
                else:
                    self.tqdm.update(self.slicing)
            except AttributeError:
                pass

    def __call__(self, genomes: Union[List[CompleteGraphGenome], np.ndarray]):
        unsolved = [genome for genome in genomes if genome.solution is None]

        if unsolved:
            self.tqdm = tqdm.tqdm(total=len(unsolved), initial=0,
                                  desc="Collecting genome results", leave=False)
            if self.remote_computation:
                results = self._async_fitness(unsolved)
            else:
                results = self._concurrent_fitness(unsolved)

            for genome, result in zip(unsolved, results):
                if hasattr(result, "graph") and result.graph != genome.graph:
                    result.graph = genome.graph
                genome.solution = result

        results = self._get_results(genomes) if self.custom_result_func is None \
                    else self.custom_result_func(genomes)
        if unsolved:
            self.tqdm.close()
        return results

    def _get_results(self, genomes):
        direction = 1 if self.maximize else -1
        if self.fitness_goal == AngularSolverFitness.RUNTIME:
            results = np.array([genome.solution.runtime * direction for genome in genomes])
            if self.zero_fitness_non_optimal:
                min_res = results.min()
                results = np.array([genome.solution.runtime * direction
                                    if genome.solution.is_optimal else min(0, min_res)
                                    for genome in genomes])
        elif self.fitness_goal == AngularSolverFitness.MAKESPAN:
            results = np.array([genome.solution.makespan * direction for genome in genomes])
        elif self.fitness_goal == AngularSolverFitness.MIN_SUM:
            results = np.array([genome.solution.min_sum * direction for genome in genomes])
        elif self.fitness_goal == AngularSolverFitness.LOCAL_MIN_SUM:
            results = np.array([genome.solution.local_min_sum * direction for genome in genomes])
        results_min = results.min()
        results = np.array(
            [
                result if genome.solution.error_message is None else results_min
                for result, genome in zip(results, genomes)]
            )
        return results

    def _async_fitness(self, unsolved):
        import angular_solver
        if self.concurrent_jobs is None and self.slicing == 'auto':
            results = group(angular_solver.solve.s(
                genome.graph,
                self.solver,
                solver_config=self.solver_params,
                ) for genome in unsolved)().get(on_message=self._on_message)
        else:
            slice_size, slice_amount = self._get_slicing(unsolved, self.concurrent_jobs)
            to_solve = [genome.graph for genome in unsolved]
            slices = []
            for i in range(slice_amount-1):
                slices.append((i*slice_size, (i+1)*slice_size))
            slices.append(((slice_amount-1)*slice_size, len(to_solve)))
            
            group_results = group(angular_solver.bulk_solve.s(
                    to_solve[i: j],
                    self.solver,
                    solver_config=self.solver_params) for i, j in slices)().get(on_message=self._on_message)
            results = []
            for group_result in group_results:
                for result in group_result:
                    results.append(result)
        return results

    def _concurrent_fitness(self, unsolved):
        results = {}
        futures = []
        concurrent_jobs = self.concurrent_jobs if self.concurrent_jobs is not None else cpu_count()

        to_solve = []
        for i, genome in enumerate(unsolved):
            to_solve.append(genome.graph)

        inner_solver = getattr(sys.modules[__name__], self.solver)(**self.solver_params)
        if inner_solver.is_multicore():
            results = [self._inner_solve([i], to_solve, inner_solver) for i in range(len(to_solve))]
            results_list = [i[0][1] for i in results]
            return results_list
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_jobs) as executor:
                slice_size, slice_amount = self._get_slicing(unsolved, concurrent_jobs)
                if slice_size > 1:
                    for i in range(slice_amount-1):
                        futures.append(
                            executor.submit(
                                self._inner_solve,
                                range(i*slice_size, (i+1)*slice_size),
                                to_solve,
                                inner_solver))
                futures.append(
                    executor.submit(
                        self._inner_solve,
                        range((slice_amount-1)*slice_size, len(to_solve)),
                        to_solve,
                        inner_solver))
                concurrent.futures.wait(futures)
                for future in concurrent.futures.as_completed(futures):
                    result_list = future.result()
                    for i, sol in result_list:
                        results[i] = sol
                results_list = [results[i] for i in sorted(results.keys())]
            return results_list

    def _get_slicing(self, unsolved, concurrent_jobs):
        if self.slicing == 'auto':
            slice_size = round(len(unsolved) / concurrent_jobs)
            slice_amount = concurrent_jobs
        else:
            slice_size = self.slicing
            slice_amount = math.ceil(len(unsolved) / slice_size)
        return slice_size, slice_amount

    def _inner_solve(self, numbers, graphs, solver, **kwargs):
        result = []
        for i in numbers:
            graph = graphs[i]
            sol = solver.solve(graph)
            result.append((i, sol))
            if self.tqdm:
                self.tqdm.update()
        return result
