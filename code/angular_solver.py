'''
Celery task to solve instances with a solver

To initialize worker thread to the following:
- Change directory to the folder containing this file
- Call command:
    celery -A angular_solver -Q angular_task --concurrency=1 -O fair worker
'''
import concurrent.futures
import math
import sys
from os import cpu_count
from typing import List

from celery import Celery

from database import AngularGraphSolution, Graph
from solver import ALL_SOLVER

from utils import is_debug_env

log = ''
app = Celery('angular_solver', config_source='celeryconfig')

@app.task
def solve(graph, solver_str, solver_config=None, solve_config=None):
    if solve_config is None:
        print("solve_config param was None. Changing to empty dict")
        solve_config = {}
    if solver_config is None:
        print("solver_config param was None. Changing to empty dict")
        solver_config = {}
    try:
        solver = ALL_SOLVER[solver_str](**solver_config)
    except KeyError:
        solver = ALL_SOLVER[solver_str]()
    
    if solve_config:
        return solver.solve(graph, **solve_config)
    return solver.solve(graph)

@app.task
def bulk_solve(to_solve: List[Graph], solver_str, solver_config=None, solve_config=None, slice_size='auto', threads='auto'):
    results = {}
    futures = []
    solver_config = solver_config if solver_config is not None else {}
    solve_config = solve_config if solve_config is not None else {}
    if threads == 'auto':
        threads = cpu_count()
    if slice_size == 'auto':
        slice_size = min([round(len(to_solve) / threads), 1])
        slice_amount = threads
    else:
        slice_amount = math.ceil(len(to_solve) / slice_size)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        inner_solver = ALL_SOLVER[solver_str](**solver_config)

        for i in range(slice_amount-1):
            futures.append(
                executor.submit(
                    _inner_solve,
                    range(i*slice_size, (i+1)*slice_size),
                    to_solve,
                    inner_solver))
        futures.append(
            executor.submit(
                _inner_solve,
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

def _inner_solve(numbers, graphs, solver, **kwargs) -> List[AngularGraphSolution]:
    result = []
    for i in numbers:
        graph = graphs[i]
        sol = solver.solve(graph, **kwargs)
        result.append((i, sol))
    return result
