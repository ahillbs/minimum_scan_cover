from typing import List
import math
import configargparse
import numpy as np
import sqlalchemy
import yaml
import sys
import tqdm
from IPython import embed

from celery import group

from angular_solver import solve, bulk_solve
from database import Config, ConfigHolder, Graph, Task, TaskJobs, get_session
from solver import ALL_SOLVER
from utils import is_debug_env

class OnMessageCB:
    def __init__(self, progressbar: tqdm.tqdm) -> None:
        super().__init__()
        self.progressbar = progressbar
    
    def __call__(self, body: dict) -> None:
        if body["status"] in ['SUCCESS', 'FAILURE']:
            if body["status"] == 'FAILURE':
                print("Found an error:", body)
            try:
                self.progressbar.update()
            except AttributeError:
                pass

def _load_config():
    parser = configargparse.ArgumentParser(description="Parser for the solver tests")
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file',
        is_config_file_arg=True)
    parser.add_argument('--create-only', action="store_true", help="Only creates task and jobs; not process them")
    parser.add_argument('--solvers', required=True, type=str, nargs='+', help="Name of the solvers that shall be used")
    parser.add_argument('--solvers-args', nargs='+', type=yaml.safe_load, help="Arguments for solver instatiation")
    parser.add_argument('--url-path', type=str, help="Path to database")
    parser.add_argument('--min-n', type=int, default=5, help="Minimal amount of vertices a graph can have")
    parser.add_argument('--max-n', type=int, default=None, help="Maximal amount of vertices a graph can have")
    parser.add_argument('--min-m', type=int, default=0, help="Minimal amount of edges a graph can have")
    parser.add_argument('--max-m', type=int, default=sys.maxsize, help="Maximal amount of vertices a graph can have")
    parser.add_argument('--instance-types', type=str, nargs="*", default=[], help="Types of instances you want to select. Default will be all instance types")
    parser.add_argument('--task-id', type=int, default=None, help="Only select instances belonging to a specific task. Default will select from all tasks.")
    parser.add_argument('--max-amount', type=int, help="Maximum amount of instances that will be tested")
    parser.add_argument('--repetitions', type=int, default=1, help="Amount of repetitions for every test for every solver")
    parser.add_argument('--slice-size', type=str, default="auto", help="Slice sizes for bulk solves if needed (Default: auto)")
    parser.add_argument('--manual-query', action="store_true", help="Instead of standard query arguments, open ipython to construct custom query")
    parser.add_argument('--name', type=str, default="Main_instance_test", help="Describing name for the task")
    parser.add_argument('--with-start-sol', action="store_true", default=False, help="NEED: Preious solution from task-id instances! Starts solving with start solution")
    parsed = parser.parse_args()
    return parsed

def _create_task(arg_config, session):
    solvers = arg_config.solvers
    solvers_args = arg_config.solvers_args
    assert len(solvers) == len(solvers_args),\
        "The amount of solver arguments must match the amount of solvers"
    for solver in solvers:
        assert solver in ALL_SOLVER,\
            f"Solver {solver} not found! Please make sure that all solver are properly named."
    task = Task(task_type="instance_test", status=Task.STATUS_OPTIONS.CREATED, name=arg_config.name)
    config = ConfigHolder.fromNamespace(arg_config, task=task,
                                        ignored_attributes=["url_path", "solvers", "solvers_args", "create_only", "config", "name"])
    jobs = _get_instances(task, config, session)
    for solver, solver_args in zip(solvers, solvers_args):
        subtask = Task(parent=task, name=f"{solver}_test", task_type="instance_test", status=Task.STATUS_OPTIONS.CREATED)
        task.children.append(subtask)
        subconfig_namespace = configargparse.Namespace(solver=solver,
                                                       solver_args=solver_args)
        subconfig = ConfigHolder.fromNamespace(subconfig_namespace, task=subtask)
        add_prev_job = (subconfig.with_start_sol is not None and subconfig.with_start_sol)
        if isinstance(jobs[0], TaskJobs):
            for task_job in jobs:
                prev_job = task_job if add_prev_job else None
                for i in range(config.repetitions):
                    subtask.jobs.append(TaskJobs(task=subtask, graph=task_job.graph, prev_job=prev_job))
        else:
            for graph in jobs:
                for i in range(config.repetitions):
                    subtask.jobs.append(TaskJobs(task=subtask, graph=graph))
    session.add(task)
    session.commit()
    return task, config

def _get_instances(task, config: ConfigHolder, session: sqlalchemy.orm.Session):
    if config.manual_query:
        query = None
        print("Manual query chosen. Please fill a query. After finishing the query just end ipython.\n\
            Query result must be of type Graph or TaskJobs!")
        embed()
        assert query is not None, "query must be filled!"
        session.add(Config(task=task, value=query.statement(), param="statement"))
        return query.all()

    if config.task_id is None:
        query = session.query(Graph)
    else:
        query = session.query(TaskJobs).join(Graph).filter(TaskJobs.task_id == config.task_id)
    if config.min_n is not None:
        query = query.filter(Graph.vert_amount >= config.min_n)
    if config.max_n is not None:
        query = query.filter(Graph.vert_amount <= config.max_n)
    if config.min_m is not None:
        query = query.filter(Graph.edge_amount >= config.min_m)
    if config.max_m is not None:
        query = query.filter(Graph.edge_amount <= config.max_m)
    if config.instance_types:
        query = query.filter(Graph.i_type.in_(config.instance_types))
    if config.max_amount is not None:
        query = query[:config.max_amount]
    return query[:]

def process_task(config: ConfigHolder, task: Task, session: sqlalchemy.orm.Session):
    try:
        task.status = Task.STATUS_OPTIONS.PROCESSING
        session.commit()
        for subtask in tqdm.tqdm(task.children, desc=f"Task {task.id}: Processing subtasks"):
            if subtask.status not in [Task.STATUS_OPTIONS.ERROR, Task.STATUS_OPTIONS.INTERRUPTED, Task.STATUS_OPTIONS.FINISHED]:
                subconfig = ConfigHolder(subtask)
                if config.local:
                    subconfig.local = True
                process_task(subconfig, subtask, session)
        to_process = [job for job in task.jobs if job.solution is None]
        process_jobs(to_process, config, session)
        task.status = Task.STATUS_OPTIONS.FINISHED
    except Exception as e:
        print(e)
        to_process = [job for job in task.jobs if job.solution is None]
        if str(e).lower() != "Backend does not support on_message callback".lower() and to_process:
            task.status = Task.STATUS_OPTIONS.ERROR
            task.error_message = str(e)
            if is_debug_env():
                raise e
        else:
            task.status = Task.STATUS_OPTIONS.FINISHED
    finally:
        session.commit()

def _get_slicing(unsolved, slicing):
    if slicing == 'auto':
        slice_size = 16
        slice_amount = math.ceil(len(unsolved) / 5)
    else:
        slice_size = slicing
        slice_amount = math.ceil(len(unsolved) / slice_size)
    return slice_size, slice_amount




def process_jobs(jobs: List[TaskJobs], config: ConfigHolder, session: sqlalchemy.orm.Session):
    if not jobs:
        return
    processbar = tqdm.tqdm(total=len(jobs), desc=f"Task {jobs[0].task_id}: Process jobs")    
    on_message = OnMessageCB(progressbar=processbar)
    # ToDo: To speed up solving time, maybe use bulksolve
    slice_size, slice_amount = _get_slicing(jobs, config.slice_size)
    slices = [(i*slice_size, (i+1)*slice_size) for i in range(slice_amount-1)]
    if slice_amount > 0:
        slices.append(tuple([(slice_amount-1)*slice_size, len(jobs)]))
    solver_args = config.solver_args
    if "time_limit" in solver_args:
        time_limit = solver_args["time_limit"]
    else:
        time_limit = 900
    if hasattr(config, "local") and config.local:
        for job in jobs:
            sol = solve(
                job.graph,
                config.solver,
                solver_config=config.solver_args,
                solve_config={
                    "start_solution":(None if job.prev_job is None else job.prev_job.solution.order),
                    "time_limit":(time_limit if job.prev_job is None else time_limit - job.prev_job.solution.runtime)
                    }
                )
            job.solution = sol
            processbar.update()
            session.commit()
    else:
        for start, end in slices:
            results = group(solve.s(
                    job.graph,
                    config.solver,
                    solver_config=config.solver_args,
                    solve_config={
                        "start_solution":(None if job.prev_job is None else job.prev_job.solution.order),
                        "time_limit":(time_limit if job.prev_job is None else time_limit - job.prev_job.solution.runtime)
                        }
                )
                for job in jobs[start:end])().get(on_message=on_message)

            for job, result in zip(jobs[start:end], results):
                result.graph = job.graph
                if job.prev_job is not None:
                    result.runtime = float(result.runtime) + float(job.prev_job.solution.runtime)
                job.solution = result
            if session:
                session.commit()
            

def main():
    parsed_config = _load_config()
    session = get_session(parsed_config.url_path)
    task, config = _create_task(parsed_config, session)
    if not parsed_config.create_only:
        process_task(config, task, session)

if __name__ == "__main__":
    main()