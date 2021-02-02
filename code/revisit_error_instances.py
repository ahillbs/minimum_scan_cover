import argparse
import tqdm
from database import get_session, Task, TaskJobs, Graph, AngularGraphSolution, Config, ConfigHolder
from solver import ALL_SOLVER
from utils import is_debug_env

def main():
    if not is_debug_env():
        print("No debugger detected! This file is inteded to run with debugger. Proceed without with care")
    arguments = _parse_args()
    session = get_session(arguments.url_path)
    error_jobs = session.query(TaskJobs).join(AngularGraphSolution)\
        .filter(AngularGraphSolution.error_message != None).order_by(TaskJobs.task_id).all()
    print(f"{len(error_jobs)} found. Will now start resolve these instances...")
    prev_task = None
    for job in tqdm.tqdm(error_jobs, desc="Resolvng error solutions"):
        if prev_task is None or prev_task != job.task:
            config = ConfigHolder(job.task)
            solver = ALL_SOLVER[config.solver](**config.solver_args)
            prev_task = job.task
        sol = solver.solve(job.graph)
        if sol.error_message is None:
            old_sol = job.solution
            session.delete(old_sol)
            job.solution = sol
            session.commit()
        else:
            print(f"Still error message for job {job} with message: {sol.error_message}")
    session.commit()

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url_path", default="angular.db", help="Path to database")
    return parser.parse_args()

if __name__ == "__main__":
    main()