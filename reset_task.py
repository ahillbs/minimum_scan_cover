import argparse
from IPython import embed
from database import (CelestialBody, CelestialGraph, Config, Graph, AngularGraphSolution, Task,
                      TaskJobs, get_session)


def main():
    args = _parse_args()
    session = get_session(args.url_path)
    #configs = session.query(Config).filter(Config.param.like("%solver%")).all()
    tasks = args.tasks
    if not isinstance(tasks, list):
        tasks = [tasks]
    resetted_tasks = []
    for task in tasks:
        resetted_tasks.append(session.query(Task).filter(Task.id == task).one())

    to_delete = []
    for task in resetted_tasks:
        if args.deletejobs:
            task_jobs = task.jobs
            to_delete = to_delete + task_jobs
        task.status = Task.STATUS_OPTIONS.CREATED
        if task.parent is not None:
            task.parent.status = Task.STATUS_OPTIONS.CREATED

    if input(f"{len(to_delete)} jobs will be reset for tasks {tasks} and the tasks will be reset. Sure? y/[n]: ").lower() == "y":
        for job in to_delete:
            if job.solution is not None:
                session.delete(job.solution)
                job.solution = None
        session.commit()

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url-path", default="angular.db", help="Url to database. (Default: angular.db)")
    parser.add_argument("--tasks", nargs="+", help="Task ids to reset")
    parser.add_argument("--deletejobs", action="store_true", help="Also delete jobs instead of just resetting the task status")
    return parser.parse_args()

if __name__ == "__main__":
    main()