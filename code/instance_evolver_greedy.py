import datetime
import math
import tqdm
import configargparse
import numpy as np
from database import Config, ConfigHolder, Graph, Task, TaskJobs, get_session, AngularGraphSolution


def _get_task_and_config(session, arg_config):
    if session is not None:
        task = Task(task_type="instance_evolver_greedy", status=Task.STATUS_OPTIONS.CREATED, name=arg_config.name)
        config = ConfigHolder.fromNamespace(arg_config, task, ignored_attributes=["create_only", "url_path", "name"])
        session.add(task)
        session.commit()
    else:
        task = None
        config = arg_config
    return task, config

def _greedy_evolver_test(arg_config):
    session = get_session(arg_config.url_path) if arg_config.url_path else None
    task, config = _get_task_and_config(session, arg_config)
    from angular_evolver.edge_order_evolver import create_circle_n_k
    for n in range(config.min_n, config.max_n):
        for k in range(config.min_k, min(config.max_k, math.floor(n/2)+1)):
            graph = create_circle_n_k(n, k)
            job = TaskJobs(task=task, graph=graph)
            task.jobs.append(job)
    session.commit()

    if not arg_config.create_only:
        process_task(config, task, session)

def process_task(config, task, session):
    """Processing greedy task jobs

    Args:
        config (ConfigHolder): ConfigHolder for the task
        task (Task): Task for the current jobs
        session (Session): Database session
    """
    try:
        from angular_evolver import edge_order_evolver as OE
        crossover = OE.OrderUniformCrossover()
        selection = OE.uniform_wheel_selection
        mutation = OE.EdgeOrderMutation()
        solver = OE.AngularMinSumGreedySolver()
        result_sol_func = lambda x: np.array([item.solution[1] for item in x])
        fitness = OE.AngularSolverFitness(solver.__class__.__name__,
                                        remote_computation=False,
                                        fitness_goal=OE.AngularSolverFitness.MIN_SUM,
                                        solver_params={"no_sol_return": True},
                                        custom_result_func=result_sol_func)
        termination_iter = OE.IterationTerminationConditionMet(max_iter=config.max_iter)
        termination_no_imp = OE.NoImprovementsTermination(config.max_no_imp)
        termination_comb = OE.TerminationCombination([termination_iter, termination_no_imp])
        callback = OE.update_callback
        task.status = Task.STATUS_OPTIONS.PROCESSING
        if session:
            session.commit()
        to_processed = [job for job in task.jobs if job.solution is None]
        for job in tqdm.tqdm(to_processed, desc="Solving evolve greedy jobs"):

            init_pop = np.zeros(config.pop_size, dtype=object)
            init_pop[:] = [OE.EdgeOrderGraphGenome(job.graph) for i in range(config.pop_size)]

            from genetic_algorithm import GeneticAlgorithm
            ga = GeneticAlgorithm(genomes=init_pop,
                                selection=selection,
                                mutation=mutation,
                                fitness=fitness,
                                crossover=crossover,
                                termCon=termination_comb,
                                callback=callback,
                                elitism=0.05)
            last = ga.evolve()
            arg_max = np.argmax(np.array([genome.solution[1] for genome in last]))
            sol = solver.solve(last[arg_max].graph)
            if session:
                job.solution = sol
                task.last_updated = datetime.datetime.now()
                session.commit()

            if session:
                task.status = Task.STATUS_OPTIONS.FINISHED
                task.last_updated = datetime.datetime.now()
                session.commit()
    except Exception as e:
        if session:
            task.status = Task.STATUS_OPTIONS.ERROR
            task.error_message = str(e)
            session.commit()
        print(e)

def _argument_parser():
    parser = configargparse.ArgumentParser(description="Parser for the greedy worst case instances evolver")
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file',
        is_config_file_arg=True)
    parser.add_argument('--min_k', type=int, default=3, help="Minimum k for all instances (default: 3)")
    parser.add_argument('--max_k', type=int, default=np.iinfo(np.int).max, help="Maximum k for all instances (default: max int)")
    parser.add_argument('--min_n', type=int, default=7, help="Minimum n for all instances (default: 7)")
    parser.add_argument('--max_n', type=int, default=30, help="Maximum n for all instances (default: 30)")
    parser.add_argument('--url_path', type=str, default="angular.db", help="Path to database (default: angular.db)")
    parser.add_argument('--max_iter', type=int, default=200, help="Amount of iterations until completion (default: 200)")
    parser.add_argument('--max_no_imp', type=int, default=40, help="Amount of iterations without improvement before terminating the EA (default: 40)")
    parser.add_argument('--pop_size', type=int, default=200, help="Size of every population (default: 200)")
    parser.add_argument('--name', type=str, default="", help="Describing name for the task")
    parser.add_argument('--create_only', action="store_true", default=False, help="Only create task, do not process it")
    parsed = parser.parse_args()
    return parsed

def main():
    config = _argument_parser()
    _greedy_evolver_test(config)


if __name__ == "__main__":
    main()