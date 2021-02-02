import os
import subprocess
from inspect import isclass

import configargparse
import numpy as np
import sqlalchemy
import yaml
from IPython import embed


from angular_solver import solve
from database import Config, ConfigHolder, Graph, Task, get_session, DatabaseGraphGenome
from genetic_algorithm import (GeneticAlgorithm, Genome,
                               IterationTerminationConditionMet, SaveCallback,
                               k_point_crossover, linear_rank_selection,
                               one_point_crossover, uniform_crossover,
                               uniform_wheel_selection)
from instance_generation import (create_circle, create_circle_n_k,
                                 create_random_circle)
from solver import MscColoringSolver, AngularMinSumGreedySolver
from solver.min_sum_simple_solver import solve_min_sum_simple_n_gon
from solver.mip import (AngularGraphScanMakespanAbsolute,
                        AngularGraphScanMakespanAbsoluteReduced,
                        AngularGraphScanMakespanHamilton,
                        AngularGraphScanMinSumHamilton,
                        AngularDependencySolver,
                        AngularDependencyLocalMinSumSolver,
                        AngularGraphScanLocalMinSumHamilton)
from solver.cp import (ConstraintAbsSolver,
                       ConstraintDependencySolver)
from utils import (Multidict, visualize_graph_2d, visualize_min_sum_sol_2d,
                   visualize_solution_2d)
from angular_evolver import (AngularSolverFitness, CompleteGraphGenome, GraphGenome, GraphGenomeCreator,
                             CompleteGraphGenomeCreator, mutate_2d_points, mutate_vertex_edge_genomes)
from solver import ALL_SOLVER

class GroupedAction(configargparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        group, dest = self.dest.split('.', 2)
        groupspace = getattr(namespace, group, configargparse.Namespace())
        setattr(groupspace, dest, values)
        setattr(namespace, group, groupspace)

def string_to_callable(function_name):
    assert function_name != 'eval', "Eval is not allowed!"
    warning_displayed_once = getattr(StringToCallableAction, "warning_displayed", False)
    if not warning_displayed_once:
        print("WARNING: Do not use StringToCallableAction in production code! This is just a hack for faster development!")
        setattr(StringToCallableAction, "warning_displayed", True)
    try:
        call = ALL_SOLVER[function_name]
    except KeyError:
        call = globals()[function_name]
    return call

class StringToCallableAction(configargparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        
        warning_displayed_once = getattr(StringToCallableAction, "warning_displayed", False)
        if not warning_displayed_once:
            print("WARNING: Do not use StringToCallableAction in production code! This is just a hack for faster development!")
            setattr(StringToCallableAction, "warning_displayed", True)
        call = globals()[values]
        if callable(call):
            setattr(namespace, self.dest, call)
        else:
            raise TypeError(f"{values} is not callable")

def _instantiate_callables(func_name, obj_args):
    callable_obj = string_to_callable(func_name)
    if not callable_obj:
        raise AttributeError(f"{func_name} function is not set.".capitalize())
    if not isclass(callable_obj):
        return callable_obj
    if not obj_args:
        obj_args = {}
    return callable_obj(**obj_args)

def _get_task_and_config(session, arg_config):
    task = None
    config = None
    if arg_config.url_path:
        if hasattr(arg_config, "task") and arg_config.task is not None:
            task = session.query(Task).filter(Task.id == arg_config.task).one()
            if arg_config.override_config and \
               input(f"Are you sure to override the configs for {task.id}? (y/N)").lower() in ["y", "yes"]:
                print(f"Override config from task {task.id})")
                for task_config in task.configs:
                    session.delete(task_config)
                arg_config.override_config = False
                config = ConfigHolder.fromNamespace(arg_config, task, ["override_config", "url_path", "PreEvolveInteractive", "create_only"])
                session.add(config)
                session.commit()
            else:
                print("Using config from database")
                config = ConfigHolder(task)
        else:
            if input("New Task will be created (Y/n)?").lower() in ["", "yes", "y"]:
                print("Will create a new task.")
                task = Task(task_type="instance_evolver", status=Task.STATUS_OPTIONS.CREATED, name=arg_config.name)
                session.add(task)
                session.commit()
                arg_config.task = task.id
                config = ConfigHolder.fromNamespace(arg_config, task, ignored_attributes=["url_path", "create_only", "name", "override_config"])
                session.add_all(config.database_configs)
                session.commit()
                savepath = input(f"Task ID is {task.id}. Type a filepath to save the ID in a config file (default: Skip save): ")
                if savepath:
                    _save_task_file(savepath, config, task)
            else:
                config = arg_config
        return task, config

def _save_task_file(savepath, config, task):
    n_s = configargparse.Namespace()
    n_s.task = task.id
    parser = configargparse.Parser()
    parser.add_argument("--task")
    parser.add_argument("--database")
    parsed = parser.parse_args(args=[f"--task={task.id}", f"--database={config.url_path}"])
    parser.write_config_file(n_s, [savepath])

def _evolve_instances(arg_config):
    session = get_session(arg_config.url_path)
    task, config = _get_task_and_config(session, arg_config)

    if not arg_config.create_only:
        process_task(config, task, session)

def process_task(config, task, session):
    # First init all callable classes
    try:
        mutation = _instantiate_callables(config.mutation_func, None)
        selection = _instantiate_callables(config.selection_func, None)
        crossover = _instantiate_callables(config.crossover_func, None)
        fitness = _instantiate_callables(config.fitness_func, config.fitness_func_initargs)
        if config.term_condition == 'IterationTerminationConditionMet' and not config.term_condition_initargs:
            term_con = IterationTerminationConditionMet(max_iter=config.generations)
        else:
            term_con = _instantiate_callables(config.term_condition, config.term_condition_initargs)
        if config.callback == 'SaveCallback' and config.callback_initargs is None:
            callback = SaveCallback(config.generations, config.population_amount, task, session)
        else:
            callback = _instantiate_callables(config.callback, config.callback_initargs)
        task.status = Task.STATUS_OPTIONS.PROCESSING
        if session:
            session.commit()
        # Now load population if provided, else generate it
        starting_generation, population = _load_population(config, task, session)

        if config.PreEvolveInteractive:
            print("Config set up. To change the population just change the 'population' variable.")
            print("For other variables just refer to the locals.")
            embed()

        gen_algo = GeneticAlgorithm(
            genomes=population,
            selection=selection,
            mutation=mutation,
            fitness=fitness,
            crossover=crossover,
            callback=callback,
            termCon=term_con,
            elitism=config.elitism,
            mutationChance=config.mutation_chance_genome,
            mutationChanceGene=config.mutation_chance_gene
        )
        gen_algo.evolve(generation=starting_generation)
        task.status = Task.STATUS_OPTIONS.FINISHED
        if session:
            session.commit()
    except InterruptedError as e:
        task.status = task.STATUS_OPTIONS.INTERRUPTED
        if session:
            session.commit()
    except Exception as e:
        if session:
            task.status = Task.STATUS_OPTIONS.ERROR
            task.error_message = str(e)
            session.commit()
        print(e)
        raise e

def _load_population(config, task, session: 'Session'):
    population = []
    curr_generation = 0
    if session is not None:
        try:
            last_gen = session.query(DatabaseGraphGenome)\
                .filter(DatabaseGraphGenome.task_id == task.id)\
                .order_by(DatabaseGraphGenome.generation.desc())\
                .limit(1)\
                .one()
            curr_generation = last_gen.generation
            queue = session.query(DatabaseGraphGenome)\
                .filter(DatabaseGraphGenome.task_id == task.id, DatabaseGraphGenome.generation == curr_generation)\
                .order_by(DatabaseGraphGenome.generation.desc())\
                .limit(config.population_amount)
            population = np.zeros(config.population_amount, dtype=object)
            population[:] = [genome for genome in queue]
            assert isinstance(population[0], Genome), "Loaded data does not contain valid genomes"
        except sqlalchemy.orm.exc.NoResultFound as e:
            pass

    if len(population) < config.population_amount:
        if population:
            print("Given population smaller than wanted. Fill with random instances")
        temp_pop = np.zeros(config.population_amount - len(population), dtype=object)
        create_instances = _instantiate_callables(config.instance_creation_func, config.instance_creation_initargs)
        temp_pop[:] = [
                        create_instances(task, generation=curr_generation)
                        for i in range(config.population_amount - len(population))
                    ]
        session.add_all(temp_pop.tolist())
        session.commit()
        population = np.hstack([population[:len(population)],
                                temp_pop]) # ToDo: This call needs to be reworked
    elif len(population) > config.population_amount:
        print("Given population too large. Will slice off the end")
        population = population[:config.population_amount]

    return curr_generation, population


def _argument_parser():
    parser = configargparse.ArgumentParser(description="Parser for the instance evolver")
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file (default: inst_evo_settings.yaml)',
        default="inst_evo_settings.yaml",
        is_config_file_arg=True)
    parser.add_argument(
        '--PreEvolveInteractive',
        action='store_true',
        help='Ipython interactive for instance creation (default: False)',
        default=False)
    parser.add_argument('--override-config', action="store_true", default=False, help="Set this flag to override configuration with passed arguments")
    parser.add_argument('--url-path', type=str, default="angular.db", help="Path to database. Creates Database if it does not exist (Default: angular.db)")
    parser.add_argument('--task', type=int, help="Id of the task that shall be continued")
    parser.add_argument('--generations', type=int, default=200, help="Amount of generations evolved. If a save is loaded, it will only evolve the difference for the generations (default: 200)")
    parser.add_argument('--elitism', type=float, default=0.01, help="Elitism rate (default: 0.01)")
    #parser.add_argument('--genome-creator',
    parser.add_argument('--instance-creation-func', type=str, help="Function for initial creation of instances")
    parser.add_argument('--instance-creation-initargs', type=yaml.safe_load, help="Parameter for instance creation")
    parser.add_argument('--population-amount', type=int, default=200, help="Amont of genomes per generation (default: 200)")
    parser.add_argument('--mutation-chance-genome', type=float, default=0.03, help="Chance a genome will be selected for mutation (default: 0.03)")
    parser.add_argument('--mutation-chance-gene', type=float, default=0.03, help="Chance a gene is changed (default: 0.03)")
    parser.add_argument('--mutation-func', type=str, help="Mutation callable used. Required if no safefile config is used")
    parser.add_argument('--selection-func', type=str, help="Selection callable used. Required if no safefile is used")
    parser.add_argument('--crossover-func', type=str, help="Crossover callable used. Required if no safefile is used")
    parser.add_argument('--fitness-func', type=str, help="Fitness callable used. Required if no safefile is used")
    parser.add_argument('--fitness-func-initargs', type=yaml.safe_load, default=None, help="Fitness callable init keyword arguments. Omitted when emtpy.")
    parser.add_argument('--term-condition', type=str, default='IterationTerminationConditionMet', help="Termination callable used. (default: IterationTerminationConditionMet)")
    parser.add_argument('--term-condition-initargs', type=yaml.safe_load, default=None, help="Keyword arguments dict for termination condition callable init. Not needed for standard term-condition.")
    parser.add_argument('--callback', type=str, default='SaveCallback', help="Callback used in genetic_algorithm (default: SaveCallback)")
    parser.add_argument('--callback-initargs', type=yaml.safe_load, default=None, help="Callback keyword arguments dict for init. Not needed for standard SaveCallback else omitted if not provided")
    parser.add_argument('--create-only', action="store_true", help="Only create task instead of also solving it")
    parser.add_argument('--name', type=str, default="", help="Optional name description of the task")
    parsed = parser.parse_args()
    #parser.write_config_file()
    #print(vars(parsed))
    return parsed

if __name__ == "__main__":
    CONFIG = _argument_parser()
    _evolve_instances(CONFIG)
