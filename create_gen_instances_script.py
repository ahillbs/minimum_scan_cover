import math
import numpy as np
from database import Task, TaskJobs, Config, ConfigHolder, get_session, Graph
from instance_generation import create_random_instance, create_random_instance_fixed_edges
import configargparse


def _load_config():
    parser = configargparse.ArgumentParser(description="Small script to create celestial instances and group them in a task")
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file (default: cel_instances_settings.yaml)',
        default="gen_instances_settings.yaml",
        is_config_file_arg=True)
    parser.add_argument('--min-n', type=int, default=6, help="Minimum amount of vertices (default: 6)")
    parser.add_argument('--max-n', type=int, default=25, help="Maximum amount of vertices (default: 25)")
    parser.add_argument('--steps-n', type=int, default=1, help="Step size between vertices (default: 1)")
    parser.add_argument('--min-m', type=int, default=0, help="Minimum amount of edges, other will be cut (default: 0)")
    parser.add_argument('--max-m', type=int, default=None, help="Maximum amount of edges, higher will be cut (default: None)")
    parser.add_argument('--edge-min', type=float, default=35, help="Minimum chance/amount an edge will be added (default: 0.1)") #0.1
    parser.add_argument('--edge-max', type=float, default=80, help="Maximum chance/amount an edge will be added (default: 0.8)") # 0.8
    parser.add_argument('--edge-step', type=float, default=0.1, help="Chance increase for edge added per step (default: 0.1)")
    parser.add_argument('--repetitions', type=int, default=3, help="The amount of instances that will be created per vertex amount per chance step")
    parser.add_argument('--fixed-edges', action="store_true", default=False, help="Use fixed edges or edge probability. If True, you need to set edge-min and edge-max")
    parser.add_argument('--max-amount', type=int, default=50, help="Maximum amount of instances when using fixed edges (Default: 50)")
    parser.add_argument('--url-path', type=str, default="angular.db", help="Path to sqlite database")
    parser.add_argument('--name', type=str, default="CelestialGraphInstances", help="Name of the task (default: CelestialGraphInstances)")
    parser.add_argument('--seed', type=int, default=None, help="Seed for current instance creation (default: None; will set a random seed)")
    try:
        
        return parser.parse_args()
    except SystemExit as system_exit:
        
        parser._remove_action(*[action for action in parser._actions if getattr(action, "is_config_file_arg", False)])
    return parser.parse_args()
    
def main():
    config = _load_config()
    session = get_session(config.url_path)
    
    if config.seed is None:
        seed = int(np.random.default_rng(None).integers(np.array([2**63]))[0])
        config.seed = seed
    else:
        seed = int(config.seed)
        
    gen = np.random.default_rng(seed)
    if config.fixed_edges:
        graphs = _fixed_generation(config, gen)
    else:
        graphs = _probability_generation(config, gen)
    task = Task(name=config.name, task_type="GeometricGraphInstances", status=Task.STATUS_OPTIONS.FINISHED)
    # Convert the namespace config into database config
    task_config_holder = ConfigHolder.fromNamespace(config, task=task, ignored_attributes=["url_path", "name", "config"])
    task.jobs = [TaskJobs(graph=graph, task=task) for graph in graphs]
    
    
    session.add(task)
    session.commit()
    print(len(task.jobs), "instances were created. Corresponding task is", task.id)

def _fixed_generation(config, gen):
    graphs = []
    used_n = []
    count = 0
    for n in range(config.min_n, config.max_n):
        if n*(n-1)/2 <= config.min_n:
            print("n={n} is too small. Will omit it for generating instances.")
        else:
            used_n.append(n)
    
    min_edges = int(config.edge_min)
    max_edges = int(config.edge_max)
    diff = max_edges - min_edges
    remaining = int(config.max_amount)
    assert config.max_n * (config.max_n -1) >= 2*max_edges, "Max n is too small! Please choose n such that the maximum amount of edges can be created"
    while (remaining > 0):
        chosen = gen.choice(range(min_edges, max_edges+1), size=min([diff, remaining]), replace=False)
        for m in chosen:
            # Calculate eligable n
            i = config.min_n
            while (i*(i-1) / 2 < m):
                i += 1
            n_range = range(i, config.max_n+1)
            chosen_n = gen.choice(n_range)
            graph = create_random_instance_fixed_edges(chosen_n, m)
            graphs.append(graph)
        remaining -= len(chosen)
    return graphs



def _probability_generation(config, gen):
    
    graphs = []
    for n in range(config.min_n, config.max_n, config.steps_n):
        chance = config.edge_min
        while chance <= config.edge_max:
            for i in range(config.repetitions):
                graph = create_random_instance(n, edge_chance=chance, seed=gen)
                if(config.min_m <= graph.edge_amount <= config.max_m):
                    graphs.append(graph)
            chance += config.edge_step
    return graphs

if __name__ == "__main__":
    main()