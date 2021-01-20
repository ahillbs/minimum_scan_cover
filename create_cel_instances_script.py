import numpy as np
from database import Task, TaskJobs, Config, ConfigHolder, get_session
from instance_generation import create_random_celest_graph
import configargparse


def _load_config():
    parser = configargparse.ArgumentParser(description="Small script to create celestial instances and group them in a task")
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file (default: cel_instances_settings.yaml)',
        default="cel_instances_settings.yaml",
        is_config_file_arg=True)
    parser.add_argument('--min-n', type=int, default=6, help="Minimum amount of vertices (default: 6)")
    parser.add_argument('--max-n', type=int, default=25, help="Maximum amount of vertices (default: 25)")
    parser.add_argument('--steps-n', type=int, default=1, help="Step size between vertices (default: 1)")
    parser.add_argument('--min-m', type=int, default=0, help="Minimum amount of edges, other will be cut (default: 0)")
    parser.add_argument('--max-m', type=int, default=None, help="Maximum amount of edges, higher will be cut (default: None)")
    parser.add_argument('--cel-min', type=float, default=0.1, help="Minimum size of the celestial object (default: 0.1)")
    parser.add_argument('--cel-max', type=float, default=0.8, help="Maximum size of the celestial object (default: 0.8)")
    parser.add_argument('--cel-range', type=float, default=0.0, help="Range in which the celestial object size can range per step (default: 0)")
    parser.add_argument('--cel-step', type=float, default=0.1, help="Size grows per step for the celestial object (default: 0.1)")
    parser.add_argument('--repetitions', type=int, default=5, help="The amount of instances that will be created per vertex amount per step size")
    parser.add_argument('--vertex-shape', type=float, default=[1,1], nargs='*', help="Shape of how the vertices are placed (elliptically). Can also contain a list of sizes which will be repeated. (default: [1,1] = cicle)")
    parser.add_argument('--url-path', type=str, default="angular.db", help="Path to sqlite database")
    parser.add_argument('--name', type=str, default="CelestialGraphInstances", help="Name of the task (default: CelestialGraphInstances)")
    parser.add_argument('--seed', type=int, default=None, help="Seed for current instance creation (default: None; will set a random seed)")
    try:
        #open("cel_instances_settings.yaml")
        return parser.parse_args()
    except SystemExit as system_exit:
        
        parser._remove_action(*[action for action in parser._actions if getattr(action, "is_config_file_arg", False)])
    return parser.parse_args()
    
def main():
    config = _load_config()
    session = get_session(config.url_path)
    graphs = []
    if config.seed is None:
        seed = int(np.random.default_rng(None).integers(np.array([2**63]))[0])
        config.seed = seed
    else:
        seed = int(config.seed)
        
    gen = np.random.default_rng(seed)
    counter = 0
    assert len(config.vertex_shape) % 2 == 0
    shapes = np.array(config.vertex_shape).reshape(round(len(config.vertex_shape)/2), 2)
    l_shapes = len(shapes)
    for n in range(config.min_n, config.max_n, config.steps_n):
        size = config.cel_min
        while size <= config.cel_max:
            for i in range(config.repetitions):
                graph = create_random_celest_graph(n,vertex_shape=shapes[counter % l_shapes], celest_bounds=(size, size+config.cel_range), seed=gen)
                if config.min_m <= graph.edge_amount and (not config.max_m or config.max_m >= graph.edge_amount):
                    graphs.append(graph)
                    counter += 1
            size += config.cel_step
    task = Task(name=config.name, task_type="CelestialGraphInstances", status=Task.STATUS_OPTIONS.FINISHED)
    # Convert the namespace config into database config
    task_config_holder = ConfigHolder.fromNamespace(config, task=task, ignored_attributes=["url_path", "name", "config"])
    task.jobs = [TaskJobs(graph=graph, task=task) for graph in graphs]
    
    
    session.add(task)
    session.commit()
    print(counter, "instances were created. Corresponding task is", task.id)

if __name__ == "__main__":
    main()