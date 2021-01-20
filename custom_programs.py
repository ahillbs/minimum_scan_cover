from instance_generation.general_instances import create_random_instance
from genetic_algorithm.callbacks import update_callback
from database.angular_graph_solution import AngularGraphSolution
from genetic_algorithm.genetic_algorithm_class import GeneticAlgorithm
from pandas.core.frame import DataFrame
import tqdm
import numpy as np
import pandas as pd
import math
import random
from database import Graph, CelestialBody, CelestialGraph

from solver import MscColoringSolver, ALL_SOLVER
from instance_generation import create_circle, create_random_circle, create_circle_n_k, create_random_celest_graph
from solver.min_sum_simple_solver import solve_min_sum_simple_n_gon
from utils import visualize_solution_2d, visualize_graph_2d, visualize_min_sum_sol_2d, Multidict, callback_rerouter, get_lower_bounds, calculate_times
from angular_solver import solve
from solver.min_sum_simple_solver import solve_min_sum_simple_n_gon

def greedy_evolver_test():
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
    termination = OE.IterationTerminationConditionMet(max_iter=200)
    callback = OE.update_callback
    graph = OE.create_circle_n_k(12, 3)
    init_pop = np.zeros(200, dtype=object)
    init_pop[:] = [OE.EdgeOrderGraphGenome(graph) for i in range(200)]

    from genetic_algorithm import GeneticAlgorithm
    ga = GeneticAlgorithm(genomes=init_pop,
                          selection=selection,
                          mutation=mutation,
                          fitness=fitness,
                          crossover=crossover,
                          termCon=termination,
                          callback=callback)
    last = ga.evolve()
    return last
def get_sin(n,k):
    return np.sin((k * 2 * np.pi) / n)
def get_cos(n,k):
    return np.cos((k * 2 * np.pi) / n)
def get_distance(n,k):
    return get_sin(n,k) / (get_sin(n,k)**2 + (get_cos(n,k)-1)**2)
def create_graph():
    d1 = get_distance(9,3)
    d2 = get_distance(9,2)
    d = d1 + np.random.default_rng().random() * (d2-d1)
    body = CelestialBody((0,0), d, None)
    g = create_circle_n_k(9,4)
    graph = CelestialGraph([body], g.vertices, g.edges)
    visualize_graph_2d(graph)
    for i in range(10):
        for j in range(3):
            rand_c = create_random_celest_graph(9+i,celest_bounds=(0.4,0.6))
            rand_g = create_random_instance(9+i, edge_chance=random.uniform(0.3, 1))
            visualize_graph_2d(rand_g, savePath=f"instance_vis/rand_e{9+i}_{j}.pdf")
            visualize_graph_2d(rand_c, savePath=f"instance_vis/cel_e{9+i}_{j}.pdf")

def _use_test():
    from tests.test_solvers import test_cp_lcm_gcd
    test_cp_lcm_gcd()

def _test_dep_mip_with_subtours():
    from solver.mip import AngularDependencySolver
    from instance_generation import create_circle_n_k
    solver_w = AngularDependencySolver(with_vertex_subtour_constr=True)
    solver_o = AngularDependencySolver()
    for n in range(5, 10):
        g = create_circle_n_k(n, 3)
        print("#########################################################")
        print("################ SOLVER WITH SUBTOURS ###################")
        print("#########################################################")
        s_w = solver_w.solve(g)
        print("#########################################################")
        print("################ SOLVER WITHOUT SUBTOURS ################")
        print("#########################################################")
        s_o = solver_o.solve(g)

        print("#########################################################")
        print("################ RUNTIME WITH:",s_w.runtime,"WITHOUT:", s_o.runtime,"###################")
        print("#########################################################")
        if s_w.is_optimal and s_o.is_optimal:
            assert math.isclose(s_w.min_sum, s_o.min_sum)

def main():
    #solution_example_intro()
    #color_solver_check()
    #correct_msc_instances()
    #visualize_solutions_overview()
    #visualize_geo_solutions_overview()
    #visualize_solutions_overview_second_batch()
    #bnb_sols()
    #vis_full_circle_solutions()
    create_graph()

    #test_makespan_bipartite()
    #general_instances_test()
    #inspect_ga_performance()
    #_use_test()
    
    return

    graphs = [create_random_celest_graph(i, celest_bounds=(j*0.1+0.1, j*0.1+0.1), seed=None) for i in range(5, 8) for j in range(5)]

class DataHolder:
    def __init__(self, dataFrame: DataFrame):
        self.dataFrame = dataFrame
        self.graph = None
        self.solution_type = None
        self.solver = None
    
    def __call__(self, gen_algo: GeneticAlgorithm):
        data = []
        for genome in gen_algo.genomes:
            arg_sorted = np.argsort(genome.order_numbers)
            order = self.graph.edges[arg_sorted].tolist()
            times, head_sum = calculate_times(self.graph.edges[arg_sorted], angles=self.graph.costs, return_angle_sum=True)
            min_sum = sum(head_sum)
            local_min_sum = max(head_sum)
            sol = AngularGraphSolution(
                self.graph,
                0,
                gen_algo.solver,
                gen_algo.solution_type,
                is_optimal=False,
                order=order,
                error_message=None
            )            
            data.append({
                "Graph": self.graph.name,
                "Generation": gen_algo.generation,
                "solution_type": sol.solution_type,
                "solver": sol.solver,
                "min_sum": sol.min_sum,
                "local_min_sum": sol.local_min_sum,
                "makespan": sol.makespan
            })
            assert math.isclose(genome.solution, -sol.local_min_sum) or math.isclose(genome.solution, -sol.min_sum)
        self.dataFrame = self.dataFrame.append(DataFrame(data), ignore_index=True)
        update_callback(gen_algo)

def inspect_ga_performance():
    from solver.meta_heur import AngularGeneticMinSumSolver, AngularGeneticLocalMinSumSolver
    from instance_generation import create_random_celest_graph
    dh = DataHolder(DataFrame())
    graphs = []
    ga_lms = AngularGeneticLocalMinSumSolver(cb=dh)
    ga_ms = AngularGeneticMinSumSolver(cb=dh)
    for n in range(15, 25):
        for m in range(5):
            graph = create_random_celest_graph(n)
            graphs.append(graph)
            dh.graph = graph

            ga_lms.solve(graph)
            ga_ms.solve(graph)

    df_graphs = DataFrame([{
        "graph_name": g.name ,
        "vert_amount": g.vert_amount,
        "edge_amount": g.edge_amount
    } for g in graphs])
    
    print(dh.dataFrame)
    dh.dataFrame.to_pickle("GA_comparison.pk")
    df_graphs.to_pickle("GA_comparison_graphs.pk")


def general_instances_test():
    from solver.x_split_solver import SplitXSolver
    from solver import MscColoringSolver
    from create_gen_instances_script import create_random_instance
    from database import TaskJobs

    solver_x = SplitXSolver()
    solver_c = MscColoringSolver()
    jobs = []
    for i in tqdm.trange(7, 20, desc="Vertices"):
        for j in tqdm.trange(3,10, desc="Edge probability"):
            for k in tqdm.trange(3, desc="Repetition"):
                g = create_random_instance(i, edge_chance=0.1*j)
                g.id = i*100+j*10+k
                sol_x = solver_x.solve(g)
                sol_c = solver_c.solve(g)
                try:
                    assert sol_x.makespan < max(get_lower_bounds(g)) * 5.5 * math.ceil(math.log2(g.vert_amount))
                    assert sol_c.makespan < max(get_lower_bounds(g)) * 5.5 * math.ceil(math.log2(g.vert_amount))
                except AssertionError as e:
                    print("Solution x-split:", sol_x.makespan, max(get_lower_bounds(g)) * 5.5 * math.ceil(math.log2(g.vert_amount)), "time:", sol_x.runtime)
                    print("Solution color:", sol_c.makespan, max(get_lower_bounds(g)) * 5.5 * math.ceil(math.log2(g.vert_amount)), "time:", sol_x.runtime)
                    raise e
                jobs.append(TaskJobs(task=None, solution=sol_x, graph=g))
                jobs.append(TaskJobs(task=None, solution=sol_c, graph=g))
    from utils.visualization import visualize_solution_scatter, VisTypes
    visualize_solution_scatter(jobs, "Bla", vis_type=VisTypes.LB_Runtime, logscale=True, loc=9)
    
test_graphs = [
    Graph(V=np.array([[345,  53],
       [482,  51],
       [333, 455],
       [425, 128],
       [417, 316],
       [417, 382],
       [ 65, 101],
       [235, 207],
       [ 96, 229],
       [470,  60],
       [ 74, 253],
       [ 76, 478],
       [491, 473],
       [166, 176]]),
       ad_matrix=np.array([[0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.],
       [0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
       [1., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])
    ),
    Graph(
        V=np.array([[390, 428],
       [393, 220],
       [197,  19],
       [326, 465],
       [256, 383],
       [365, 430],
       [ 77, 138],
       [  3, 323],
       [338,  11],
       [275, 187]]),
       ad_matrix=np.array([[0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
       [1., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 1., 0., 1., 0., 0., 0., 1., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 1., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]])
       ),
    Graph(
        V=np.array([[357, 339],
    [394, 133],
    [437,  12],
    [283, 373],
    [421, 365],
    [283, 289],
    [283, 135],
    [335, 471],
    [ 43, 162],
    [375, 183],
    [109, 147],
    [486, 303]]),
    ad_matrix=np.array([
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
    [1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.],
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
    ])
    ),
    Graph(V=np.array([[158,  19],
       [335,  22],
       [ 27, 395],
       [ 36, 500],
       [367, 153],
       [349, 418],
       [  4, 314],
       [115, 430],
       [ 75,  28],
       [ 24, 399],
       [371,  63],
       [106, 261],
       [ 66,  38],
       [102, 114],
       [138, 318]]),
       ad_matrix=np.array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
       [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0.],
       [0., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1.],
       [0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 0., 1., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
       [0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0.]])
       ),
       Graph(V=np.array([[152, 467],
       [160, 434],
       [ 14, 224],
       [101, 453],
       [ 35, 381],
       [ 47, 220],
       [346,  46],
       [268, 366],
       [440, 111],
       [296, 416],
       [123,  78]]),
       ad_matrix=np.array([[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
       [1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [1., 1., 0., 1., 1., 1., 0., 1., 0., 0., 0.]])),
       Graph(V=np.array([[298, 285],
       [309,  31],
       [352,  84],
       [153, 480],
       [163, 285],
       [261, 408],
       [271, 441],
       [277, 141]]),
       ad_matrix=np.array([[0., 1., 0., 0., 0., 0., 0., 1.],
       [1., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 1., 1., 0., 0., 0.],
       [0., 1., 1., 0., 0., 1., 0., 1.],
       [0., 0., 1., 0., 0., 0., 0., 1.],
       [0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 1., 1., 0., 0., 0.]])),
       Graph(V=np.array([[ 29, 247],
       [252, 388],
       [171, 402],
       [ 85, 418],
       [ 96, 349],
       [382, 309],
       [112, 430],
       [174, 117]]),
       ad_matrix=np.array([[0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 1., 1., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 1., 1., 0., 0., 1., 0., 0.],
       [0., 1., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 1., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]])),
       Graph(V=np.array([[375, 112],
       [209, 302],
       [356, 155],
       [371,  64],
       [312,  76],
       [387, 117],
       [ 25,  33],
       [414, 371],
       [110, 449],
       [160, 464],
       [231, 274],
       [ 34, 224],
       [ 42, 437],
       [198,   8],
       [160, 348],
       [ 68, 218],
       [350,  47],
       [282, 206],
       [246, 256],
       [303, 443],
       [483,  19]]),
       ad_matrix=np.array([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
        0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1.,
        0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0.,
        0., 0., 0., 0., 0.],
       [1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
        1., 0., 0., 1., 1.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0.,
        0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 1.],
       [0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
        0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 1.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        1., 0., 0., 0., 0.],
       [0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 0.],
       [0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
        0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0.]]))
]
test_associations = [
    np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]),
    np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1]),
    np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]),
    np.array([1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1]),
    np.array([1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0]),
    np.array([0, 1, 1, 0, 0, 1, 1, 1]),
    np.array([0, 0, 0, 1, 1, 0, 1, 0]),
    np.array([1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0])
    ]
def test_makespan_bipartite():
    from instance_generation import create_random_bipartite
    from utils import get_lower_bounds
    from solver.bipartite_solver import AngularBipartiteMakespanSolver    
    
    
    
    solver = AngularBipartiteMakespanSolver()
    for g,a in zip(test_graphs, test_associations):
        sol = solver.solve(g, colors=a, debug=True)
        sol2 = solver.solve(g, colors=a, strategy=1, debug=True)
        lb = max(get_lower_bounds(g))
        print("Sol:", sol.makespan, "Sol2:", sol2.makespan, "bound:", lb*4.5)
        
        assert sol.makespan <= lb*4.5
    for i in range(7, 30):
        for j in range(50):
            print("Build graph")
            g, association = create_random_bipartite(i, max_cone=89)
            g2, association2 = create_random_bipartite(i)
            lb = max(get_lower_bounds(g))
            lb2 = max(get_lower_bounds(g2))
            assert lb < 90
            print("Graph creation complete. Start solver")
            sol = solver.solve(g, colors=association, debug=True)
            print("Sol:", sol.makespan, "bound:", lb*4.5)
            assert sol.makespan <= lb*4.5
            
    

    
def correct_msc_instances():
    from database import Task, TaskJobs, Config, Graph, get_session, CelestialGraph, AngularGraphSolution
    from solver import MscColoringSolver
    session = get_session("angular.db")
    color_tasks = session.query(Task).filter(Task.name.like("%Color%")).all()
    msc_color_tasks = [t for t in color_tasks if session.query(Config).filter(Config.task_id == t.id, Config.param == "solver_args", Config._value_str == "{}").count() > 0]
    error_tasks = [j for t in tqdm.tqdm(color_tasks) for j in tqdm.tqdm(t.jobs) if j.solution == None or j.solution.error_message != None]
    solver = MscColoringSolver()
    for j in tqdm.tqdm(error_tasks, desc="Process error msc color instances"):
        sol = solver.solve(j.graph)
        j.solution = sol
        session.commit()

def visualize_geo_solutions_overview():
    from database import Task, TaskJobs, Config, Graph, get_session, CelestialGraph, AngularGraphSolution
    from utils.visualization import visualize_solution_scatter, VisTypes
    session = get_session("angular.db")
    min_sum_tasks = session.query(Task).filter(Task.parent_id.in_([80,107])).all()
    local_min_sum_tasks = session.query(Task).filter(Task.parent_id.in_([89,109])).all() + session.query(Task).filter(Task.id == 82).all()
    makespan_tasks = session.query(Task).filter(Task.parent_id.in_([97,111])).all()
    min_sum_jobs = session.query(TaskJobs).filter(TaskJobs.task_id.in_([t.id for t in min_sum_tasks])).all()
    local_min_sum_jobs = session.query(TaskJobs).filter(TaskJobs.task_id.in_([t.id for t in local_min_sum_tasks])).all()
    makespan_jobs = session.query(TaskJobs).filter(TaskJobs.task_id.in_([t.id for t in makespan_tasks])).all()
    #visualize_solution_scatter(min_sum_jobs, "Bla", vis_type=VisTypes.All, logscale=True, loc=9)
    visualize_solution_scatter(local_min_sum_jobs, "Bla", vis_type=VisTypes.All, logscale=True, loc=9)
    #visualize_solution_scatter(makespan_jobs, "Bla", vis_type=VisTypes.All, logscale=True, loc=9)

def visualize_solutions_overview():
    from database import Task, TaskJobs, Config, Graph, get_session, CelestialGraph, AngularGraphSolution
    from instance_evolver import GraphGenome
    from utils import visualize_graph_2d, visualize_min_sum_sol_2d, visualize_solution_2d
    from matplotlib import pyplot as plt
    session = get_session("angular.db")
    from utils.visualization import visualize_solution_scatter, VisTypes
    min_sum_tasks = session.query(Task).filter(Task.parent_id == 2).all()
    local_min_sum_tasks = session.query(Task).filter(Task.parent_id == 11).all() 
    makespan_tasks = session.query(Task).filter(Task.parent_id == 19).all()
    min_sum_jobs = session.query(TaskJobs).filter(TaskJobs.task_id.in_([t.id for t in min_sum_tasks])).all()
    local_min_sum_jobs = session.query(TaskJobs).filter(TaskJobs.task_id.in_([t.id for t in local_min_sum_tasks])).all()
    makespan_jobs = session.query(TaskJobs).filter(TaskJobs.task_id.in_([t.id for t in makespan_tasks])).all()
    visualize_solution_scatter(min_sum_jobs, "Bla", vis_type=VisTypes.LB_Runtime, logscale=True, loc=9)
    visualize_solution_scatter(local_min_sum_jobs, "Bla", vis_type=VisTypes.LB_Runtime, logscale=True, loc=9)
    visualize_solution_scatter(makespan_jobs, "Bla", vis_type=VisTypes.LB_Runtime, logscale=True, loc=9)

def visualize_solutions_overview_second_batch():
    from database import Task, TaskJobs, Config, Graph, get_session, CelestialGraph, AngularGraphSolution
    from instance_evolver import GraphGenome
    from utils import visualize_graph_2d, visualize_min_sum_sol_2d, visualize_solution_2d
    from matplotlib import pyplot as plt
    session = get_session("angular.db")
    from utils.visualization import visualize_solution_scatter, VisTypes
    min_sum_tasks = session.query(Task).filter(Task.parent_id == 45).all()
    local_min_sum_tasks = session.query(Task).filter(Task.parent_id == 40).all()
    makespan_tasks = session.query(Task).filter(Task.parent_id == 30, ~Task.name.like("%Reduced%")).all()
    min_sum_jobs = session.query(TaskJobs).filter(TaskJobs.task_id.in_([t.id for t in min_sum_tasks])).all()
    local_min_sum_jobs = session.query(TaskJobs).filter(TaskJobs.task_id.in_([t.id for t in local_min_sum_tasks])).all()
    makespan_jobs = session.query(TaskJobs).filter(TaskJobs.task_id.in_([t.id for t in makespan_tasks])).all()
    visualize_solution_scatter(min_sum_jobs, "Bla", vis_type=VisTypes.LB_Runtime, logscale=True, loc=9)
    visualize_solution_scatter(local_min_sum_jobs, "Bla", vis_type=VisTypes.LB_Runtime, logscale=True, loc=9)
    visualize_solution_scatter(makespan_jobs, "Bla", vis_type=VisTypes.LB_Runtime, logscale=True, loc=9)
    print("finished")

def color_solver_check():
    from database import Task, TaskJobs, Config, ConfigHolder, Graph, get_session, CelestialGraph, AngularGraphSolution
    session = get_session("angular.db")
    
    color_configs = session.query(Config).filter(Config._value_str == '"MscColoringSolver"').all()
    tasks = [c.task for c in color_configs if c.task != None]
    bad_solution_jobs = [j for t in tqdm.tqdm(tasks) for j in t.jobs if j.solution != None and len(j.solution.order) != j.graph.edge_amount]
    bad_tasks = {j.task:[] for j in bad_solution_jobs}
    for job in bad_solution_jobs:
        bad_tasks[job.task].append(job)
    
    for task in tqdm.tqdm(bad_tasks, desc="Processing bad tasks"):
        holder = ConfigHolder(task)
        solver = ALL_SOLVER[holder.solver](**holder.solver_args)
        for job in tqdm.tqdm(bad_tasks[task], desc="Processing bad jobs"):
            sol = solver.solve(job.graph)
            job.solution = sol
            session.commit()
    
def vis_full_circle_solutions():
    from instance_generation import create_circle_n_k
    graphs = [create_circle_n_k(n,n) for n in range(4, 9)]
    from solver.cp import ConstraintDependencySolver, ConstraintDependencyLocalMinSumSolver
    from solver.mip import AngularDependencySolver, AngularDependencyLocalMinSumSolver
    sols_MSSC = []
    sols_MLSSC = []

    import pickle
    try:
        with open("Circle_sols.pk", "rb") as f:
            sols_MSSC, sols_MLSSC = pickle.load(f)
    except (EOFError, FileNotFoundError):
        solver_MSSC = AngularDependencySolver()
        solver_MLSSC = AngularDependencyLocalMinSumSolver()
        for g in tqdm.tqdm(graphs):
            sols_MSSC.append(solver_MSSC.solve(g))
            sols_MLSSC.append(solver_MLSSC.solve(g))

        with open("Circle_sols.pk", "wb") as f:
            pickle.dump((sols_MSSC, sols_MLSSC), f)
    from utils import visualize_min_sum_sol_2d
    print("Min Sum Sols:")
    for s in sols_MSSC:
        visualize_min_sum_sol_2d(s)
    return
    
    

def bnb_sols():
    from database import Task, TaskJobs, Config, Graph, get_session, CelestialGraph, AngularGraphSolution
    from solver.bnb import MinSumAbstractGraphSolver, MinSumOrderSolver
    from create_gen_instances_script import create_random_instance
    import tqdm
    import pickle
    sols = None
    try:
        with open("bnb_sols.pk", "rb") as f:
            bnb1sols, bnb2sols = pickle.load(f)
    except (EOFError, FileNotFoundError):
        pass
    bnb1 = MinSumAbstractGraphSolver(time_limit=900)
    bnb2 = MinSumOrderSolver(time_limit=900)
    if not bnb1sols or not bnb2sols:
        graphs = [create_random_instance(n, edge_chance=e) for e in np.arange(0.5,1, 0.1) for n in range(5,8) for i in range(2)]    
        bnb1 = MinSumAbstractGraphSolver(time_limit=900)
        bnb2 = MinSumOrderSolver(time_limit=900)
        bnb1sols = []
        bnb2sols = []
        for g in tqdm.tqdm(graphs):
            bnb1sols.append(bnb1.solve(g))
            bnb2sols.append(bnb2.solve(g))
        sols = bnb1sols + bnb2sols
    
    for i,(s1, s2) in enumerate(zip(bnb1sols, bnb2sols)):
        s1.graph.id = i
        s1.graph_id = 1
        s2.graph.id = i
        s2.graph_id = 1
        

    with open("bnb_sols.pk", "wb") as f:
        pickle.dump((bnb1sols, bnb2sols), f)
    from utils.visualization import visualize_solution_scatter, VisTypes
    
    
    
    visualize_solution_scatter(bnb1sols+bnb2sols, "Branch and Bound runtimes", solution_type="runtime", vis_type=VisTypes.Absolute)
    visualize_solution_scatter(bnb1sols+bnb2sols, "Branch and Bound Vs Lower Bound", solution_type="min_sum", vis_type=VisTypes.VsLB)

def visualize_makespan_evolve_data():
    from database import Task, TaskJobs, Config, Graph, get_session, CelestialGraph, AngularGraphSolution
    from instance_evolver import GraphGenome
    from utils import visualize_graph_2d, visualize_min_sum_sol_2d, visualize_solution_2d
    from matplotlib import pyplot as plt
    session = get_session("angular_old.db")
    with session.no_autoflush:
        genomes = [session.query(GraphGenome)\
            .filter(GraphGenome.task_id == 5, GraphGenome.generation == i).all()
            for i in range(89)]
        
        import pandas as pa
        df = pa.DataFrame([{
            "Generation": gen.generation,
            #"GraphId": gen.graph_id,
            "Runtime": float(gen.solution.runtime)            
        } for genome_generation in genomes for gen in genome_generation])
        mean_df = df.groupby(["Generation"]).mean()
        max_df = df.groupby(["Generation"]).max()
        
        fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)
        plt.suptitle("Absolute turn cost IP solver instance generation results")
        fig.set_size_inches(10,3.8)
        ax1.set_title("Mean runtime")
        ax2.set_title("Max runtime")
        ax1.set_ylabel("Runtime (s)")
        mean_df.plot(ax=ax1)
        max_df.plot(ax=ax2)
        ax1.get_legend().set_visible(False)
        ax2.get_legend().set_visible(False)
        plt.show()
        
        visualize_graph_2d(genomes[15][0].graph)
        visualize_graph_2d(genomes[25][0].graph)
        visualize_graph_2d(genomes[45][0].graph)
        visualize_graph_2d(genomes[60][0].graph)
        visualize_graph_2d(genomes[80][0].graph)
        visualize_solution_2d(genomes[15][0].solution)
        visualize_solution_2d(genomes[25][0].solution)
        visualize_solution_2d(genomes[45][0].solution)
        visualize_solution_2d(genomes[60][0].solution)
        visualize_solution_2d(genomes[80][0].solution)
        


def visualize_sols_inst_gen():
    from database import Task, TaskJobs, Config, Graph, get_session, CelestialGraph, AngularGraphSolution
    from utils import visualize_graph_2d, visualize_min_sum_sol_2d, visualize_solution_2d
    session = get_session("angular_copy.db")
    with session.no_autoflush:
        task = session.query(Task).filter(Task.task_type == "CelestialGraphInstances").one()
        task_jobs = session.query(TaskJobs).filter(TaskJobs.task == task).all()
        cel_graphs_old = session.query(CelestialGraph)\
            .filter(CelestialGraph.id.in_([job.graph_id for job in task_jobs]), CelestialGraph.vert_amount <= 8)\
            .all()
        cel_graphs = [i for i in cel_graphs_old if i.id in [44,68,81,82,96,1, 21, 12, 3]]

        min_sum_sols = [session.query(AngularGraphSolution).filter(AngularGraphSolution.graph_id == cel.id, AngularGraphSolution.is_optimal == True, AngularGraphSolution.solution_type == "min_sum").all() for cel in cel_graphs]
        local_min_sum_sols = [session.query(AngularGraphSolution).filter(AngularGraphSolution.graph_id == cel.id, AngularGraphSolution.is_optimal == True, AngularGraphSolution.solution_type == "local_min_sum").all() for cel in cel_graphs]
        makespan_sols = [session.query(AngularGraphSolution).filter(AngularGraphSolution.graph_id == cel.id, AngularGraphSolution.is_optimal == True, AngularGraphSolution.solution_type == "makespan").all() for cel in cel_graphs]
        for g, min_s, l_s, m_s in zip(cel_graphs, min_sum_sols, local_min_sum_sols, makespan_sols):
            print("Graph-id", g.id)
            visualize_graph_2d(g)

            if min_s:
                print("Minsum:")
                visualize_min_sum_sol_2d(min_s[0])
            if l_s:
                print("LocalMinSum")
                visualize_min_sum_sol_2d(l_s[0])
            if m_s:
                print("Makespan")
                visualize_solution_2d(m_s[0])
    return

def solution_example_intro():
    solver1 = ALL_SOLVER["ConstraintAbsSolver"]()
    solver2 = ALL_SOLVER["ConstraintDependencySolver"]()
    solver3 = ALL_SOLVER["ConstraintDependencyLocalMinSumSolver"]()
    s52 = create_circle_n_k(5,2)
    visualize_graph_2d(s52)
    sol1 = solver1.solve(s52)
    sol2 = solver2.solve(s52)
    sol3 = solver3.solve(s52)
    visualize_solution_2d(sol1)
    visualize_min_sum_sol_2d(sol1)
    print("makespan", sol1.makespan)
    visualize_solution_2d(sol2)
    visualize_min_sum_sol_2d(sol2)
    print("MinSum", sol2.min_sum)
    visualize_solution_2d(sol3)
    visualize_min_sum_sol_2d(sol3)
    print("LocalMinSum", sol2.local_min_sum)
    return
    

if __name__ == "__main__":
    main()



#test_ip.test_ip_solver()
#test_ip.test_ip_solver()