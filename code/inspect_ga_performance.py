import math
import numpy as np
import pandas as pd
from genetic_algorithm import GeneticAlgorithm, update_callback
from database import AngularGraphSolution
from solver.meta_heur import AngularGeneticMinSumSolver, AngularGeneticLocalMinSumSolver
from instance_generation import create_random_celest_graph
from utils import calculate_times

class DataHolder:
    def __init__(self, dataFrame: pd.DataFrame):
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
        self.dataFrame = self.dataFrame.append(pd.DataFrame(data), ignore_index=True)
        update_callback(gen_algo)

def inspect_ga_performance():
    dh = DataHolder(pd.DataFrame())
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

    df_graphs = pd.DataFrame([{
        "graph_name": g.name ,
        "vert_amount": g.vert_amount,
        "edge_amount": g.edge_amount
    } for g in graphs])
    
    print(dh.dataFrame)
    dh.dataFrame.to_pickle("GA_comparison.pk")
    df_graphs.to_pickle("GA_comparison_graphs.pk")

if __name__ == "__main__":
    inspect_ga_performance()