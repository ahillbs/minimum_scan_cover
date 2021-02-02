from os import cpu_count
import time
import math
import multiprocessing
import concurrent.futures
from database import Graph, AngularGraphSolution
from utils import get_tripeledges_from_abs_graph
from .. import Solver
from ..mip import AngularDependencySolver
from ..greedy import AngularMinSumGreedySolver

class MinSumOrderSolver(Solver):
    solution_type = "min_sum"
    def __init__(self, time_limit=900, **kwargs):
        self.relax_solver = AngularDependencySolver()
        self.max_time = time_limit
        self.poly_solver = kwargs.pop("poly_solver", AngularMinSumGreedySolver())
        self.lock = multiprocessing.Lock #Lock()
        self.start_time = time.time()
        self.graph = None
        self.upper_bound = kwargs.pop("upper_bound", None)        
        self.abs_graph = None
        self.edge_vars = None
        self.edge_vert = None
        self.possible_nodes_amount = None
        self.searched_nodes = 0
    
    def is_multicore(self):
        return True

    def solve(self, graph: Graph, **kwargs):
        self._initiate_variables(graph, kwargs.pop("upper_bound", None))

        best_order = self._start_solves(graph)
        print("Searched nodes:", self.searched_nodes,
              "of possible", self.possible_nodes_amount,
              "(", self.searched_nodes/self.possible_nodes_amount, "%)")
        sol = AngularGraphSolution(
            graph,
            time.time() - self.start_time,
            self.__class__.__name__,
            self.solution_type,
            is_optimal=self.start_time + self.max_time > time.time(),
            order=best_order)
        return sol

    def _start_solves(self, graph: Graph):
        print("Possible searches:", self.possible_nodes_amount)
        futures = []
        best_order = None
        best_val = None

        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            for orig_edge in graph.edges:
                orig_tuple = tuple(orig_edge)
                order = [orig_tuple]
                remaining_edges = [tuple(edge) for edge in graph.edges if tuple(edge) != orig_tuple]
                heading_order = [[] for i in range(graph.vert_amount)]
                futures.append(executor.submit(
                    self._inner_solve, graph, order, remaining_edges, heading_order
                    ))
            for future in concurrent.futures.as_completed(futures):
                order, val = future.result()
                if not best_val or (val and val < best_val):
                    best_order = order
                    best_val = val
        return best_order

    def _initiate_variables(self, graph: Graph, upper_bound):
        self.start_time = time.time()
        self.graph = graph
        self.upper_bound = upper_bound
        self.relax_solver.build_model(graph)
        self.relax_solver.set_output(False)
        self.relax_solver.set_max_threads(1)
        self.abs_graph = self.relax_solver.abstract_graph
        self.edge_vars = self.relax_solver.edges
        # With the edges as tripel translation we can identify the right edges easily
        tripel_edges = get_tripeledges_from_abs_graph(self.abs_graph)
        self.edge_vert = {i: {} for i in range(graph.vert_amount)}
        for edge in tripel_edges:
            self.edge_vert[edge[0]][edge[1:]] = tripel_edges[edge]
        self.possible_nodes_amount = math.factorial(graph.edge_amount)
        self.searched_nodes = 0

    def _inner_solve(self, graph: Graph, order, remaining_edges, heading_order):
        if self.start_time + self.max_time < time.time():
            return None, None
        self.searched_nodes += 1
        new_edge = order[-1]
        new_heading = [heading.copy() for heading in heading_order]

        new_heading[new_edge[0]].append(new_edge[1])
        new_heading[new_edge[1]].append(new_edge[0])
        # ToDo: Use a greedy to partially solve instances
        
        if remaining_edges and self._calculate_lb_ub(graph, order):
            relax_model = self.relax_solver.model.relax()
            for i, headings in enumerate(new_heading):
                prev = None
                for vertex in headings:
                    if prev is not None:
                        edge = self.edge_vert[i][(prev, vertex)]
                        relax_model.addConstr(
                            relax_model.getVarByName(self.edge_vars[edge].VarName) == 1
                            )
                    prev = vertex
            relax_model.optimize()
            if relax_model.Status == 2: #Model is optimal
                lower_bound = relax_model.objVal
                if self.upper_bound and lower_bound >= self.upper_bound:
                    return None, None
            if relax_model.Status == 3: #Model infeasable
                return None, None

        best_order = None
        best_val = None
        if self._calculate_lb_ub(graph, order) or not remaining_edges:
            best_order, best_val = self.poly_solver.solve(
                graph,
                presolved=order,
                headings=new_heading,
                remaining=remaining_edges,
                no_sol_return=True,
                ub=self.upper_bound
                )
            """if not remaining_edges:
                # All edges are used, we can calculate current solution
                cost = 0
                for i, headings in enumerate(new_heading):
                    prev = None
                    for vertex in headings:
                        if prev is not None:
                            edge = self.edge_vert[i][(prev, vertex)]
                            cost += self.abs_graph.costs[edge]
                        prev = vertex
            """
            if len(best_order) == graph.edge_amount:
                with self.lock():
                    if not self.upper_bound or best_val < self.upper_bound:
                        self.upper_bound = best_val
                        print("Better solution found: ", best_val)
        
        for next_edge in remaining_edges:
            inner_remaining_edges = [edge for edge in remaining_edges if edge != next_edge]
            new_order, value = self._inner_solve(graph,
                order + [next_edge],
                inner_remaining_edges,
                new_heading
                )
            if new_order and len(new_order) == graph.edge_amount and value is not None and value <= self.upper_bound:
                best_order = new_order
                best_val = value
        return best_order, best_val

    def _calculate_lb_ub(self, graph, order):
        if round(graph.edge_amount/10) > 3:
            return len(order) % round(graph.edge_amount/10) == 0
        return len(order) % 3

class SymmetricMinSumOrderSolver(MinSumOrderSolver):
    """Can be used if the graph is symmetric. Then it is sufficient to only follow the edge order starting with all edges for one vertex
    """
    def _start_solves(self, graph: Graph):
        self.possible_nodes_amount = int(self.possible_nodes_amount/graph.edge_amount)
        print("Possible searches:", self.possible_nodes_amount)
        futures = []
        best_order = None
        best_val = None

        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            for orig_edge in graph.edges:
                if 0 in orig_edge:
                    orig_tuple = tuple(orig_edge)
                    order = [orig_tuple]
                    remaining_edges = [tuple(edge) for edge in graph.edges
                                       if tuple(edge) != orig_tuple]
                    heading_order = [[] for i in range(graph.vert_amount)]
                    futures.append(executor.submit(
                        self._inner_solve, graph, order, remaining_edges, heading_order
                        ))
            for future in concurrent.futures.as_completed(futures):
                order, val = future.result()
                if not best_val or (val and val < best_val):
                    best_order = order
                    best_val = val
        return best_order
