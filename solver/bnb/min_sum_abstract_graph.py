import numpy as np
import time
from .. import Solver
from ..mip import AngularDependencySolver
from database import Graph, AngularGraphSolution
from utils import convert_graph_to_angular_abstract_graph, calculate_times
from utils.dependency_graph import DependencyNode, calculate_order, DisconnectedDependencyGraphException, CircularDependencyException, calculate_cycle
import asyncio

class MinSumAbstractGraphSolver(Solver):
    solution_type = "min_sum"
    def __init__(self, params=None, check_all_node_cycles=False, time_limit=900):
        self.params = params
        self.abs_graph = None
        self.upper_bound = None
        self.upper_bound_sol = None
        self.solve_time_start = None
        self.relax_solver = AngularDependencySolver()
        self.check_all_node_cycles = check_all_node_cycles
        self.max_time = time_limit
        self.time_limit = time_limit
    
    def is_multicore(self):
        return True

    def solve(self, graph: Graph, **kwargs):
        self.time_limit = kwargs.pop("time_limit", self.max_time)
        self.upper_bound = kwargs.pop("upper_bound", None)
        self.relax_solver.build_model(graph)
        self.relax_solver.set_output(False)
        self.abs_graph = self.relax_solver.abstract_graph
        self.edge_vars = self.relax_solver.edges
        self.max_edges = 0
        for v_i in range(len(graph.vertices)):
            # Constraint over all vertices: least 2k-1 connections between incident edges
            incident_vertices = [
                i for i in range(self.abs_graph.vert_amount)
                if np.intersect1d(self.abs_graph.vertices[i], v_i).size > 0
                ]
            self.max_edges += (len(incident_vertices)-1)

        self.solve_time_start = time.time()
        self.lock = asyncio.Lock()
        asyncio.get_event_loop().run_until_complete(
            self._inner_solve(self.abs_graph, {}, 0, 0, {})
        )

        status = 'OPTIMAL'
        if self.time_limit and time.time() - self.solve_time_start > self.time_limit:
            status = 'TIME_LIMIT'
        returned_order = None
        self.upper_bound = None
        if self.upper_bound_sol:
            dep_graph = self._get_dep_graph(self.upper_bound_sol)
            order = calculate_order(dep_graph, calculate_circle_dep=True)
            returned_order = [tuple(self.abs_graph.vertices[i]) for i in order]
        sol = AngularGraphSolution(graph,
                                    time.time() - self.solve_time_start,
                                    solution_type=self.solution_type,
                                    solver=self.__class__.__name__,
                                    is_optimal=status == "OPTIMAL",
                                    order=returned_order)
            
        self.upper_bound_sol = None
        return sol

    async def _inner_solve(self, graph, fixed_edges: dict, edge_index: int, chosen_amount: int, relaxed_sol: dict):
        if self.time_limit and time.time() - self.solve_time_start > self.time_limit:
            return
        # First we check if it could already be a viable solution
        if self.check_all_node_cycles or (chosen_amount == self.max_edges) or (edge_index == graph.edge_amount):
            used_edges = {key for key in fixed_edges if fixed_edges[key] == 1}
            if self._contains_cycle(graph, used_edges):
                return
        
        
        prev_edge_key = self.edge_vars.keys()[edge_index-1] if edge_index > 0 else None
        if not relaxed_sol or relaxed_sol[prev_edge_key] != fixed_edges[prev_edge_key]:
            relax_model = self.relax_solver.model.relax()
            for edge in fixed_edges:
                relax_model.addConstr(
                    relax_model.getVarByName(self.edge_vars[edge].VarName) == fixed_edges[edge]
                    )
            relax_model.optimize()
            if relax_model.Status == 2: #Model is optimal
                lower_bound = relax_model.objVal
                if self.upper_bound and lower_bound > self.upper_bound:
                    return
            if relax_model.Status == 3: #Model infeasable
                return
            relaxed_sol = {
                    edge: relax_model.getVarByName(self.edge_vars[edge].VarName).x
                    for edge in self.edge_vars
                    }
        
        if chosen_amount == self.max_edges:
            cost = sum([graph.costs[key] for key in used_edges])
            async with self.lock:
                if not self.upper_bound or self.upper_bound > cost:
                    self.upper_bound = cost
                    self.upper_bound_sol = used_edges
                    print("Better solution found: ", cost)
            return
        if edge_index == graph.edge_amount:
            return

        curr_edge_key = self.edge_vars.keys()[edge_index]
        fixed_edges_0 = fixed_edges.copy()
        fixed_edges_0[curr_edge_key] = 0
        fixed_edges_1 = fixed_edges.copy()
        fixed_edges_1[curr_edge_key] = 1
        
        await self._inner_solve(graph, fixed_edges_0, edge_index+1, chosen_amount, relaxed_sol)
        await self._inner_solve(graph, fixed_edges_1, edge_index+1, chosen_amount+1, relaxed_sol)

    def _contains_cycle(self, abstract_graph, used_edges):
        unseen = {i for i in range(len(abstract_graph.vertices))}
        queued = set()
        while unseen:
            queued.add((None, unseen.pop()))
            prev = {}
            seen = set()
            while queued:
                edge = queued.pop()
                sub = {key for key in used_edges if key[0] == edge[1]}
                seen.add(edge[1])
                for key in sub:
                    destination = key[1]
                    if destination in unseen:
                        try:
                            pass#unseen.remove(destination)
                        except KeyError:
                            print("Double unseen event!")
                            pass # It can happen that some nodes will be seen multiple times
                        queued.add(key)
                        prev[key] = edge
                    if destination in seen:
                        path = [key, edge]
                        while path[-1][0] and path[-1][0] != destination:
                            path.append(prev[path[-1]])
                        if path[-1][0] == destination:
                            #print("Found cycle", path)
                            return True
                        #print("No cycle found for:", prev)
        return False

    def _get_dep_graph(self, used_edges):
        dep_graph = {key: DependencyNode(key) for key in range(len(self.abs_graph.vertices))}
        for come, to in used_edges:
            dep_graph[come].add_dependency(dep_graph[to])
        return dep_graph