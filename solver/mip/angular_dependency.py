import math
from typing import Union, Optional, List, Tuple
import gurobipy as grb
import numpy as np

from utils import Multidict, callback_rerouter, convert_graph_to_angular_abstract_graph, calculate_times, is_debug_env
from utils.dependency_graph import DependencyNode, calculate_order, DisconnectedDependencyGraphException, CircularDependencyException, calculate_cycle
from solver import Solver
from database import AngularGraphSolution, Graph

class AngularDependencySolver(Solver):
    solution_type = "min_sum"

    def __init__(self, time_limit=900, with_vertex_subtour_constr=False, **kwargs):
        self.with_vertex_subtour_constr = with_vertex_subtour_constr
        self.graph = None
        self.abstract_graph = None
        self.model = None
        self.edges = None
        self.v_incident_edges = dict()
        #self.vertex_edges = None
        super().__init__(kwargs.pop("params", {"TimeLimit": time_limit}))

    def is_multicore(self):
        return True

    def solve(self, graph: Graph, **kwargs):
        error_message = None
        returned_order = None
        is_optimal = False
        runtime = 0
        try:
            self.build_model(graph)
            if "time_limit" in kwargs:            
                self.model.setParam("TimeLimit", kwargs.pop("time_limit"))
            self.add_start_solution(graph, kwargs.pop("start_solution", None))
            self._add_callbacks(kwargs.pop("callbacks", None))
            if kwargs.pop("relax", False):
                old_edges = self.edges
                used_edges = None
                rel_model = self.model.relax()
                keys, self.edges = grb.multidict({key: rel_model.getVarByName(self.edges[key].VarName) for key in self.edges})
                rel_model.optimize(callback_rerouter)
                runtime = self.model.Runtime
                
            else:
                circle_found = True
                max_runtime = self.params["TimeLimit"]
                while(circle_found and max_runtime > 0):
                    self.model.optimize(callback_rerouter)
                    max_runtime -= self.model.Runtime
                    runtime = abs(self.params["TimeLimit"] - max_runtime)
                    try:
                        used_edges = grb.tupledict({key: self.edges[key] for key in self.edges if not math.isclose(0, self.edges[key].x, abs_tol=10**-6)})
                        circle_found = self._check_for_cycle(used_edges, self.model, lazy=False)
                        if circle_found and max_runtime > 0:
                            self.model.setParam("TimeLimit", max_runtime)
                    except AttributeError as e:
                        # Can happen if no solution was found in the time limit
                        # If not, raise error
                        if runtime < self.params["TimeLimit"]:
                            raise e
                    

            is_optimal = self.model.Status == grb.GRB.OPTIMAL
            if is_debug_env():
                local_subtours = 0
                for circle in self.found_circles:
                    verts = [key[1] for key in circle]
                    for v_i in self.v_incident_edges:
                        if self.v_incident_edges[v_i].issuperset(verts):
                            local_subtours += 1
                            break
                print("Overall subtours:", len(self.found_circles), "local subtours:", local_subtours,\
                    "in percent:", local_subtours*100/len(self.found_circles))
            
            try:
                used_edges = {key: self.edges[key] for key in self.edges if not math.isclose(0, self.edges[key].x, abs_tol=10**-6)}
                dep_graph = self._get_dep_graph(used_edges)
                order = calculate_order(dep_graph, calculate_circle_dep=True)
                returned_order = [tuple(self.abstract_graph.vertices[i]) for i in order]
            except (CircularDependencyException, AttributeError) as e:
                # If we have a circular dependency after the time limit we just didnt managed to get a feasable solution in time
                # Else something went wrong and the error should be raised
                if runtime < self.params["TimeLimit"]:
                    raise e
        except Exception as e:
            error_message = str(e)
            if is_debug_env():
                raise e
        #times = calculate_times(returned_order, self.graph)
        sol = AngularGraphSolution(self.graph,
                                    runtime,
                                    solution_type=self.solution_type,
                                    solver=self.__class__.__name__,
                                    is_optimal=is_optimal,
                                    order=returned_order,
                                    error_message=error_message)
        return sol

    def build_model(self, graph: Graph):
        self.found_circles = []
        self.graph = graph
        self.abstract_graph = convert_graph_to_angular_abstract_graph(graph, simple_graph=False)
        self.model = grb.Model()
        self.model.setParam("LazyConstraints", 1)
        for param in self.params:
            self.model.setParam(param, self.params[param])

        #self.vertex_edges = grb.tupledict()
        costs = self._add_variables(self.abstract_graph)
        self._add_objective(costs)
        self._add_constraints()
        
        self.model.update()
    
    def add_start_solution(self, graph: Graph, solution: Union[AngularGraphSolution, List[Tuple[int, int]]]):
        if solution is None:
            return
        if isinstance(solution, AngularGraphSolution):
            assert graph == solution.graph, "Solution does not match the graph"
            edge_indices = {(v1, v2): i for i, (v1, v2) in enumerate(graph.edges)}
            heads = [[] for i in range(graph.vert_amount)]
            order = solution.order
        else:
            order = solution
        for edge in order:
            for vert in edge:
                other = set(edge).difference([vert]).pop()
                heads[vert].append(other)
        cost = 0
        for i, head in enumerate(heads):
            prev = None
            for vertex in head:
                if prev is not None:
                    sorted_edge_prev = tuple(sorted([i, prev]))
                    sorted_edge = tuple(sorted([i, vertex]))
                    cost += self.abstract_graph.costs[edge_indices[sorted_edge_prev], edge_indices[sorted_edge]]
                    abs_edge = self.edges[edge_indices[sorted_edge_prev], edge_indices[sorted_edge]]
                    abs_edge.Start = 1
                prev = vertex        
    
    def set_output(self, get_output: bool):
        try:
            self.model.setParam("OutputFlag", int(get_output))
        except Exception as e:
            raise e
    def set_max_threads(self, max_threads: int):
        self.model.setParam(grb.GRB.Param.Threads, max_threads)

    def _add_variables(self, abs_graph: Graph):
        edges, costs = grb.multidict(abs_graph.costs)
        self.edges = self.model.addVars(edges, vtype=grb.GRB.BINARY, name="Abs_graph_edges")
        return costs

    def _add_constraints(self):
        l = len(self.abstract_graph.vertices)
        for v_i in range(l):
            v = self.abstract_graph.vertices[v_i]
            for v_j in v:
                incident_vertices = [
                    i for i in range(l)
                    if np.intersect1d(self.abstract_graph.vertices[i], v_j).size > 0
                    ]
                sub_out = self.edges.subset(v_i, incident_vertices)
                sub_in = self.edges.subset(incident_vertices, v_i)
                if len(sub_out) > 1:
                    self.model.addConstr(sub_out.sum() <= 1)
                if len(sub_in) > 1:
                    self.model.addConstr(sub_in.sum() <= 1)

        for v_i in range(len(self.graph.vertices)):
            # Constraint over all vertices: least 2k-1 connections between incident edges
            incident_vertices = [
                i for i in range(l)
                if np.intersect1d(self.abstract_graph.vertices[i], v_i).size > 0
                ]
            self.v_incident_edges[v_i] = set(incident_vertices)
            if len(incident_vertices) > 1:
                self.model.addConstr(
                    self.edges.sum(incident_vertices, incident_vertices) == len(incident_vertices)-1,
                    name="IncidentEdgeNumConstr")
        
        # No self circle
        for t in self.edges:
            sub = self.edges.subset(t, t)
            self.model.addConstr(sub.sum() <= 1, name="SelfCycleConstr")
        
        

    def _add_objective(self, costs):
        self.model.setObjective(sum([costs[edge] * self.edges[edge] for edge in self.edges]), grb.GRB.MINIMIZE)

    def _general_circle_elimination(self, model: grb.Model):
        edges_solution = model.cbGetSolution(self.edges)
        used_edges = grb.tupledict({key: edges_solution[key] for key in edges_solution if not math.isclose(0, edges_solution[key], abs_tol=10**-5)})
        for edge in used_edges:
            if used_edges[edge] < 0.7:
                print("Found an edge with value less than 0.7")
        
        self._check_for_cycle(used_edges, model)
        return
        # Turn used_edges into a dependency graph
        dep_graph = self._get_dep_graph(used_edges)

        try:
            calculate_order(dep_graph, calculate_circle_dep=True)
        except CircularDependencyException as dep_exception:
            self._add_cycle_constr(dep_exception.circle_nodes, model)
        # For now we try to ignore this disconnected graphs
        except DisconnectedDependencyGraphException as disc_exception:
            cycle = calculate_cycle(disc_exception.disconnected_nodes)
            self._add_cycle_constr(cycle, model)
        #print("WARNING: DISCONNECTED DEPENDENCY GRAPH DETECTED!")
    
    def _check_for_cycle(self, used_edges: grb.tupledict, model: grb.Model, lazy=True):
        unseen = {i for i in range(len(self.abstract_graph.vertices))}
        queued = set()
        while unseen:
            queued.add((None, unseen.pop()))
            prev = {}
            seen = set()
            while queued:
                edge = queued.pop()
                sub = used_edges.subset(edge[1], '*')
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
                            self.found_circles.append(path)
                            expr = None
                            if self.with_vertex_subtour_constr:
                                abs_vert_on_path = [key[1] for key in path]
                                for v_i in range(self.graph.vert_amount):
                                    if self.v_incident_edges[v_i].issuperset(abs_vert_on_path):
                                        diff = self.v_incident_edges[v_i].difference(abs_vert_on_path)
                                        incoming = self.edges.subset(diff, abs_vert_on_path)
                                        outgoing = self.edges.subset(abs_vert_on_path, diff)
                                        expr = incoming.sum() + outgoing.sum() >= 1
                                        break
                                
                            if expr is None:
                                expr = sum(self.edges[i] for i in path) <= len(path)-1
                            
                            if lazy:
                                model.cbLazy(expr)
                            else:
                                model.addConstr(expr)
                            #print("Found cycle", path)
                            return True
                        #print("No cycle found for:", prev)
        return False
                            

    def _add_cycle_constr(self, cycle_nodes, model: grb.Model):
        cycle = [nodes.value for nodes in cycle_nodes]
        l = len(cycle)
        cycle_edges = [
            (cycle[i], cycle[((i+1) % l)])
            for i in range(l)
            ]
        cycle_edges_rev = [
            (cycle[((i+1) % l)], cycle[i])
            for i in range(l)
            ]
        #print("CYCLE EDGES:", cycle_edges)
        l = len(cycle_edges)-1
        try:
            cycle_edges_vars = grb.tupledict({i: self.edges[i] for i in cycle_edges})
            cycle_edges_rev_vars = grb.tupledict({i: self.edges[i] for i in cycle_edges_rev})
            exp = grb.LinExpr(cycle_edges_vars.sum())
            exp_rev = grb.LinExpr(cycle_edges_rev_vars.sum())
            model.cbLazy(exp <= l)
            model.cbLazy(exp_rev <= l)
            #model.addConstr(exp <= l)
        except grb.GurobiError as err:
            print("ERROR: Gurobi error with number:", err.errno)

    def _get_dep_graph(self, used_edges):
        dep_graph = {key: DependencyNode(key) for key in range(len(self.abstract_graph.vertices))}
        for come, to in used_edges:
            dep_graph[come].add_dependency(dep_graph[to])
        return dep_graph

    def _add_callbacks(self, callbacks: Optional[Union[Multidict, dict]] = None):
        # Add callbacks
        own_callbacks = Multidict({grb.GRB.Callback.MIPSOL: self._general_circle_elimination})
        if callbacks:
            own_callbacks.update(callbacks)
        callback_rerouter.inner_callbacks = own_callbacks

    def _cleanup(self, **kwargs):
        callback_rerouter.inner_callbacks = None
        self.abstract_graph = None
        self.graph = None
        self.model = None
        self.edges = None
        self.add_path_at_start = self._overridden_path_at_start
            

class AngularDependencyLocalMinSumSolver(AngularDependencySolver):
    solution_type = "local_min_sum"

    def __init__(self, **kwargs):
        self.local_sum = None
        super().__init__(**kwargs)

    def _add_variables(self, abs_graph: Graph):
        super()._add_variables(abs_graph)
        self.local_sum = self.model.addVar(lb=0, name="local_sum")

    def _add_constraints(self):
        super()._add_constraints()
        l = len(self.abstract_graph.vertices)
        for v_i in range(len(self.graph.vertices)):
            # Constraint over all vertices: least 2k-1 connections between incident edges
            incident_vertices = [
                i for i in range(l)
                if np.intersect1d(self.abstract_graph.vertices[i], v_i).size > 0
                ]
            edges = self.edges.subset(incident_vertices, incident_vertices)
            self.model.addConstr(sum(self.abstract_graph.costs[key] * edges[key] for key in edges)
                                 <= self.local_sum,
                                 name="local_sum_constr")
    
    def _add_objective(self, costs):
        self.model.setObjective(self.local_sum, grb.GRB.MINIMIZE)
            