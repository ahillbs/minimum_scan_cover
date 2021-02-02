import math
from typing import Union, Optional, List, Tuple

import gurobipy
import numpy as np

from utils import Multidict, get_angles, callback_rerouter, calculate_times, is_debug_env
from solver import Solver
from database import Graph, AngularGraphSolution

class AngularGraphScanMakespanHamilton(Solver):

    def __init__(self, time_limit=900, **kwargs):
        self.times = None
        self.angles = None
        self.edges = None
        self.graph = None
        self.model = None
        self.solution_type = "makespan"
        super().__init__(kwargs.pop("params", {"TimeLimit": time_limit}))

    def solve(self, graph, **kwargs):
        return self.build_ip_and_optimize(graph, **kwargs)
    
    def is_multicore(self):
        return True

    def build_ip_and_optimize(self, graph: Graph, start_solution: Optional[dict] = None, callbacks: Optional[Union[Multidict, dict]] = None, time_limit=None):
        error_message = None
        runtime = 0
        is_optimal = False
        times = None
        try:
            self.graph = graph
            self.model = gurobipy.Model()
            self.model.setParam("LazyConstraints", 1)
            for param in self.params:
                self.model.setParam(param, self.params[param])
            # Override general time limit, if one is provided
            if time_limit:
                self.model.setParam("TimeLimit", time_limit)

            times_first, times_reverse = self._add_times_variables()

            self.edges, self.angles = self._add_hamilton_paths()

            self._add_times_equals_constraints(times_first, times_reverse)

            self.times = gurobipy.tupledict()
            self.times.update(times_first)
            self.times.update(times_reverse)

            self._add_taken_edges_constraints()

            self._set_objective()

            self._add_callbacks(callbacks)
            self._add_pre_solution(graph, start_solution)
            self.model.update()
            self.model.optimize(callback_rerouter)
            runtime = self.model.Runtime
            is_optimal = self.model.Status == gurobipy.GRB.OPTIMAL
            times = {t: self.times[t].x for t in self.times}
        except Exception as e:
            if self.model.Status != gurobipy.GRB.TIME_LIMIT:
                error_message = str(e)
                if is_debug_env():
                    raise e

        sol = AngularGraphSolution(self.graph,
                                     runtime,
                                     times=times,
                                     solution_type=self.solution_type,
                                     is_optimal=is_optimal,
                                     solver=self.__class__.__name__,
                                     error_message=error_message)
        return sol

    def _build_hamilton_path(self, current_node_index: int):
        # Get adjacent vertices
        ad_vert = [j for j in range(len(self.graph.vertices)) if self.graph.ad_matrix[current_node_index, j] > 0]

        # Calculate angles
        angles = self._get_angles(current_node_index, ad_vert)
        edges = None
        if angles[0]:
            edges = self.model.addVars(angles[0], vtype=gurobipy.GRB.BINARY, name="edges")
            self.model.update()
            for vertex_index in ad_vert:
                # outgoing edges
                self.model.addConstr(
                    edges.sum(current_node_index, '*', vertex_index) <= 1,
                    name="Incoming_{0}_{1}".format(current_node_index, vertex_index)
                )
                # incoming edges
                self.model.addConstr(
                    edges.sum(current_node_index, vertex_index, '*') <= 1,
                    name="Outgoing_{0}_{1}".format(current_node_index, vertex_index)
                )            
            # Only |V_i|-1 edges shall be used
            self.model.addConstr(edges.sum('*', '*', '*') == len(ad_vert)-1)
        return edges, angles[1]

    def _get_angles(self, index, ad_indexes) -> list:
        v_a = np.array(self.graph.vertices[index])
        vert_arr = np.array([self.graph.vertices[i] for i in ad_indexes])
        l = len(vert_arr)
        degrees = get_angles(v_a, vert_arr)
        tuple_list = [((index, ad_indexes[i], ad_indexes[j]), degrees[i, j])
             for i in range(l) for j in range(l) if i != j]
        arcs, multidict = gurobipy.multidict(tuple_list) if tuple_list else (None, None)
        return arcs, multidict

    def _subtour_elimination(self, model: gurobipy.Model):
        #nodecnt = model.cbGet(gurobipy.GRB.Callback.MIPSOL_NODCNT)
        #obj = model.cbGet(gurobipy.GRB.Callback.MIPSOL_OBJ)
        #solcnt = model.cbGet(gurobipy.GRB.Callback.MIPSOL_SOLCNT)
        #solution = model.cbGetSolution(self.edges)
        # for every subsolution, try to find shortcuts
        for base_vertex_index, edges in self.edges.items():
            edge_solution = model.cbGetSolution(edges)
            used_edges = {key: edge_solution[key] for key in edge_solution if edge_solution[key] > 0.1}
            ad_vert = {key[1]: key for key in used_edges}
            visited = set()
            all_ad_vert = {key[1]: key for key in edge_solution}
            vertices_amount = len(all_ad_vert)

            while len(visited) < vertices_amount:
                current_vertices = []
                vertex, edge_key = self._get_start_vertex(used_edges, visited)
                while vertex not in visited:
                    visited.add(vertex)
                    current_vertices.append(vertex)
                    vertex = edge_key[2]
                    try:
                        edge_key = ad_vert[vertex]
                    except KeyError:
                        # Should only happen if we get to an end of a path
                        # Therefore, just add them to also mark them as visited
                        visited.add(vertex)
                        current_vertices.append(vertex)
                # Found a subtour, need to eliminate it
                if len(current_vertices) < vertices_amount:
                    not_current_vertices = [i for i in all_ad_vert if i not in current_vertices]
                    model.cbLazy(
                        edges.sum(base_vertex_index, current_vertices, not_current_vertices) +
                        edges.sum(base_vertex_index, not_current_vertices, current_vertices)
                        >= 1
                    )
                    print("Subtour found for point", base_vertex_index,
                          "with vertices", current_vertices)

    def _add_callbacks(self, callbacks):
        own_callbacks = Multidict({gurobipy.GRB.Callback.MIPSOL: self._subtour_elimination})
        if callbacks:
            own_callbacks.update(callbacks)
        callback_rerouter.inner_callbacks = own_callbacks

    def _get_start_vertex(self, used_edges, visited):
        ad_vert = {key[1]: key for key in used_edges if key[1] not in visited}
        ad_vert2 = {key[2]: key for key in used_edges}
        not_in_vert2 = {key for key in ad_vert if key not in ad_vert2}
        # This happens if only subtours are left
        if ad_vert and not not_in_vert2:
            v = next(iter(ad_vert))
            return v, ad_vert[v]
        # Else use the first vertex found from start vertices
        for key in used_edges:
            if key[1] in not_in_vert2 and key[1] not in visited:
                return key[1], key
        raise KeyError("Could not find an unvisited vertex")

    def _add_times_variables(self):
        times_first = self.model.addVars(
            [
                (i, j)
                for i in range(len(self.graph.vertices))
                for j in range(i+1, len(self.graph.vertices))
                if self.graph.ad_matrix[i, j] > 0
            ]
            , name="time")
        times_reverse = self.model.addVars({(e[1], e[0]) for e in times_first}, name="time")
        return times_first, times_reverse

    def _add_hamilton_paths(self):
        sorted_edges = {}
        angles = gurobipy.tupledict()
        for i in range(len(self.graph.vertices)):
            edges, angle = self._build_hamilton_path(i)
            if edges:
                angles.update(angle)
                sorted_edges[i] = edges
        return sorted_edges, angles

    def _add_times_equals_constraints(self, times_first, times_reverse):
        for time in times_first:
            reverse = (time[1], time[0])
            self.model.addConstr(times_first[time] - times_reverse[reverse] == 0)

    def _add_taken_edges_constraints(self):
        big_m = math.log2(len(self.graph.vertices)) * 360
        for a_k in self.angles:
            self.model.addConstr(
                self.times[a_k[0], a_k[1]] - self.times[a_k[0], a_k[2]] + 
                self.angles[a_k] + big_m * self.edges[a_k[0]][a_k]
                <= big_m
                )

    def _set_objective(self):
        max_time = self.model.addVar(name="Max_t")
        self.model.setObjective(max_time, gurobipy.GRB.MINIMIZE)
        for time in self.times:
            self.model.addConstr(self.times[time] - max_time <= 0, name="below_max_t")

    def _add_pre_solution(self, graph: Graph, start_solution: Union[AngularGraphSolution, List[Tuple[int, int]]]):
        if start_solution is None:
            return
        if not isinstance(start_solution, AngularGraphSolution):
            times = calculate_times(start_solution, graph)
        else:
            times = start_solution.times
        for key in times:
                self.times[key].Start = times[key]
                


class AngularGraphScanMinSumHamilton(AngularGraphScanMakespanHamilton):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solution_type = "min_sum"
    def _set_objective(self):
        self.model.setObjective(sum([self.angles[key] * self.edges[v_key][key] for v_key in self.edges for key in self.edges[v_key]]), gurobipy.GRB.MINIMIZE)

class AngularGraphScanLocalMinSumHamilton(AngularGraphScanMakespanHamilton):
    solution_type = "local_min_sum"
    def _set_objective(self):
        local_sum = self.model.addVar(lb=0, name="local_sum")
        for v_key in self.edges:
            self.model.addConstr(sum([self.angles[key] * self.edges[v_key][key] for key in self.edges[v_key]]) <= local_sum)
        self.model.setObjective(local_sum, gurobipy.GRB.MINIMIZE)
        