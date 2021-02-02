from typing import Optional, Union
import numpy as np

import gurobipy as grb

from utils import Multidict, get_angles, get_angle, callback_rerouter, get_array_greater_zero, convert_graph_to_angular_abstract_graph
from solver import Solver
from database import Graph, AngularGraphSolution

class AngularGraphTsp(Solver):
    def __init__(self, **kwargs):        
        self.graph = None
        self.model = None
        self.edges = None
        self.vertex_edges = None
        super().__init__(kwargs.pop("params", {"TimeLimit": 900}))

    def is_multicore(self):
        return True

    def solve(self, graph: Graph, **kwargs):
        self.abstract_graph = convert_graph_to_angular_abstract_graph(graph)
        self.graph = graph
        
        self.model = grb.Model()
        for param in self.params:
            self.model.setParam(param, self.params[param])

        self._add_callbacks(kwargs.pop("callbacks", None))

        self.vertex_edges = grb.tupledict()
        costs = self._add_variables(self.abstract_graph)
        for vertex_index in len(max(self.abstract_graph.vertices)):
            edges = self._build_hamilton_path(vertex_index, self.abstract_graph)
            edges_dict = grb.tupledict({(vertex_index, u, v): self.edges[u, v] for u, v in edges})
            self.vertex_edges.update(edges_dict)
        self._add_objective(costs)

        self.model.optimize()
        # ToDo: Add subconstraints and write to solution


    def _add_variables(self, abs_graph: Graph):
        self.edges, costs = grb.tupledict(abs_graph.costs)
        self.model.addVars(self.edges)
        return costs


    def _build_hamilton_path(self, current_node_index: int, abstract_graph: Graph):
        edges = grb.tupledict({
            (u, v): self.edges[u, v] for u, v in self.edges
            if current_node_index in abstract_graph.vertices[u] or
            current_node_index in abstract_graph.vertices[v]})
        vertices_index = {i for i in len(abstract_graph.vertices)
                          if current_node_index in abstract_graph.vertices[i]}

        for vertex_index in vertices_index:
            self.model.addConstr(edges.sum(vertex_index, '*') + edges.sum('*', vertex_index) == 2)

        # Only |V_i|-1 edges shall be used
        self.model.addConstr(sum(edges) == len(vertices_index)-1)
        return edges

    def _add_objective(self, costs):
        self.model.setObjective(sum([costs[edge] * edge for edge in self.edges]), grb.GRB.MINIMIZE)

    def _subtour_elimination(self, model: grb.Model):
        pass #ToDo: Subtour elimination

    def _general_circle_elimination(self, model: grb.Model):
        # Calculate the adjacency matrix for the solution
        # ToDo: ^ this
        edges_solution = model.cbGetSolution(self.edges)
        ad_matrix = np.zeros(self.abstract_graph.ad_matrix.shape)
        for u, v in edges_solution:
            ad_matrix[u, v] = edges_solution[(u, v)]
        if max(np.triu(ad_matrix)) == 0 or max(np.tril(ad_matrix)) == 0:
            ad_matrix = ad_matrix + ad_matrix.T
        to_calc = {0}
        
        processed = {}
        while to_calc:
            i = to_calc.pop()
            non_zero = ad_matrix[i].nonzero()
            for j in non_zero:
                if j in to_calc or j in processed:
                    # Found a circle: ELIMINATE
                    # ToDo: implement elimination constraint
                    pass
                to_calc.add(j)
        if len(processed) < len(self.abstract_graph.vertices):
            # The processed vertices itself are a subtour/circle
            pass




        pass #ToDo: circle elimination

    def _add_callbacks(self, callbacks: Optional[Union[Multidict, dict]] = None):
        # Add callbacks
        own_callbacks = Multidict({grb.GRB.Callback.MIPSOL: self._subtour_elimination})
        own_callbacks[grb.GRB.Callback.MIPSOL] = self._general_circle_elimination
        if callbacks:
            own_callbacks.update(callbacks)
        callback_rerouter.inner_callbacks = own_callbacks

    def _cleanup(self):
        callback_rerouter.inner_callbacks = None
        self.graph = None
        self.model = None
        self.edges = None
