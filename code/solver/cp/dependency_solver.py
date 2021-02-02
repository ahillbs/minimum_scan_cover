from os import cpu_count
from typing import List, Dict, Union, Optional
import math
from fractions import Fraction
import numpy as np
from ortools.sat.python import cp_model

from .. import Solver

from utils import Multidict, get_angles, convert_graph_to_angular_abstract_graph, calculate_order_from_times
from utils.dependency_graph import DependencyNode, DependencyGraph, calculate_order,\
    DisconnectedDependencyGraphException, CircularDependencyException, calculate_cycle
from database import AngularGraphSolution, Graph

class ConstraintDependencySolver(Solver):
    solution_type = "min_sum"

    def __init__(self, time_limit=900, use_lcm=False, **kwargs):
        self.graph = None
        self.abstract_graph = None
        self.model = None
        self.edges = None
        self.vertices = None
        self.multiplier = None
        self.use_lcm = use_lcm
        super().__init__(kwargs.pop("params", {"max_time_in_seconds": time_limit, "num_search_workers":cpu_count()}))

    def is_multicore(self):
        return True

    def solve(self, graph: Graph, **kwargs):
        self._build_model(graph)

        solver = cp_model.CpSolver()
        for param in self.params:
            setattr(solver.parameters, param, self.params[param])
        if "time_limit" in kwargs:
            solver.parameters.max_time_in_seconds = kwargs["time_limit"]

        status = solver.Solve(self.model)
        if status in [cp_model.FEASIBLE, cp_model.OPTIMAL]:
            obj_val = solver.ObjectiveValue()
            values = {tuple(self.abstract_graph.vertices[key]): solver.Value(self.vertices[key]) for key in self.vertices}
            order = calculate_order_from_times(values, self.graph)

            sol = AngularGraphSolution(
                graph,
                solver.UserTime(),
                self.__class__.__name__,
                self.solution_type,
                is_optimal=(status == cp_model.OPTIMAL),
                order=order,
                )
        else:
            sol = AngularGraphSolution(
                graph,
                solver.UserTime(),
                self.__class__.__name__,
                self.solution_type,
                is_optimal=(status == cp_model.OPTIMAL),
                error_message="No feasable solution found",
                )

        return sol



    def _build_model(self, graph):
        self.graph = graph
        self.abstract_graph = convert_graph_to_angular_abstract_graph(graph, simple_graph=False)
        self.model = cp_model.CpModel()

        self._add_variables()
        costs = self._to_int_cost(self.abstract_graph.costs)
        self._add_objective(costs)
        self._add_constraints()

    def _add_objective(self, costs):
        self.model.Minimize(sum([costs[edge] * self.edges[edge] for edge in self.edges]))

    def _to_int_cost(self, costs):
        if self.use_lcm:
            return self._to_int_cost_lcm(costs)
        else:
            return self._to_int_cost_gcd(costs)

    def _to_int_cost_gcd(self, costs):
        self.multiplier = 10**8        
        degree_array = np.array([costs[arc] * self.multiplier for arc in costs])
        assert degree_array.min() >= 1
        degree_array_int = np.array(degree_array, dtype=np.int)
        gcd = np.gcd.reduce(degree_array_int)
        self.multiplier /= gcd
        new_degrees = {arc: np.int(costs[arc] * self.multiplier) for arc in costs}
        return new_degrees 
    
    def _to_int_cost_lcm(self, costs):
        arcs = costs.keys()
        angles = np.array([costs[arc] for arc in arcs])
        angles = np.deg2rad(angles)
        pi_angles = angles / np.pi
        pi_angles = np.round(pi_angles, 8)
        new_costs = {key: angle for key, angle in zip(costs.keys(), pi_angles)}
        degrees_fractions = {arc: Fraction(str(new_costs[arc])) for arc in new_costs}
        denominators = np.array([degrees_fractions[key].denominator for key in degrees_fractions])
        self.multiplier = np.lcm.reduce(denominators)
        for key in new_costs:
            frac = degrees_fractions[key] * self.multiplier
            assert frac.denominator == 1 #or (frac.numerator / frac.denominator).is_integer()
            new_costs[key] = frac.numerator
        return new_costs


    def _add_variables(self):
        self.edges = {key: self.model.NewBoolVar(name=f"Abs_graph_edges_{key}") for key in self.abstract_graph.costs}
        self.vertices = {
            i: self.model.NewIntVar(0, self.abstract_graph.edge_amount, name=f"Abs_graph_edges_{self.abstract_graph.vertices[i]}")
            for i in range(self.abstract_graph.vert_amount)
            }

    def _add_constraints(self):
        l = len(self.abstract_graph.vertices)
        incident_edges_sets = {}
        for v_i in range(len(self.graph.vertices)):
            # Constraint over all vertices: least 2k-1 connections between incident edges
            incident_vertices = [
                i for i in range(l)
                if np.intersect1d(self.abstract_graph.vertices[i], v_i).size > 0
                ]
            
            incident_edges_keys = [
                key for key in self.edges
                if key[0] in incident_vertices and key[1] in incident_vertices
                ]
            incident_edges = {key: self.edges[key] for key in incident_edges_keys}
            if len(incident_vertices) > 1:
                self.model.Add(sum(incident_edges.values()) == len(incident_vertices)-1)

            incident_edges_sets[v_i] = incident_edges
            

        
        for v_i in range(l):
            from_v_i_edges = [self.edges[key] for key in self.edges if key[0] == v_i]
            to_v_i_edges = [self.edges[key] for key in self.edges if key[1] == v_i]
            
            # Constraint to ensure a only paths are allowed
            for vert in self.abstract_graph.vertices[v_i]:
                from_v_i_edges = [incident_edges_sets[vert][key] for key in incident_edges_sets[vert] if key[0] == v_i]
                to_v_i_edges = [incident_edges_sets[vert][key] for key in incident_edges_sets[vert] if key[1] == v_i]
                if from_v_i_edges:
                    self.model.Add(sum(from_v_i_edges) <= 1)
                if to_v_i_edges:
                    self.model.Add(sum(to_v_i_edges) <= 1)
            
            """
            v = self.abstract_graph.vertices[v_i]
            for v_j in v:
                incident_vertices = [
                    i for i in range(l)
                    if np.intersect1d(self.abstract_graph.vertices[i], v_j).size > 0
                    ]

                #self.edges.subset(v_i, incident_vertices)
                sub_out = [self.edges[key] for key in self.edges if key[0] == v_i and key[1] in incident_vertices]
                #self.edges.subset(incident_vertices, v_i)
                sub_in = [self.edges[key] for key in self.edges if key[1] == v_i and key[0] in incident_vertices]
                self.model.AddLinearConstraint(sum(sub_out), 0, 1)
                self.model.AddLinearConstraint(sum(sub_in), 0, 1)"""

        
            
            """ints = [i for i in range(len(incident_vertices))]
            # For circle constraint, nodes can only range from 0 to n-1
            # Therefore, we translate the indexes
            vert_to_int = {key: new_int for key, new_int in zip(incident_vertices, ints)}
            arcs = [
                (vert_to_int[v1], vert_to_int[v2], self.edges[(v1, v2)])
                for v1, v2 in incident_edges_keys
                ]
            # Add dummy node from all other nodes for a circle
            arcs.extend([(len(ints), vert_to_int[v], self.model.NewBoolVar('')) for v in incident_vertices])
            arcs.extend([(vert_to_int[v], len(ints), self.model.NewBoolVar('')) for v in incident_vertices])
            self.model.AddCircuit(arcs)"""



        # Order of indexes
        for edge in self.edges:
            self.model.Add(
                self.vertices[edge[1]] - self.vertices[edge[0]] >= 1
                ).OnlyEnforceIf(self.edges[edge])
        #    sub = self.edges.subset(t, t)
        #    self.model.Add(sub.sum() <= 1)"""

class ConstraintDependencyLocalMinSumSolver(ConstraintDependencySolver):
    solution_type = "local_min_sum"

    def _add_objective(self, costs: dict):
        max_cost = max(costs.values())
        self.max_vert = {
            i: self.model.NewIntVar(0, max_cost*self.graph.vert_amount, name=f"Max_graph_edges_{self.graph.vertices[i]}")
            for i in range(self.graph.vert_amount)
            }
        l = len(self.abstract_graph.vertices)
        for v_i in range(self.graph.vert_amount):
            # Constraint over all vertices: sum of edge costs is equal to corresponding max_vert
            incident_vertices = [
                i for i in range(l)
                if np.intersect1d(self.abstract_graph.vertices[i], v_i).size > 0
                ]
            incident_edges_keys = [
                key for key in self.edges
                if key[0] in incident_vertices and key[1] in incident_vertices
                ]
            incident_edges = {key: self.edges[key] for key in incident_edges_keys}
            self.model.Add(self.max_vert[v_i] == sum([costs[key] * self.edges[key] for key in incident_edges_keys]))

        self.max = self.model.NewIntVar(0, max_cost*self.graph.vert_amount, name="Max_graph_vert")
        self.model.AddMaxEquality(self.max, self.max_vert.values())
        self.model.Minimize(self.max)