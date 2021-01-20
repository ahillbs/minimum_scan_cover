from os import cpu_count
import itertools
from typing import List, Dict
from fractions import Fraction
import numpy as np
import scipy.sparse as sparse
from ortools.sat.python import cp_model

from database import Graph, AngularGraphSolution
from .. import Solver
from utils import get_angles

class ConstraintAbsSolver(Solver):
    solution_type = "makespan"

    def __init__(self, time_limit=900, use_lcm=False, **kwargs):
        cpus = cpu_count()
        super().__init__(kwargs.pop("params", {"max_time_in_seconds": time_limit, "num_search_workers":cpus}))
        self.model = None
        self.graph = None
        self.max_time = None
        self.degrees = None
        self.use_lcm = use_lcm

    def is_multicore(self):
        return True        

    def solve(self, graph: Graph, **kwargs):
        self.model = cp_model.CpModel()
        self.graph = graph

        time = 0
        is_optimal = False
        times_dict = None
        error_message = None
        try:

            arcs, self.degrees = self._calculate_degrees()
            time, diffs, absolutes, self.max_time = self._add_variables(arcs)
            self._add_constraints(arcs, self.degrees, time, diffs, absolutes, self.max_time)

            #self._add_initial_heading(time, initial_heading)

            #self._add_pre_solution(time, start_solution)
            self._setObj()

            solver = cp_model.CpSolver()
                
            for param in self.params:
                val = self.params[param]
                setattr(solver.parameters, param, val)
            if "time_limit" in kwargs:
                solver.parameters.max_time_in_seconds = kwargs["time_limit"]

            status = solver.Solve(self.model)
            if status in [cp_model.FEASIBLE, cp_model.OPTIMAL]:
                values = np.array([solver.Value(time[key]) for key in time])
                times = np.degrees((values / self.multiplier) * np.pi)
                times_dict = {key: value for key, value in zip(time.keys(), times)}
            is_optimal = (status == cp_model.OPTIMAL)
            time = solver.UserTime()
        except Exception as e:
            error_message = str(e)
        sol = AngularGraphSolution(
            graph,
            time,
            self.__class__.__name__,
            self.solution_type,
            is_optimal=is_optimal,
            times=times_dict,
            error_message=error_message
            )
        return sol

    def _setObj(self):
        self.model.Minimize(self.max_time)

    def _calculate_degrees(self) -> (List[tuple], Dict[tuple, float]):
        # Calculate degrees
        degrees = {}
        arcs = []
        for i in range(len(self.graph.vertices)):
            arcs_i, degrees_i = self._get_angles(i)
            if arcs_i:
                degrees.update(degrees_i)
                arcs.extend(arcs_i)
        
        if self.use_lcm:
            return self._calculate_degrees_lcm(degrees, arcs)
        else:
            return self._calculate_degrees_gcd(degrees, arcs)

    def _calculate_degrees_gcd(self, degrees, arcs) -> (List[tuple], Dict[tuple, float]):        
        self.multiplier = 10**8        
        degree_array = np.array([degrees[arc] * self.multiplier for arc in arcs])
        try:
            assert degree_array.min() >= 1, "Values of the degree are still smaller than 1 after multiplication with 10^8. Will set the value to 0"
        except AssertionError as e:
            print(e)
        degree_array_int = np.array(degree_array, dtype=np.int)
        gcd = np.gcd.reduce(degree_array_int)
        self.multiplier /= gcd
        new_degrees = {arc: np.int(degrees[arc] * self.multiplier) for arc in arcs}
        return arcs, new_degrees        

    def _calculate_degrees_lcm(self, degrees, arcs) -> (List[tuple], Dict[tuple, float]):
        degrees_fractions = {arc: Fraction(str(degrees[arc])) for arc in arcs}
        denominators = np.array([degrees_fractions[key].denominator for key in degrees_fractions])
        if len(denominators) > 0:
            self.multiplier = np.lcm.reduce(denominators)#np.lcm(*np.array([[i,j] for i,j in itertools.product(denominators, denominators)]).T).max()
        else:
            self.multiplier = 1
        
        for key in degrees:            
            frac = degrees_fractions[key]* self.multiplier
            assert frac.denominator == 1 #or (frac.numerator / frac.denominator).is_integer()
            degrees[key] = frac.numerator
        return arcs, degrees

    def _get_angles(self, index) -> (List[tuple], dict):
        v_a = np.array(self.graph.vertices[index])
        ad_indexes = [j for j in range(len(self.graph.vertices)) if self.graph.ad_matrix[index, j] > 0]
        vert_arr = np.array([self.graph.vertices[i] for i in ad_indexes])
        l = len(vert_arr)
        angles = get_angles(v_a, vert_arr, in_degree=False)

        pi_angle = angles / np.pi
        pi_angle = np.round(pi_angle, 8) # round to get rid of some rounding errors before
        tuple_dict = {(index, ad_indexes[i], ad_indexes[j]): pi_angle[i, j]
                      for i in range(l) for j in range(i+1, l)}

        #arcs, multidict = grb.multidict(tuple_list) if tuple_list else (None, None)
        return tuple_dict.keys(), tuple_dict

    def _add_variables(self, arcs):
        length = len(self.graph.vertices)
        
        # Why 2 * self.multiplier?
        # Since we devided the radians by pi, 2 is a full circle.
        # The multiplier multiplies the times, so that we get integer instead of fractions
        ub = int(2 * self.multiplier * (1 + np.ceil(np.log2(length))))
        # Add variables
        time_keys = [
            (i, j)
            for i in range(len(self.graph.vertices))
            for j in range(i+1, len(self.graph.vertices))
            if self.graph.ad_matrix[i, j] > 0
            ]
        time = {key: self.model.NewIntVar(0, ub, "time"+str(key)) for key in time_keys}
        diffs = {arc: self.model.NewIntVar(-ub, ub, "diff"+str(arc)) for arc in arcs}
        absolutes = {arc: self.model.NewIntVar(0, ub, "abs"+str(arc)) for arc in arcs}
        
        max_time = self.model.NewIntVar(0, ub, name="Max_t")
        return time, diffs, absolutes, max_time

    def _add_constraints(self, arcs, degrees, times, diffs, absolutes, max_time):
        # Add Constraints
        for time in times:
            self.model.Add(times[time] <= max_time)

        for arc in arcs:
            vi_to_vp = (min(arc[:2]), max(arc[:2]))
            vi_to_vk = (min(arc[0], arc[2]), max(arc[0], arc[2])) #(arc[0], arc[2])
            self.model.Add(times[vi_to_vp] - times[vi_to_vk] == diffs[arc])
            self.model.AddAbsEquality(absolutes[arc], diffs[arc])
            self.model.Add(absolutes[arc] >= degrees[arc])
            

class MinSumAbsSolver(ConstraintAbsSolver):
    """DOES NOT WORK

    Arguments:
        ConstraintAbsSolver {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    solution_type = "min_sum"
    def __init__(self, **kwargs):
        cpus = cpu_count()
        super().__init__()
        self.model = None
        self.graph = None
        self.edges = None

    def solve(self, graph: Graph, **kwargs):
        sol = super().solve(graph, **kwargs)
        return sol

    def _add_variables(self, arcs):
        length = len(self.graph.vertices)
        
        # Upper bound for the time
        u_b = self.graph.edge_amount#int(self.multiplier * len(arcs) *  max([self.degrees[arc] for arc in arcs]))
        # Add variables
        time_keys = [
            (i, j)
            for i in range(len(self.graph.vertices))
            for j in range(i, len(self.graph.vertices))
            if self.graph.ad_matrix[i, j] > 0
            ]
        self.time = {key: self.model.NewIntVar(0, u_b, "time"+str(key)) for key in time_keys if key[0] < key[1]}
        
        diffs = {}#{arc: self.model.NewIntVar(-u_b, u_b, "diff"+str(arc)) for arc in arcs}
        absolutes = {}#{arc: self.model.NewIntVar(0, u_b, "abs"+str(arc)) for arc in arcs}
        max_time = None#self.model.NewIntVar(0, u_b, name="Max_t")
        #self.above_zero = {arc: self.model.NewBoolVar("AboveZero"+str(arc)) for arc in arcs}
        
        self.edges = {arc: self.model.NewBoolVar("Bool"+str(arc)) for arc in arcs}
        #self.positive_arcs = {arc: self.model.NewBoolVar("positive"str(arc)) for arc in arcs}
        self.head_order = {}
        for i, v_i in enumerate(self.graph.vertices):
            # Constraint over all vertices: least 2k-1 connections between incident edges
            incident_vertices = np.nonzero(self.graph.ad_matrix[i])[0]
            for j in incident_vertices:
                self.head_order[i, j] = self.model.NewIntVar(0, len(incident_vertices)-1, "head_order"+str((i, j)))

        return self.time, diffs, absolutes, max_time

    def _add_constraints(self, arcs, degrees, times, diffs, absolutes, max_time):
        # Add Constraints
        #for time in times:
            #self.model.Add(times[time] <= max_time)
        ub = int(2 * self.multiplier * (1 + np.ceil(np.log2(self.graph.vert_amount))))
        #self.model.AddAllDifferent([times[key] for key in times])
        for arc in arcs:
            vi_to_vp = (min(arc[:2]), max(arc[:2]))
            vi_to_vk = (min(arc[0], arc[2]), max(arc[0], arc[2])) #(arc[0], arc[2])
            self.model.Add(times[vi_to_vk] - times[vi_to_vp] > 0).OnlyEnforceIf(self.edges[arc])#== diffs[arc])
            
            #self.model.AddAbsEquality(absolutes[arc], diffs[arc])
            #self.model.Add(absolutes[arc] >= 1)
            #self.model.Add(absolutes[arc] == absolutes[(arc[0], arc[2], arc[1])])

            #self.model.Add(diffs[arc] > 0).OnlyEnforceIf(self.above_zero[arc])# - ub*self.above_zero[arc], -ub, 0)
            #self.model.Add(diffs[arc] < 0).OnlyEnforceIf(self.above_zero[arc].Not())
            self.model.Add(self.head_order[(arc[0], arc[1])] - self.head_order[(arc[0], arc[2])] == 1).OnlyEnforceIf(self.edges[arc])
            #self.model.AddAbsEquality(self.head_abs[arc], self.head_diff[arc])
            #self.model.Add(self.head_order[(arc[0], arc[1])] > self.head_order[(arc[0], arc[2])]).OnlyEnforceIf(self.above_zero[arc])# - ub*self.above_zero[arc] < 0)
            #self.model.Add(self.head_order[(arc[0], arc[1])] < self.head_order[(arc[0], arc[2])]).OnlyEnforceIf(self.above_zero[arc].Not())
            #self.model.AddLinearConstraint(self.head_abs[arc] + self.pot_edge[arc] - 2, 0, ub)
            #self.model = cp_model.CpModel()
            #self.model.AddBoolOr([self.above_zero[arc].Not(), self.pot_edge[arc].Not(), self.edges[arc]])
            
            
            #self.model.AddLinearConstraint(diffs[arc] +  ub * self.edges[arc], -ub, ub)#.OnlyEnforceIf(self.edges[arc])
        length = len(self.graph.vertices)
        for i, v_i in enumerate(self.graph.vertices):
            # Constraint over all vertices: least 2k-1 connections between incident edges
            incident_vertices = np.nonzero(self.graph.ad_matrix[i])[0]
            self.model.AddAllDifferent([self.head_order[i,j] for j in incident_vertices])
            self.model.Add(
                sum([self.edges[arc] for arc in arcs if arc[0] == i]) == len(incident_vertices)-1
                )


        #for i in range(length):
        #    self.model.Add(sum([self.edges[arc] for arc in arcs if arc[0] == i]) == length-1)

    def _setObj(self):
        #super()._setObj()
        self.model.Minimize(sum([self.degrees[key] * self.edges[key] for key in self.edges]))

    def _calculate_degrees(self) -> (List[tuple], Dict[tuple, float]):
        keys, degrees = super()._calculate_degrees()
        rev_keys = [(key[0], key[2], key[1]) for key in keys]
        new_degrees = {rev_key: degrees[key] for rev_key, key in zip(rev_keys, keys)}
        new_degrees.update(degrees)
        return keys+rev_keys, new_degrees
