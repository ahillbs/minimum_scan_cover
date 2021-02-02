from typing import Optional, Union, List, Tuple
import numpy as np

import gurobipy as grb

from utils import Multidict, get_angles, get_angle, callback_rerouter, get_array_greater_zero, calculate_times, is_debug_env
from solver import Solver
from database import Graph, AngularGraphSolution

class AngularGraphScanMakespanAbsolute(Solver):
    solution_type = "makespan"
    def __init__(self, time_limit=900, **kwargs):
        self.graph = None
        self.model = None
        super().__init__(kwargs.pop("params", {"TimeLimit": time_limit}))
    
    def is_multicore(self):
        return True
    
    def solve(self, graph, **kwargs):
        return self.build_ip_and_optimize(graph, **kwargs)

    def build_ip_and_optimize(self, graph: Graph, initial_heading: Optional[Union[list, np.array]] = None, start_solution: Optional[dict] = None, callbacks: Optional[Union[Multidict, dict]] = None, **kwargs):
        try:
            error_message = None
            runtime = 0
            is_optimal = False
            times = None
            try:
                self.graph = graph
                self.model = grb.Model()
                for param in self.params:
                    self.model.setParam(param, self.params[param])
                if "time_limit" in kwargs:
                    self.model.setParam("TimeLimit", kwargs["time_limit"])

                self._add_callbacks(callbacks)

                arcs, degrees = self._calculate_degrees()
                time, diffs, absolutes, max_time = self._add_variables(arcs)
                self._add_constraints(arcs, degrees, time, diffs, absolutes, max_time)

                self.model.setObjective(max_time, grb.GRB.MINIMIZE)

                self._add_initial_heading(time, initial_heading)

                self._add_pre_solution(time, graph, start_solution)

                self.model.update()
                self.model.optimize(callback_rerouter)
                #for v in time:
                    #if v.x != 0 or "time" in v.varName:
                    #print('%s %g' % (time[v].varName, time[v].x))
                #for v in absolutes:
                #    if absolutes[v].x >= 180:
                #        print('%s %g' % (absolutes[v].varName, absolutes[v].x))
                #print('%s %g' % (max_time.varName, max_time.x))
                runtime = self.model.Runtime
                times = {t: time[t].x for t in time}
                
                is_optimal = self.model.Status == grb.GRB.OPTIMAL
                
            except Exception as e:
                error_message = str(e)
                if is_debug_env():
                    raise e
            sol = AngularGraphSolution(self.graph,
                                        runtime=runtime,
                                        solver=self.__class__.__name__,
                                        solution_type="makespan",
                                        is_optimal=is_optimal,
                                        times=times,
                                        error_message=error_message)
            return sol

        except Exception as exception:
            raise exception
        finally:
            self._cleanup()
        return None

    def _calculate_degrees(self) -> (grb.tuplelist, grb.tupledict):
        # Calculate degrees
        degrees = grb.tupledict()
        arcs = grb.tuplelist()
        for i in range(len(self.graph.vertices)):
            arcs_i, degrees_i = self._get_angles(i)
            if arcs_i:
                degrees.update(degrees_i)
                arcs.extend(arcs_i)
        return arcs, degrees

    def _get_angles(self, index) -> (grb.tuplelist, grb.tupledict):
        v_a = np.array(self.graph.vertices[index])
        ad_indexes = [j for j in range(len(self.graph.vertices)) if self.graph.ad_matrix[index, j] > 0]
        vert_arr = np.array([self.graph.vertices[i] for i in ad_indexes])
        l = len(vert_arr)
        degrees = get_angles(v_a, vert_arr)
        tuple_list = [((index, ad_indexes[i], ad_indexes[j]), degrees[i, j])
             for i in range(l) for j in range(l) if i != j]
        # Correct entries with NaN
        for i in range(len(tuple_list)):
            if np.isnan(tuple_list[i][-1]):
                tuple_list[i] = (tuple_list[i][0], 0)
        arcs, multidict = grb.multidict(tuple_list) if tuple_list else (None, None)
        return arcs, multidict

    def _add_variables(self, arcs) -> (grb.Var, grb.Var, grb.Var, grb.Var):
        # Add variables
        time = self.model.addVars(
            [
                (i, j)
                for i in range(len(self.graph.vertices))
                for j in range(len(self.graph.vertices))
                if self.graph.ad_matrix[i, j] > 0
            ],
            name="time")
        diffs = self.model.addVars(arcs, name="diffs", lb=-grb.GRB.INFINITY)
        absolutes = self.model.addVars(arcs, name="abs")
        max_time = self.model.addVar(name="Max_t")

        return time, diffs, absolutes, max_time

    def _add_constraints(self, arcs, degrees, times, diffs, absolutes, max_time):
        # Add Constraints
        for time in times:
            if time[0] < time[1]:
                rev = (time[1], time[0])
                self.model.addConstr(times[time] == times[rev], name="time_eq_constr")
                self.model.addConstr(times[time] <= max_time)
                self.model.addConstr(times[rev] <= max_time)
        self.model.update()
        self.model.getVars()

        for arc in arcs:
            vi_to_vp = (arc[0], arc[1])
            vi_to_vk = (arc[0], arc[2])
            self.model.addConstr(times[vi_to_vp] - times[vi_to_vk] == diffs[arc], name="diff_constr")
            self.model.addGenConstrAbs(absolutes[arc], diffs[arc], "absolute_gen_constr")
            self.model.addConstr(absolutes[arc] >= degrees[arc], name="degree_constraint")

    def _add_initial_heading(self, times, initial_heading):
        if get_array_greater_zero(initial_heading):
            # create degrees for all initial headings
            degrees = grb.tupledict()
            arcs = grb.tuplelist()
            for index in range(len(self.graph.vertices)):
                v_a = np.array(self.graph.vertices[index])
                ad_indexes = [j for j in range(len(self.graph.vertices)) if self.graph.ad_matrix[index, j] > 0]
                vert_arr = np.array([self.graph.vertices[i] for i in ad_indexes])
                l = len(vert_arr)
                degrees = np.array([get_angle(v_a, vert, initial_heading[index]) for vert in vert_arr])
                tuple_list = [((index, ad_indexes[i]), degrees[i])
                              for i in range(l)]
                # Correct entries with NaN
                for i in range(len(tuple_list)):
                    if tuple_list[i] == np.NaN:
                        tuple_list[i][-1] = 0
                
                arcs_i, multidict_i = grb.multidict(tuple_list) if tuple_list else (None, None)
                degrees.update(multidict_i)
                arcs.extend(arcs_i)
            
            for arc in arcs:
                self.model.addConstr(times[arc] >= degrees[arc], name="degree_constraint_init")

    def _add_pre_solution(self, times, graph, start_solution: Union[AngularGraphSolution, List[Tuple[int,int]]]):
        if start_solution:
            if not isinstance(start_solution, AngularGraphSolution):
                start_times = calculate_times(start_solution, graph)
            else:
                start_times = start_solution.times
            for key in times:
                times[key].Start = start_times[key]
            
                

    def _cleanup(self):
        callback_rerouter.inner_callbacks = None
        self.model = None
        self.graph = None

    def _add_callbacks(self, callbacks: Optional[Union[Multidict, dict]] = None):
        # Add callbacks
        own_callbacks = Multidict()
        if callbacks:
            own_callbacks.update(callbacks)
        callback_rerouter.inner_callbacks = own_callbacks

class AngularGraphScanMakespanAbsoluteReduced(AngularGraphScanMakespanAbsolute):

    def _get_angles(self, index) -> (grb.tuplelist, grb.tupledict):
        v_a = np.array(self.graph.vertices[index])
        ad_indexes = [j for j in range(len(self.graph.vertices)) if self.graph.ad_matrix[index, j] > 0]
        vert_arr = np.array([self.graph.vertices[i] for i in ad_indexes])
        l = len(vert_arr)
        degrees = get_angles(v_a, vert_arr)
        tuple_list = [((index, ad_indexes[i], ad_indexes[j]), degrees[i, j])
             for i in range(l) for j in range(i+1, l)]
        arcs, multidict = grb.multidict(tuple_list) if tuple_list else (None, None)
        return arcs, multidict

    def _add_variables(self, arcs) -> (grb.Var, grb.Var, grb.Var, grb.Var):
        # Add variables
        time = self.model.addVars(
            [
                (i, j)
                for i in range(len(self.graph.vertices))
                for j in range(len(self.graph.vertices))
                if self.graph.ad_matrix[i, j] > 0 and i < j
            ],
            name="time")
        diffs = self.model.addVars(arcs, name="diffs", lb=-grb.GRB.INFINITY)
        absolutes = self.model.addVars(arcs, name="abs")
        max_time = self.model.addVar(name="Max_t")
        return time, diffs, absolutes, max_time

    def _add_constraints(self, arcs, degrees, times, diffs, absolutes, max_time):
        # Add Constraints
        self.model.addConstrs(times[time] <= max_time for time in times)

        for arc in arcs:
            vi_to_vp = (min(arc[:2]), max(arc[:2]))
            vi_to_vk = (min(arc[0], arc[2]), max(arc[0], arc[2])) #(arc[0], arc[2])
            self.model.addConstr(times[vi_to_vp] - times[vi_to_vk] == diffs[arc], name="diff_constr")
            self.model.addGenConstrAbs(absolutes[arc], diffs[arc], "absolute_gen_constr")
            self.model.addConstr(absolutes[arc] >= degrees[arc], name="degree_constraint")
    
