import numpy as np
import itertools

import gurobipy as grb
from ortools.sat.python import cp_model

from . import Solver
from database import Graph

class Coloring_CP_Solver(Solver):
    def __init__(self, **kwargs):
        # Settings if needed
        super().__init__()
    def is_multicore(self):
        return True

    def solve(self, graph: Graph, **kwargs):
        model = cp_model.CpModel()
        colors, max_color = self._add_variables(graph, model)
        self._add_constraints(graph, model, colors, max_color)
        self._setObj(model, max_color)
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 6
        status = solver.Solve(model)
        if status in [cp_model.FEASIBLE, cp_model.OPTIMAL]:
            values = np.array([solver.Value(colors[key]) for key in colors])
            return values
        raise Exception("Somehow not feasable solution")

    def _setObj(self, model, max_color):
        model.Minimize(max_color)

    def _add_variables(self, graph: Graph, model: cp_model.CpModel):
        ub = graph.vert_amount
        vertex_colors = {
            i: model.NewIntVar(0, ub, "color"+str(i)) for i in range(graph.vert_amount)
            }
        max_color = model.NewIntVar(0, ub, "max_color")
        return vertex_colors, max_color

    def _add_constraints(self, graph: Graph, model: cp_model.CpModel, colors, max_color):
        # Add Constraints
        for edge in graph.edges:
            model.AddAllDifferent([colors[i] for i in edge])
        
        model.AddMaxEquality(max_color, [colors[i] for i in colors])

class Coloring_IP_Solver(Solver):
    def __init__(self, **kwargs):
        # Settings if needed
        self.standard_params = kwargs.pop('standard_params', {'TimeLimit': 90})
        super().__init__()

    def is_multicore(self):
        return True

    def solve(self, graph: Graph, **kwargs):
        model = grb.Model()
        # First set standard params
        for param_key in self.standard_params:
            model.setParam(param_key, self.standard_params[param_key])
        # Next add/override special params
        try:
            new_params = kwargs.pop('params')
            for param_key in new_params:
                model.setParam(param_key, new_params[param_key])
        except KeyError:
            pass # Happens if params not found in kwargs. Do not add params then

        vertices = model.addVars(
            [i for i in range(len(graph.vertices))],
            name="Vertex_color", vtype=grb.GRB.INTEGER, lb=1)
        product = [i for i in itertools.combinations(vertices, 2)]
        product = np.array(np.nonzero(np.triu(graph.ad_matrix))).T
        vertices_diffs = model.addVars([(u, v) for u, v in product], name="Vertex_diff", lb=-grb.GRB.INFINITY)
        vertices_abs = model.addVars([(u, v) for u, v in product], name="Vertex_abs", lb=1)
        

        model.addConstrs((vertices[v] - vertices[w] == vertices_diffs[v, w] for v, w in product), name="Unequal constr")
        #model.addConstrs((vertices_diffs[v, w] == grb.abs_(vertices_abs[v, w]) for v, w in product), name="Gen abs constr")        
        for v, w in product:
        #    # Quick and dirty
        #    if v > w:
        #        continue
        #    model.addConstr(vertices[v] - vertices[w] == vertices_diffs[v, w], name="Unequal constr")
            model.addGenConstrAbs(vertices_abs[v, w], vertices_diffs[v, w], name="Gen abs constr")
        
        c_max = model.addVar(name="max_color", vtype=grb.GRB.INTEGER)
        model.addGenConstrMax(c_max, [vertices[v] for v in vertices], 1.0, "c_max_constr")
        #model.addConstrs((c_max - vertices[v] >= 0 for v in vertices), name="c_max_constr")
        #for v in vertices:
        #    model.addConstr(v <= c_max, name="c_max constr")
        exp = 0
        exp += c_max
        model.setObjective(exp, grb.GRB.MINIMIZE)
        # Set start solution if possible
        try:
            start = kwargs.pop("start_solution")
            for i in range(len(start)):
                vertices[i].Start = start[i]
        except KeyError:
            pass
        model.optimize()

        try:
            if kwargs.pop("with_status"):
                return ([int(vertices[v].x) for v in vertices], self.translate[model.status])
        except KeyError:
            pass

        return [int(vertices[v].x) for v in vertices]


class Oldschool_Coloring_IP_Solver(Solver):
    def __init__(self, **kwargs):
        # Settings if needed
        self.standard_params = kwargs.pop('standard_params', {'TimeLimit': 90})
        super().__init__()

    def is_multicore(self):
        return True

    def solve(self, graph: Graph, **kwargs):
        model = grb.Model()
        # First set standard params
        for param_key in self.standard_params:
            model.setParam(param_key, self.standard_params[param_key])
        # Next add/override special params
        try:
            new_params = kwargs.pop('params')
            for param_key in new_params:
                model.setParam(param_key, new_params[param_key])
        except KeyError:
            pass # Happens if params not found in kwargs. Do not add params then

        vertices = model.addVars(
            [(i,j) for i in range(graph.vert_amount) for j in range(graph.vert_amount)],
            name="Vertex_color", vtype=grb.GRB.BINARY)
        colors = model.addVars(
            [i for i in range(len(graph.vertices))],
            name="Colors", vtype=grb.GRB.BINARY)
        # Constraints
        for c in colors:
            model.addConstrs([vertices[i, c] + vertices[j, c] <= colors[c] for i, j in graph.edges])
        # Set objective
        model.setObjective(colors.sum(), grb.GRB.MINIMIZE)
        # Set start solution if possible
        try:
            start = kwargs.pop("start_solution")
            for i in range(len(start)):
                vertices[i, start[i]].Start = 1
        except KeyError:
            pass
        model.optimize()
        # Just to make sure everything is ordered the same way the vertices are ordered
        color_dict = {v: int(c) for v,c in vertices if vertices[v,c] == 1}
        return [color_dict[v] for v in sorted(color_dict)]
        