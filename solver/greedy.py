import time
import numpy as np
from database import Graph, AngularGraphSolution
from . import Solver
from utils import get_angle, get_graph_angles, calculate_times

class AngularMinSumGreedySolver(Solver):
    solution_type = "min_sum"
    def __init__(self, **kwargs):
        self.no_sol_return = kwargs.pop("no_sol_return", None)
        super().__init__(**kwargs)

    def is_multicore(self):
        return False

    def solve(self, graph: Graph, no_sol_return=None, **kwargs):
        if no_sol_return is None:
            no_sol_return = self.no_sol_return if self.no_sol_return is not None else False

        start_time = time.time()
        # Process kwargs
        upper_bound = kwargs.pop("ub", None)
        order = kwargs.pop("presolved", [])
        order = order.copy()
        headings = kwargs.pop("headings", [[] for head in graph.vertices])
        assert isinstance(headings, list)
        headings = headings.copy()
        for i, heading in enumerate(headings):
            headings[i] = heading.copy()
        remaining_edges = kwargs.pop("remaining", graph.edges)
        if isinstance(remaining_edges, np.ndarray):
            remaining_edges = remaining_edges.tolist()
        remaining_edges = remaining_edges.copy()
        # If one is empty, the other also needs to be emtpy
        assert order or len(remaining_edges) == graph.edge_amount
        assert len(remaining_edges) != graph.edge_amount or not order
        angles = graph.costs
        if angles is None:
            angles = get_graph_angles(graph)
        overall_cost = self._calc_pre_cost(headings, angles)
        while remaining_edges and (upper_bound is None or upper_bound > overall_cost):
            candidate_edge, candidate_cost = self._get_candidate(order, remaining_edges, headings, angles)

            self._set_new_headings(candidate_edge, headings)

            to_remove = self._get_edge(remaining_edges, candidate_edge)
            remaining_edges.remove(to_remove)
            order.append(to_remove)
            overall_cost += candidate_cost
        if no_sol_return:
            return order, overall_cost
        sol = AngularGraphSolution(
            graph,
            time.time() - start_time,
            self.__class__.__name__,
            self.solution_type,
            is_optimal=False,
            order=order
        )
        return sol

    def _get_edge(self, remaining_edges, candidate_edge):
        to_remove = None
        for edge in remaining_edges:
            if edge[0] == candidate_edge[0] and edge[1] == candidate_edge[1]:
                to_remove = edge
        return to_remove

    def _set_new_headings(self, candidate_edge, headings):
        for i, vert in enumerate(candidate_edge):
            other = candidate_edge[1 - i]
            headings[vert].append(other)

    def _get_candidate(self, order, remaining_edges, headings, angles):
        candidate_edge = None
        candidate_cost = None
        for edge in remaining_edges:
            cost = 0
            for i, vert in enumerate(edge):
                other = edge[1- i]
                if headings[vert]:
                    cost += angles[vert][(headings[vert][-1], other)]
            if candidate_cost is None or cost < candidate_cost:
                candidate_edge = edge
                candidate_cost = cost
        return candidate_edge, candidate_cost

    def _calc_pre_cost(self, headings, angles):
        overall_cost = 0
        for i, heading in enumerate(headings):
            prev = None
            for vertex in heading:
                if prev is not None:
                    overall_cost += angles[i][(prev, vertex)]
                prev = vertex
        return overall_cost

    

class AngularMinSumGreedyConnectedSolver(AngularMinSumGreedySolver):
    def _get_candidate(self, order, remaining_edges, headings, angles):
        candidate_edge = None
        candidate_cost = None
        for edge in remaining_edges:
            cost = 0
            for i, vert in enumerate(edge):
                other = edge[1- i]
                if headings[vert]:
                    cost += angles[vert][(headings[vert][-1], other)]
            if candidate_cost is None or (0 < cost < candidate_cost) or (candidate_cost == 0 and cost > 0):
                candidate_edge = edge
                candidate_cost = cost
        return candidate_edge, candidate_cost

class AngularLocalMinSumGreedySolver(AngularMinSumGreedySolver):
    solution_type = "local_min_sum"

    def _get_candidate(self, order, remaining_edges, headings, angles):
        candidate_edge = None
        candidate_cost = None
        heading_sums = [0 for i in range(len(headings))]
        
        for i, heading in enumerate(headings):
            prev = None
            for head in heading:
                if prev is not None:
                    heading_sums[i] += angles[i][(prev, head)]
                prev = head
        for edge in remaining_edges:
            cost = heading_sums.copy()
            for i, vert in enumerate(edge):
                other = edge[1- i]
                if headings[vert]:
                    cost[vert] = heading_sums[vert] + angles[vert][(headings[vert][-1], other)]
            if candidate_cost is None or (max(cost) < candidate_cost) or\
                (candidate_cost == 0 and max(cost) > 0):
                candidate_edge = edge
                candidate_cost = max(cost)
        return candidate_edge, candidate_cost

class AngularMakespanGreedySolver(AngularMinSumGreedySolver):
    solution_type = "makespan"

    def _get_candidate(self, order, remaining_edges, headings, angles):
        candidate_edge = None
        candidate_cost = None

        times = calculate_times(order, angles=angles)

        for edge in remaining_edges:
            new_order = order.copy()
            new_order.append(edge)
            max_time = 0
            vertex_set = set(edge)
            for index in vertex_set:
                other_index = vertex_set.difference([index]).pop()
                curr_prev = headings[index][-1] if headings[index] else None
                if curr_prev is not None:
                    angle = angles[index][(curr_prev, other_index)]
                    sorted_key = tuple(sorted([index, curr_prev]))
                    max_time = max([max_time, times[sorted_key] + angle])
            
            if ((candidate_cost is None) or (max_time < candidate_cost)) or ((candidate_cost == 0) and (max_time > 0)):
                candidate_edge = edge
                candidate_cost = max_time
        return candidate_edge, candidate_cost