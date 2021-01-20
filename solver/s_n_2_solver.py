from typing import List, Dict, Tuple, Union
import itertools
import time
import numpy as np

from utils import calculate_times
from utils.dependency_graph import DependencyNode, DependencyGraph, CircularDependencyException, DisconnectedDependencyGraphException, SolveOrder, calculate_order
from . import Solver

from database import Graph, AngularGraphSolution

class PointSweep():
    """Contains sweep direction and if the start/end is alternated
    """
    # Settings
    COUNTER_CLOCKWISE = -1
    CLOCKWISE = 1
    ALTERNATE_START = -1
    STANDARD = 0
    ALTERNATE_END = 1

    def __init__(self, direction: int, start_end_type: int, vertex_index):
        # Check that direction and start_end is set correctly
        sweep_types = [PointSweep.ALTERNATE_START, PointSweep.ALTERNATE_END, PointSweep.STANDARD]
        directions = [PointSweep.CLOCKWISE, PointSweep.COUNTER_CLOCKWISE]
        assert direction in directions, "Direction must be clockwise or couter clockwise."
        assert start_end_type in sweep_types, "start_end_type must be alternate start/end or standard."

        self.direction = direction
        self.type = start_end_type
        self.vertex_index = vertex_index

class GraphN2Solver(Solver):
    def __init__(self, *args, brute_force=True, all_minimal_solutions=False, **kwargs):
        self.brute_force = brute_force
        self.all_minimal_solutions = all_minimal_solutions
        self.max_time = kwargs.pop("max_time", 900)
        super().__init__()
    
    def is_multicore(self):
        return False

    def solve(self, graph: Graph, **kwargs) -> Union[AngularGraphSolution, List[AngularGraphSolution]]:
        # ToDo: Check if graph is G(n,2) type
        n = len(graph.vertices)
        solutions = []
        start_time = time.time()
        k = kwargs.pop("start_k", 1)
        while (not solutions and
               k < len(graph.vertices) and
               (time.time() - start_time) < self.max_time):
            sweep_types = self._generate_possible_start_ends(graph, k)
            for sweep_type in sweep_types:
                orientations = self._calculate_orientations(graph, sweep_type)
                rev_orientations = self._calculate_orientations(graph, sweep_type, reverse_start=True)
                try:
                    sol = self._calc_solution(graph, orientations, n, k, start_time)
                    if not self.all_minimal_solutions:
                        return sol
                    solutions.append(sol)
                except CircularDependencyException:
                    pass # We do not care for the not solveable
                try:
                    sol = self._calc_solution(graph, rev_orientations, n, k, start_time)
                    if not self.all_minimal_solutions:
                        return sol
                    solutions.append(sol)
                except CircularDependencyException:
                    pass # We do not care for the not solveable
                
            k += 1
        return solutions

    def _calc_solution(self, graph, orientations, n, k, start_time):
        dep_graph = self._calculate_dependecy_graph(graph, orientations)
        order = self._calculate_order(dep_graph)
        times = calculate_times(order, graph)

        obj_val = (n-2)*180 + (360/n)*k
        sol = AngularGraphSolution(
            graph,
            time.time() - start_time,
            self.__class__.__name__,
            "min_sum",
            times=times,
            obj_val=obj_val,
            sweep_order=order
        )
        return sol

    def _generate_possible_start_ends(self, graph: Graph, non_standard_amount) -> np.array:
        # ToDo: Make smart decisions, which combinations are possible for the given problem
        # For now just try them all to gain insight
        if self.brute_force:
            # Tells where the non standard are and which alternate start/end is used
            alt_positions = itertools.combinations(range(1, len(graph.vertices)), non_standard_amount-1)
            configs = [s for s in itertools.product(*[[-1, 1] for i in range(non_standard_amount-1)])]
            possible_start_ends = []
            for pos in alt_positions:
                for config in configs:
                    # The first entry is fixed with 1 and -1
                    arr = np.zeros(len(graph.vertices))
                    arr[0] = 1
                    arr[list(pos)] = config
                    possible_start_ends.append(arr)

                    arr2 = np.zeros(len(graph.vertices))
                    arr2[0] = -1
                    arr2[list(pos)] = config
                    possible_start_ends.append(arr2)
            return possible_start_ends
        raise NotImplementedError("Only brute force is implemented at the moment")

    def _calculate_orientations(self, graph: Graph, start_ends, reverse_start=False) -> List[PointSweep]:
        orientations = []
        vert_len = len(graph.vertices)
        assert len(start_ends) == vert_len
        start_direction = 1 if not reverse_start else -1
        last_p_s = None
        for i in range(vert_len):
            if last_p_s:
                if start_ends[i] != PointSweep.STANDARD:
                    last_p_s = PointSweep(last_p_s.direction, start_ends[i], i)
                else:
                    last_p_s = PointSweep(last_p_s.direction * -1, start_ends[i], i)
            else:
                last_p_s = PointSweep(start_direction, start_ends[i], i)

            orientations.append(last_p_s)

        return orientations

    def _calculate_dependecy_graph(self, graph: Graph, orientations: List[PointSweep]) -> DependencyGraph:
        edge_set = {tuple(sorted(edge)) for edge in graph.edges}
        dependency_struct = {edge: DependencyNode(edge) for edge in edge_set}
        modulo = len(orientations)
        for point_sweep in orientations:
            index = point_sweep.vertex_index
            # Get adjacent vertices
            cur_ad = graph.ad_matrix[index]
            prev_edge = None
            ad_index = index + point_sweep.direction
            index_exception = None
            # For alternative starts and ends, we make exceptions for the order
            if point_sweep.type == PointSweep.ALTERNATE_START:
                index_exception = (index + 2*point_sweep.direction) % modulo
                prev_edge = tuple(sorted((index, index_exception)))

            if point_sweep.type == PointSweep.ALTERNATE_END:
                index_exception = (index - 2*point_sweep.direction) % modulo
            # Safety check that the exception index has an edge with the current index
            if index_exception:
                assert cur_ad[index_exception] > 0

            while ad_index % modulo != index:
                if ad_index % modulo != index_exception and cur_ad[ad_index % modulo] > 0:
                    edge = tuple(sorted((ad_index % modulo, index)))
                    if prev_edge:
                        dependency_struct[edge].add_dependency(dependency_struct[prev_edge])
                    prev_edge = edge
                ad_index = ad_index + point_sweep.direction

            if point_sweep.type == PointSweep.ALTERNATE_END:
                edge = tuple(sorted((index_exception, index)))
                if prev_edge:
                    dependency_struct[edge].add_dependency(dependency_struct[prev_edge])
                prev_edge = edge
        return dependency_struct

    def _calculate_order(self,
                         dependency_graph: DependencyGraph
                        ) -> SolveOrder:
        return calculate_order(dependency_graph)
