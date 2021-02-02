import numpy as np
import time
import math
from . import Solver
from database import Graph, AngularGraphSolution
from utils import get_vertex_sectors, convert_graph_to_angular_abstract_graph, get_dep_graph, calculate_order, get_lower_bounds, CircularDependencyException

class AngularBipartiteMinSumSolver(Solver):
    solution_type = "min_sum"

    def __init__(self, **kwargs):
        super().__init__(kwargs.pop("params", None))
    
    def is_multicore(self):
        return False

    def solve(self, graph: Graph, **kwargs):
        # count how many vertices have largest angle north or south
        # Maybe choose which set get which starting position accordingly
        order = None
        start_time = time.time()
        error_message = None
        try:
            colors = kwargs.pop("colors", None)
            if colors is None:
                colors = self._calc_subsets(graph)
            colors = self._check_bipartite(colors, graph)
            sectors_v = {i:
                get_vertex_sectors(
                    graph.vertices[i],
                    graph.vertices[np.nonzero(graph.ad_matrix[i])],
                    start_north=bool(colors[i])
                    )
                for i in range(graph.vert_amount)
            }

            edge_vert, abs_graph = self._create_abs_graph_and_edge_translation(graph)
            
            used_edges = self._calculated_used_edges(sectors_v, graph, edge_vert)
            
            dep_graph = get_dep_graph(used_edges, abs_graph)
            order = [tuple(abs_graph.vertices[i]) for i in calculate_order(dep_graph, calculate_circle_dep=True)]
        except NotImplementedError as e:
            error_message = str(e)
            raise e
        # For small instances it could happen that two vertices have connections only to each other.
        # These edges will not be scheduled. Since they do not depend on any other edge, we just add them at the end.
        order = self._add_single_edges(order, graph)
        sol = AngularGraphSolution(
            graph,
            time.time() - start_time,
            self.__class__.__name__,
            self.solution_type,
            is_optimal=False,
            error_message=error_message,
            order=order
        )
        return sol

    def _add_single_edges(self, order, graph):
        # For small instances it could happen that two vertices have connections only to each other.
        # These edges will not be scheduled. Since they do not depend on any other edge, we just add them at the end.
        if (not order and graph.edge_amount > 0) or len(order) != graph.edge_amount:
            if order is None:
                order = []
            for edge in graph.edges:
                e_t = tuple(edge)
                if e_t not in order:
                    order.append(e_t)
        return order

    def _calculated_used_edges(self, sectors_v, graph, edge_vert):
        used_edges = []
        for vertex_key in sectors_v:
            sectors_info = sectors_v[vertex_key]
            if len(sectors_info) == 1:
                continue
            non_zeros = np.nonzero(graph.ad_matrix[vertex_key])[0]
            for from_vert, to_vert, angle in sectors_info[:-1]:
                used_edges.append(edge_vert[vertex_key][non_zeros[from_vert], non_zeros[to_vert]])
        return used_edges

    def _create_abs_graph_and_edge_translation(self, graph):
        tripel_edges, abs_graph = convert_graph_to_angular_abstract_graph(graph, simple_graph=False, return_tripel_edges=True)
        edge_vert = {i: {} for i in range(graph.vert_amount)}
        for edge in tripel_edges:
            edge_vert[edge[0]][edge[1:]] = tripel_edges[edge]
        return edge_vert, abs_graph

    def _check_bipartite(self, colors, graph):
        
        # Make sure V_1 and V_2 are bipartite partitions
        assert len(colors) == graph.vert_amount
        assert np.alltrue([colors[i] != colors[j] for i,j in graph.edges])
        return colors

    def _calc_subsets(self, graph: Graph):
        raise NotImplementedError()

class AngularBipartiteMakespanSolver(AngularBipartiteMinSumSolver):
    solution_type = "makespan"

    def solve(self, graph: Graph, **kwargs):
        # count how many vertices have largest angle north or south
        # Maybe choose which set get which starting position accordingly
        order = None
        start_time = time.time()
        error_message = None
        try:
            colors = kwargs.pop("colors", None)
            if colors is None:
                colors = self._calc_subsets(graph)
            colors = self._check_bipartite(colors, graph)
            
            force_strategy = kwargs.pop("strategy", None)
            debug = kwargs.pop("debug", False)

            # MLSSC (LocalMinSum) lower bound is exactly the value for the cone described in the paper.
            mlssc_lb = max(get_lower_bounds(graph))
            if mlssc_lb >= 90 or force_strategy == 1: # If we have a LB >= 90, we can use the same algorithm as before
                return super().solve(graph, colors=colors)
            else: # LB < 90 needs another strategy with the sector splits. See paper for more details
                if mlssc_lb > 0:
                    s = math.floor(180/mlssc_lb) # The minium amount of lines where cone > lb
                    cone = np.radians(360/(2*s))
                    ad_angles_v = {i: np.array(
                            [i if i > 0 else 2*np.pi + i for i in np.arctan2(
                            *(graph.vertices[np.nonzero(graph.ad_matrix[i])] - graph.vertices[i]).T)]
                            )
                        for i in range(graph.vert_amount)
                    }
                    sectors_v = {i:
                        {}
                        for i in range(graph.vert_amount)
                    }
                    for key in ad_angles_v:
                        angles = ad_angles_v[key] % (2*np.pi)
                        non_zeros = np.nonzero(graph.ad_matrix[key])[0]
                        argsorted = np.argsort(angles)
                        for i in argsorted:
                            angle = angles[i]
                            current_sector = math.floor(angle/cone) % (2*s)# if colors[i] else (math.ceil(angle/cone)-1) % (2*s)
                            if current_sector not in sectors_v[key]:
                                sectors_v[key][current_sector] = [non_zeros[i]]
                            else:
                                sectors_v[key][current_sector].append(non_zeros[i])
                    # Check if every vertex has at most two sectors
                    for key in sectors_v:
                        assert len(sectors_v[key]) <= 2, f"Sector build is wrong! Vertex index {key} has {len(sectors_v[key])} sectors"
                    # Give sectors directions
                    sector_directions_v = {i:
                        {j: 1-j%2 if (s%2 and colors[i]) else j%2 for j in sectors_v[i]}
                        for i in range(graph.vert_amount)
                    }
                    edge_vert, abs_graph = self._create_abs_graph_and_edge_translation(graph)
                    used_edges = []
                    sector_orders = {}
                    for key in sectors_v:
                        sector_order = [-1, -1]
                        for j in sector_directions_v[key]:
                            #if s%2 == 0:
                            #    sector_order[j%2] = j
                            #else:
                            inner_key = 1-j%2 if (s%2 and colors[key]) else j%2
                            sector_order[inner_key] = j
                        sector_orders[key] = sector_order

                    for key in sector_orders:
                        sector_order = sector_orders[key]
                        from_vert = None
                        for o in sector_order:
                            if o == -1:
                                continue
                            sectors_info = sectors_v[key][o]
                            
                            non_zeros = np.nonzero(graph.ad_matrix[key])[0]
                            
                            it = iter(sectors_info) if sector_directions_v[key][o] == 0 else reversed(sectors_info)
                            for to_vert in it:
                                if from_vert is not None:
                                    used_edges.append(edge_vert[key][from_vert, to_vert])
                                    if debug:
                                        print("Used edge:", key, from_vert, to_vert)
                                from_vert = to_vert
                    dep_graph = get_dep_graph(used_edges, abs_graph)
                    
                    try:
                        order = [tuple(abs_graph.vertices[i]) for i in calculate_order(dep_graph, calculate_circle_dep=True)]
                    except CircularDependencyException as c_e:
                        if debug:
                            translated_cycle = [abs_graph.vertices[i.value].tolist() for i in c_e.circle_nodes]
                            print("Cycle:", translated_cycle)
                        raise c_e
        except NotImplementedError as e:
            error_message = str(e)
            raise e
        order = self._add_single_edges(order, graph)
        sol = AngularGraphSolution(
            graph,
            time.time() - start_time,
            self.__class__.__name__,
            self.solution_type,
            is_optimal=False,
            error_message=error_message,
            order=order
        )
        assert sol.makespan <= mlssc_lb*4.5
        return sol