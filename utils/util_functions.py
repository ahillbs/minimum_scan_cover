import sys
from typing import Union
import numpy as np
from .multidict import Multidict
from .dependency_graph import DependencyNode

def get_angles(v: np.ndarray, ad_vertices: np.ndarray, in_degree=True) -> np.ndarray:
    """Get all angle degrees between ad_vertices for point v
    
    Arguments:
        v {np.ndarray} -- Degree source
        ad_vertices {np.ndarray} -- All other points for which
                                    the degrees between them will be calculated
    
    Returns:
        np.ndarray -- degrees from v to all verticed in ad_vertices
    """
    vectors = np.array([vertex - v for vertex in ad_vertices])
    l = len(vectors)
    dot = np.array(
        [np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
         for v1 in vectors for v2 in vectors]
    ).reshape(l, l)
    angles = np.arccos(dot)
    if not in_degree:
        return angles
    degrees = np.degrees(angles)
    return degrees

def get_angle(v: np.ndarray, w: np.ndarray, u: np.ndarray) -> float:
    """Calculates the degree between w and u, seen from v
    
    Arguments:
        v {np.ndarray} -- Point for which the angle is calculated
        w {np.ndarray} -- Point from
        u {np.ndarray} -- Point to
    
    Returns:
        float -- Degree between w and u, seen from v
    """
    u_w = np.array([u, w])
    u_w = u_w - v
    dot = np.dot(u_w[0], u_w[1]) / (np.linalg.norm(u_w[0]) * np.linalg.norm(u_w[1]))
    angle = np.arccos(dot)
    degree = np.degrees(angle)
    return degree if not np.isnan(degree) else 0


def callback_rerouter(model, where):
    """Handle all registered callbacks and redirect them to the right function
    
    Arguments:
        model {[gurobipy.Model]} -- Model in which the rerouter will be used
        where {[int/Option]} -- Value that represents the callback case.
                              E.g: gurobipy.GRB.Callback.MIPSOL
    """
    if hasattr(callback_rerouter, "inner_callbacks"):
        if where in callback_rerouter.inner_callbacks:
            for callback in callback_rerouter.inner_callbacks[where]:
                callback(model)


def get_array_greater_zero(array: Union[list, set, np.ndarray]) -> bool:
    """Returns if an array is not None and contains at least one element

    Arguments:
        array {Union[list, set, np.ndarray]} -- Array in question

    Returns:
        bool -- True if array not None and not empty
    """
    return (array is not None and len(array) > 0)


def get_extra_arcs(sols):
    """Get the amount of extra slices taken minus the roundtrip
    
    Arguments:
        sols {AngularGraphSolution} -- MinSum solutions
    
    Yields:
        float -- Calculated unrounded extra slices amount
    """
    for sol in sols:
        n = len(sol.graph.vertices)
        roundtrip = (n-2)*180
        arc_degree = 360/n
        yield (sol.obj_val - roundtrip) / arc_degree

def get_graph_angles(graph: 'Graph') -> list:
    """Calculate the angles for a given graph

    Args:
        graph (Graph): Graph for which the angles should be calculated

    Returns:
        list: list of dictionary for angles.
              Can be called the following: [i][(j,k)] for vertices indices i,j,k,
              where we have the angle between j and k, seen from i
    """
    angles = []
    for index in range(graph.vert_amount):
        # Get adjacent vertices
        ad_vert = [j for j in range(len(graph.vertices)) if graph.ad_matrix[index, j] > 0]

        v_a = np.array(graph.vertices[index])
        vert_arr = np.array([graph.vertices[i] for i in ad_vert])
        l = len(vert_arr)
        degrees = get_angles(v_a, vert_arr)
        tuple_dict = {(ad_vert[i], ad_vert[j]): degrees[i, j]
                        for i in range(l) for j in range(l) if i != j}
        angles.append(tuple_dict)
    return angles

def calculate_times(order, graph: 'Graph' = None, return_angle_sum=False, times=None, angles=None):
    if times is None:
        times = {}
    
    assert graph is not None or angles is not None,\
        "Either graph or angles must be passed"
    angles = graph.costs if angles is None else angles
    if angles is None:
        angles = get_graph_angles(graph)

    curr_head_sum = [0 for i in range(len(angles))]
    prev_heading = [None for i in range(len(angles))]

    # If no order is given just return None
    if order is None:
        if return_angle_sum:
            return None, curr_head_sum
        return None
    # Put the heading to the right spot if times were passed
    for time in times:
        prev_heading[time[0]] = time[1]
        prev_heading[time[1]] = time[0]
    
    for vertex_indexes in order:
        max_time = 0
        vertex_set = set(vertex_indexes)
        for index in vertex_set:
            other_index = vertex_set.difference([index]).pop()
            curr_prev = prev_heading[index]
            if curr_prev is not None:
                angle = angles[index][(curr_prev, other_index)]
                curr_head_sum[index] += angle
                sorted_key = tuple(sorted([index, curr_prev]))
                max_time = max([max_time, times[sorted_key] + angle])
            prev_heading[index] = other_index

        times[tuple(sorted(vertex_indexes))] = max_time
    if return_angle_sum:
        return times, curr_head_sum
    return times

def _calculate_times_mdict(times, graph) -> Multidict:
    multidict = Multidict()
    for time_key in times:
        if graph.simple and time_key[0] > time_key[1]:
            continue
        # Check edge exists in graph
        if graph.ad_matrix[time_key[0], time_key[1]] == 0:
            raise KeyError("Graph does not containt an edge with indices {0}".format(time_key))
        multidict[times[time_key]] = time_key
    return multidict


def calculate_order_from_times(times: Union['Multidict', dict], graph: 'Graph'):
    if isinstance(times, dict):
        times = _calculate_times_mdict(times, graph)
    order = []
    for time in sorted(times.keys()):
        for edge in times[time]:
            order.append(edge)
    return order

def get_dep_graph(used_edges, abstract_graph):
        dep_graph = {key: DependencyNode(key) for key in range(len(abstract_graph.vertices))}
        for come, to in used_edges:
            dep_graph[come].add_dependency(dep_graph[to])
        return dep_graph


def _calculate_times_old(order, graph: 'Graph'):
    times = {}
    for vertex_indexes in order:
        max_time = 0
        vertex_set = set(vertex_indexes)
        for index in vertex_set:
            other_index = vertex_set.difference([index]).pop()
            connections = np.where(graph.ad_matrix[index] > 0)
            
            angles = get_angles(
                graph.vertices[index], graph.vertices)
            #ToDo: use graph.vertices[connections] instead and translate key_other_vertex to new indices
            for con in connections[0]:
                key = [index, con]
                key_curr_vertex = (min(key), max(key))
                key = [other_index, con]
                key_other_vertex = (min(key), max(key))
                if (key_curr_vertex in times and
                        (times[key_curr_vertex] + angles[key_other_vertex]) > max_time):
                    max_time = max(
                        [times[key_curr_vertex] + angles[key_other_vertex], max_time]
                        )
        times[vertex_indexes] = max_time
    return times


def get_vertex_sectors(vertex, adjacent_vertices, degrees=False, start_north=False, clockwise=True, return_angles=False):
    """Get the sectors and corresponding angles for a vertex

    Args:
        vertex (tuple): Vertex position
        adjacent_vertices ([type]): Adjacent vertices positions
    Returns:
        list: Contains tuple with sector data in the following form:
              (start_vertex, end_vertex, angle)
              Order and overall start depends on parameters
    """
    # Get the angle of all adjacent vertices
    if start_north:
        angles = np.array([i if i > 0 else 2*np.pi + i for i in np.arctan2(*(adjacent_vertices - vertex).T)])
    else:
        angles = np.arctan2(*(adjacent_vertices - vertex).T)
    argsorted = np.argsort(angles)
    if not clockwise:
        argsorted = np.array([i for i in reversed(argsorted)])
    length = len(angles)
    sector_angles = [angles[argsorted[(i+1) % (length)]] - angles[argsorted[i]] for i in range(length-1)]
    if len(argsorted > 0):
        sector_angles.append(2*np.pi - angles[argsorted[-1]] + angles[argsorted[0]])
    if degrees:
        sector_angles = np.degrees(sector_angles)
    sectors = [(argsorted[i], argsorted[(i+1)%length], sector_angles[i]) for i in range(length)]
    if return_angles:
        return sectors, angles[argsorted]
    return sectors

def get_lower_bounds(graph: "Graph"):
    v_LB = []
    for i in range(graph.vert_amount):
        sectors = get_vertex_sectors(
            graph.vertices[i],
            graph.vertices[np.nonzero(graph.ad_matrix[i])],
            degrees=True)
        if sectors:
            v_LB.append(360 - max(s[2] for s in sectors))
    if v_LB:
        return v_LB
    return [0]
    
"""def celery_is_up():
    try:
        status.run()
        return True
    except celery.bin.base.Error as e:
        if e.status == celery.platforms.EX_UNAVAILABLE:
            return False
        raise e
"""
def is_debug_env():
    """Check if you are in a debugging environment

    Returns:
        bool: Returns True if it is a debug environment
    """
    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        return False
    elif gettrace():
        return True    
    return False