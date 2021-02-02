import pytest
import math
from instance_generation import create_circle_n_k
from utils.util_functions import calculate_times, _calculate_times_old, calculate_order_from_times
from utils import get_tripeledges_from_abs_graph, convert_graph_to_angular_abstract_graph

def test_calc_times_old_new():
    graph = create_circle_n_k(15, 10)
    order = [(i, j) for i in range(graph.vert_amount) for j in range(i+1, graph.vert_amount)]
    old_times = _calculate_times_old(order, graph)
    new_times = calculate_times(order, graph)
    for old, new in zip(old_times, new_times):
        assert old == new
        assert math.isclose(old_times[old], new_times[new])

def test_get_order_from_times():
    graph = create_circle_n_k(15, 10)
    order = [(i, j) for i in range(graph.vert_amount) for j in range(i+1, graph.vert_amount)]
    times = calculate_times(order, graph)
    new_order = calculate_order_from_times(times, graph)
    # As both orders could actually differ without changing a thing,
    # we instead test if we get the same times again
    new_times = calculate_times(new_order, graph)
    for key in new_times:
        assert math.isclose(times[key], new_times[key])

def test_get_tripel_edges():
    graph = create_circle_n_k(15, 5)
    tripel_control, abs_graph = convert_graph_to_angular_abstract_graph(graph, simple_graph=False, return_tripel_edges=True)
    tripel_edges = get_tripeledges_from_abs_graph(abs_graph)
    assert len(tripel_edges) == len(tripel_control)
    for edge in tripel_edges:
        assert tripel_control[edge] == tripel_edges[edge]
