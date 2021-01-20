import numpy as np
from typing import List
from database import Graph, CelestialGraph, CelestialBody

def create_circle(amount: int, offset=(0, 0), radius=1) -> np.ndarray:
    step = 2*np.pi / (amount)
    points = np.array([(np.sin(step * i), np.cos(step * i)) for i in range(amount)])
    points = points * radius
    points = points + offset
    return points

def _create_circle_radial_points(radial_positions: List[float], offset=(0, 0), radius=1) -> np.ndarray:
    points = np.array([(np.sin(radial), np.cos(radial)) for radial in radial_positions])
    points = points * radius
    points = points + offset
    return points

def create_circle_graph(radial_positions: List[float], offset=(0, 0), radius=1, edges=None) -> Graph:
    points = _create_circle_radial_points(
        radial_positions=radial_positions,
        offset=offset,
        radius=radius)
    if edges is None:
        ad_matrix = np.ones((len(points), len(points)))
        ad_matrix[np.diag_indices_from(ad_matrix)] = 0
        return CelestialGraph(points, ad_matrix=ad_matrix)
    return CelestialGraph(points, E=edges)


def create_random_circle(amount: int, offset=(0, 0), seed=None):
    if isinstance(seed, np.random.Generator):
        gen = seed
    else:
        gen = np.random.Generator(np.random.BitGenerator(seed))

    r_pos = gen.random(amount) * 2*np.pi
        
    points = np.array([(np.sin(r_pos[i]), np.cos(r_pos[i])) for i in range(amount)])
    points = points + offset
    return points

def create_random_ellipse(amount, a=1, b=1, offset=(0, 0), seed=None):
    if isinstance(seed, np.random.Generator):
        gen = seed
    else:
        gen = np.random.Generator(np.random.BitGenerator(seed))

    r_pos = gen.random(amount) * 2 * np.pi
    points = np.array([(b * np.sin(r_pos[i]), a * np.cos(r_pos[i])) for i in range(amount)])
    points = points + offset
    return points

def create_random_celest_graph(amount: int, offset=(0, 0), seed=None, celest_pos="center", celest_bounds=(0.1, 0.8), vertex_shape=(1, 1)):
    if isinstance(seed, np.random.Generator):
        gen = seed
    else:
        gen = np.random.default_rng(seed)
    assert len(vertex_shape) == 2
    points = create_random_ellipse(amount,*vertex_shape, offset=offset, seed=gen)
    bodies = []
    assert celest_pos in ["center", "random"]

    body = None
    size = gen.uniform(*celest_bounds)
    if celest_pos == "center":
        position = np.array([0, 0]) + offset
    if celest_pos == "random":
        position = gen.uniform(size=(2)) + offset
    body = CelestialBody(position, size, None)
    bodies.append(body)
    edges = np.array([[i, j] for i in range(amount) for j in range(i+1, amount)])
    graph = CelestialGraph(bodies, points, edges)
    return graph

def create_circle_n_k(n: int, k: int) -> Graph:
    points = create_circle(n)
    if n > 2*k:
        edges = np.array([sorted([i, j % n]) for i in range(n) for j in range(i+1, i+1+k)])
    else:
        edges = np.array([(i, j) for i in range(n) for j in range(n) if i != j])

    return CelestialGraph([], points, edges, i_type="snk")