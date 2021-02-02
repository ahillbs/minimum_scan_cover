from typing import Union, Optional
import numpy as np
from itertools import combinations, product

from sqlalchemy import Column, Integer, String, BINARY
from sqlalchemy.orm import reconstructor

from . import Base

def get_array_greater_zero(array: Union[list, set, np.ndarray]):
    return (array is not None and len(array) > 0)

class Graph(Base):
    __tablename__ = 'graphs'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    vert_amount = Column(Integer)
    edge_amount = Column(Integer)
    _vert_bin = Column(BINARY)
    _edge_bin = Column(BINARY)
    i_type = Column(String, default="general")
    type = Column(String)

    __mapper_args__ = {
        'polymorphic_identity':'graphs',
        'polymorphic_on':type
    }

    @property
    def ad_matrix(self):
        if self._dirty_matrix:
            self._calculate_adjacency_matrix()
            self._dirty_matrix = False
        return self._ad_matrix
    @ad_matrix.setter
    def ad_matrix(self, value):
        self._ad_matrix = value
        self._dirty_edges = True
    
    @property
    def edges(self):
        if self._dirty_edges:
            self._calc_edges()
            self._dirty_edges = False
        return self._edges
    @edges.setter
    def edges(self, value):
        self._edges = value
        self._edge_bin = self._edges.dumps()
        self._dirty_matrix = True

    @property
    def vertices(self):
        return self._vertices
    @vertices.setter
    def vertices(self, value):
        self._vertices = value
        self._vert_bin = self._vertices.dumps()
        self._dirty_matrix = True

    def __init__(self, V: Optional[Union[set, list, np.ndarray]] = None,
                 E: Optional[Union[list, set, np.ndarray]] = None,
                 ad_matrix: Optional[np.ndarray] = None,
                 c: Optional[dict] = None,
                 name=None,
                 simple=None,
                 **kwargs):
        self._dirty_edges = False
        self._dirty_matrix = False
        self._ad_matrix = None
        self.simple = True if simple is None else simple
        if get_array_greater_zero(V):
            self.vertices = np.array([v for v in V])
        else:
            self.vertices = range(len(ad_matrix))

        self._edges = np.array([])
        if get_array_greater_zero(E):
            self.edges = np.array([e for e in E])
            self._dirty_edges = False

        self._ad_matrix = np.array([])
        if get_array_greater_zero(ad_matrix):
            self.ad_matrix = ad_matrix
            self._dirty_matrix = False
        
        
        if simple:
            if not self.check_simple():
                print("WARNING: Created instance was not simple but passed as simpel by argument")
                self.simple = False
        else:
            self.simple = self.check_simple()

        self.costs = c
        if name:
            self.name = name
        else:
            # ToDo: maybe rename
            self.name = str(hash(self))
        self.vert_amount = len(self.vertices)
        self.edge_amount = len(self.edges)
        super().__init__(**kwargs)

    @reconstructor
    def _reconstruct(self):
        self.vertices = np.loads(self._vert_bin)
        self.edges = np.loads(self._edge_bin)
        self._dirty_edges = False
        self._dirty_matrix = True
        self._ad_matrix = None
        self.costs = None
        self.simple = self.check_simple()
    
    def check_simple(self):
        """Check if graph is really simple.

        Returns:
            Bool -- returns if graph is simple
        """        
        return np.all(np.triu(self.ad_matrix) == np.tril(self.ad_matrix).T)

    def __hash__(self):
        return hash(self.vertices.tobytes() + self.edges.tobytes())

    def add_vertex(self, vertex):
        self.vertices += vertex

    def add_edge(self, edge, cost=None):
        np.append(self.edges, edge)
        self.ad_matrix[edge[0], edge[1]] = 1
        if self.simple:
            self.ad_matrix[edge[1], edge[0]] = 1
        self.costs[edge] = cost

    def _calculate_adjacency_matrix(self):
        vertex_len = len(self.vertices)
        self._ad_matrix = np.zeros([vertex_len, vertex_len])
        # Transform edges to numpy filter and put edges to one
        if len(self._edges) > 0:
            numpy_filter = (self.edges.T[0], self.edges.T[1])
            self._ad_matrix[numpy_filter] = 1
            if np.tril(self._ad_matrix, -1).max() == 0 or np.triu(self._ad_matrix, 1).max() == 0:
                self._ad_matrix = self._ad_matrix + self._ad_matrix.T
        self.edge_amount = len(self._edges)
        """
        dim = self.vertices.shape[1]        
        edges_flat = self.edges.reshape(len(self.edges), 2 * dim)
        self.ad_matrix = np.array([
            int(np.any(
                np.all(
                    np.array([v1, v2]).reshape(1, 2*dim) == edges_flat, axis=1
                    )
                ))
            for v1 in self.vertices for v2 in self.vertices]).reshape(vertex_len, vertex_len)
        self.ad_matrix = self.ad_matrix + self.ad_matrix.transpose()"""
    
    def _calc_edges(self):
        # Check if upper and lower matrix are the same -> simple graph indication
        length = len(self.vertices)
        u = np.triu_indices(length, 1)
        simple = np.all(self._ad_matrix[u] == self._ad_matrix.T[u])
        indices = range(length)
        
        combined_indices = combinations(indices, 2) if not simple else np.array(u).T
        self._edges = np.array([
            (i, j)
            for i, j in combined_indices if self._ad_matrix[i, j] > 0
            ])
        self._edge_bin = self._edges.dumps()
        self.edge_amount = len(self._edges)

    def update(self):
        if self._dirty_edges:
            self._calc_edges()
            self._dirty_edges = False
        elif self._dirty_matrix:
            self._calculate_adjacency_matrix()
            self._dirty_matrix = False

    def get_subgraph(self, vertices: Union[list, set, np.array]):
        index_list = self._check_vertices_index_or_itself(vertices)
        sub_vertices = [self.vertices[i] for i in index_list]

        sub_ad_matrix = self.ad_matrix[index_list, index_list]
        return Graph(V=sub_vertices, ad_matrix=sub_ad_matrix)

    def get_bipartite_subgraph(self, set_one: Union[list, set, np.array], set_two: Union[list, set, np.array], forbidden_edges=None):
        index_list_one = self._check_vertices_index_or_itself(set_one)
        index_list_two = self._check_vertices_index_or_itself(set_two)
        # Construct new bipartite adjacency matrix
        l = len(self.vertices)
        reverse_id = [[0, 1], [1, 0]]
        new_ad_matrix = np.zeros((l, l), dtype=int)
        # Produces indices for upper matrix
        relevant_matrix_indices = np.array([p for p in product(index_list_one, index_list_two)])
        # Combines indices for upper and lower matrix
        relevant_matrix_indices = np.vstack([relevant_matrix_indices, relevant_matrix_indices@reverse_id])
        # Bring indices to the right form for indexing
        to_numpy_indexing = (relevant_matrix_indices.T[0], relevant_matrix_indices.T[1])
        new_ad_matrix[to_numpy_indexing] = self.ad_matrix[to_numpy_indexing]
        if forbidden_edges:
            assert_err_mess = """Cannot delete edges when both sets do not contain
            all vertices (at the moment)"""
            assert (new_ad_matrix.shape == self.ad_matrix.shape), assert_err_mess
            forbidden_arr = np.array(list(forbidden_edges))
            forbidden_arr = np.vstack([forbidden_arr, forbidden_arr@reverse_id])
            # Transpose and put into tuple to use them as indexing, like in np.where
            forbidden_tuple = (forbidden_arr.T[0], forbidden_arr.T[1])
            new_ad_matrix[forbidden_tuple] = 0

        return Graph(V=self.vertices, ad_matrix=new_ad_matrix)
        
    
    '''
        Check if vertices are of the same instance type or int

    '''
    def _check_vertices_index_or_itself(self, vertices: Union[list, set, np.array]) -> list:
        index_list = []
        if isinstance(vertices[0], type(self.vertices[0])):
            index_list = [i for i in range(len(self.vertices)) if self.vertices[i] in vertices]
        elif isinstance(vertices[0], int) or isinstance(vertices[0], np.integer):
            index_list = vertices
            
        else:
            raise TypeError("Vertices parameter does not match graph vertices type or int")
        return index_list
