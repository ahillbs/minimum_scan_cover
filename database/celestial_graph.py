from typing import Union, Optional, List
import numpy as np
from . import Graph, Base

from sqlalchemy import Column, Integer, String, BINARY, DECIMAL, NUMERIC, ForeignKey
from sqlalchemy.orm import reconstructor, relationship

class CelestialBody(Base):
    """Celestial body that can block communications (edges) in a celestial graph
    """
    __tablename__ = 'celestial_bodies'

    id = Column(Integer, primary_key=True)
    _position_binary = Column(BINARY, nullable=False)
    size = Column(NUMERIC(asdecimal=False), nullable=False)
    cel_graph_id = Column(Integer, ForeignKey("celestial_graph.id"), nullable=False)
    cel_graph = relationship("CelestialGraph", back_populates="celestial_bodies")

    @property
    def position(self):
        return self._position
    
    @position.setter
    def position(self, value: np.ndarray):
        self._position_binary = value.dumps()
        self._position = value
    
    def __init__(self, position: Union[list, np.ndarray], size: Union[int, float, np.float], cel_graph: "CelestialGraph"):
        self.position = np.array(position)
        self.size = size
        self.cel_graph = cel_graph
    
    @reconstructor
    def _reconstructor(self):
        self._position = np.loads(self._position_binary)
        self.size = float(self.size)

class CelestialGraph(Graph):
    """Graph with celestial bodies
    """

    __tablename__ = 'celestial_graph'
    __mapper_args__ = {
        'polymorphic_identity':'CelestialGraph',
    }

    id = Column(Integer, ForeignKey('graphs.id'), primary_key=True)
    celestial_bodies = relationship("CelestialBody", back_populates="cel_graph")

    def __init__(self,
                 celestial_bodies: List[CelestialBody],
                 V: Optional[Union[set, list, np.ndarray]] = None,
                 E: Optional[Union[list, set, np.ndarray]] = None,
                 ad_matrix: Optional[np.ndarray] = None,
                 c: Optional[dict] = None,
                 i_type="cel_graph",
                 ):
        self.celestial_bodies = celestial_bodies
        super().__init__(V=V, E=E, ad_matrix=ad_matrix, c=c, i_type=i_type)
        self._calculate_illegal_edges()

    def _calculate_illegal_edges(self):
        # First test if any satellite will we be in the sphere of any celestial body
        for vertex in self.vertices:
            for body in self.celestial_bodies:
                assert np.linalg.norm(vertex - body.position) > body.size

        # Now test if any edge cuts any celestial body
        to_delete = []
        for i in range(len(self.edges)):
            for body in self.celestial_bodies:
                p1 = self.vertices[self.edges[i][0]]
                p2 = self.vertices[self.edges[i][1]]
                p3 = body.position
                d = np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)
                epsilon = abs(d) * 0.01
                if abs(d) + epsilon < body.size:
                    to_delete.append(i)
        # First zero out in adjacency matrix, then delete illegal edges
        delete_arr = np.vstack([
            self.edges[to_delete],
            self.edges[to_delete]@np.array([[0, 1], [1, 0]])
            ])
        filter_list = (delete_arr.T[0], delete_arr.T[1])
        self.ad_matrix[filter_list] = 0
        self._dirty_edges = True
        self.update()
