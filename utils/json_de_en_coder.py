import json

import numpy as np

from utils import Multidict
from database import Graph, AngularGraphSolution


class AngularJSONEncoder(json.JSONEncoder):

    def __init__(self, *args, **kwargs):
        self.type_to_function = {
            AngularGraphSolution: self.encode_ang_sol,
            Multidict: self.encode_multidict,
            Graph: self.encode_graph
            }
        super().__init__(*args, **kwargs)

    def default(self, o):
        try:
            return self.type_to_function[type(o)](o)
        except KeyError:
            return super().default(o)

    def encode_ang_sol(self, o: AngularGraphSolution):
        if isinstance(o, AngularGraphSolution):
            return {
                    "_type": "angulargraphsolution",
                    "graph": self.default(o.graph),
                    "order": self.default(o.order),
                    "runtime": o.runtime,
                    "max_time": o.max_time                    
                }
        else:
            return super().default(o)

    def encode_multidict(self, o: Multidict):
        if isinstance(o, Multidict):
            return {
                    "_type": "multidict",
                    "dict": o._dict
                }
        else:
            return super().default(o)

    def encode_graph(self, o: Graph):
        if isinstance(o, Graph):
            serialized = {
                    "_type": "graph",
                    "vertices": o.vertices.tolist(),
                    "ad_matrix": o.ad_matrix.tolist(),
                    "name": o.name
                }
            return serialized
        else:
            return super().default(o)
    

class AngularJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        self.type_to_method = {
            "angulargraphsolution": self._decode_sol,
            "multidict": self._decode_multidict,
            "graph": self._decode_graph
        }
        kwargs["object_hook"] = self.object_hook
        super().__init__(*args, **kwargs)
        a = None

    def _decode_sol(self, sol):
        return AngularGraphSolution(
            sol["graph"],
            times=sol["order"],
            runtime=sol["runtime"],
            name=sol["name"]
            )
    def _decode_multidict(self, m_dict):
        return Multidict({float(key):m_dict["dict"] for key in m_dict["dict"]})
    def _decode_graph(self, graph):
        return Graph(
            np.array(graph["vertices"]),
            ad_matrix=np.array(graph["ad_matrix"])
            )

    def object_hook(self, obj):
        try:
            return self.type_to_method[obj["_type"]](obj)
        except KeyError:
            return obj
        