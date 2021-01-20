#import numpy as np
import pickle
import math
from sqlalchemy import Column, Integer, String, BINARY, DECIMAL, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.orm import reconstructor

from database import Base, Graph
from utils import  Multidict, calculate_times, calculate_order_from_times

class AngularGraphSolution(Base):
    __tablename__ = 'solutions'

    id = Column(Integer, primary_key=True)
    graph_id = Column(Integer, ForeignKey(Graph.id), nullable=False)
    makespan = Column(DECIMAL)
    min_sum = Column(DECIMAL)
    local_min_sum = Column(DECIMAL)
    solution_type = Column(String)
    is_optimal = Column(Boolean)
    solver = Column(String)
    error_message = Column(String)
    runtime = Column(DECIMAL)
    _sol = Column(BINARY)
    graph = relationship("Graph")

    def __init__(self, graph: Graph, runtime, solver, solution_type, is_optimal=None, error_message=None, times=None, order=None, **kwargs):
        self.graph = graph
        self.makespan = 0
        self.min_sum = 0
        self.local_min_sum = 0
        self.local_sum = []
        self.solution_type = solution_type
        self.is_optimal = is_optimal
        self.solver = solver
        self.error_message = error_message
        self.times = times
        self.order = order
        if error_message is None:
            self._calculate_objectives()
        self.runtime = runtime
        self._sol = pickle.dumps(self.order)
        # Just at the moment to allow addings more stuff dynamically
        for key in kwargs:
            setattr(self, key, kwargs[key])
    
    @reconstructor
    def _reconstruct_order(self):
        self.order = pickle.loads(self._sol)
        # This is just for older versions, where the solution was saved as multidict
        if isinstance(self.order, Multidict):
            self.times = self.order
            self.order = calculate_order_from_times(self.times, self.graph)
        
        self.times, self.local_sum = calculate_times(self.order, self.graph, return_angle_sum=True)
        if (self.makespan is None or self.min_sum is None or self.local_min_sum is None):
            try:
                self.makespan = max(self.times.values())
                self.min_sum = sum(self.local_sum)
                self.local_min_sum = max(self.local_sum)
            except (TypeError, AttributeError):
                pass # can happen if a solution has an error
        

    def get_ordered_times(self) -> Multidict:
        multidict = Multidict()
        for time_key in self.times:
            # Check edge exists in graph
            if self.graph.ad_matrix[time_key[0], time_key[1]] == 0:
                raise KeyError("Graph does not containt an edge with indices {0}".format(time_key))
            multidict[self.times[time_key]] = time_key
            self.makespan = max(0, self.times[time_key])
        return multidict

    def _calculate_objectives(self):
        assert not (self.order and self.times),\
            "Pass order or times, but not both!"
        self.makespan = None
        self.min_sum = None
        self.local_min_sum = None
        if self.times:
            warning_printed = False
            self.order = calculate_order_from_times(self.times, self.graph)
            times, self.local_sum = calculate_times(self.order, self.graph, return_angle_sum=True)
            for key in times:
                try:
                    assert math.isclose(times[key], self.times[key], abs_tol=10**-5)
                except AssertionError as e:
                    if not warning_printed:
                        warning_printed = True
                        if self.solution_type == "min_sum":
                            if hasattr(self, "obj_val") and \
                               not math.isclose(sum(self.local_sum), self.obj_val):
                                print("Warning: Solution type is min_sum but minsum is mismatched: ",
                                      sum(self.local_sum), "vs:", self.obj_val)
                        print("Warning: Times are not matching:", times[key], self.times[key])
                        max_times = max([times[i] for i in times])
                        max_self_times = max([self.times[i] for i in self.times])
                        
                        if not math.isclose(max_times, max_self_times, abs_tol=10**-5):
                            print("And max times are also mismatched. Calculated:", max_times, " passed:", max_self_times)

        elif self.order:
            self.times, self.local_sum = calculate_times(self.order, self.graph, return_angle_sum=True)
        else:
            return
        self.makespan = max([self.times[key] for key in self.times])
        self.min_sum = sum(self.local_sum)
        self.local_min_sum = max(self.local_sum)

    def __repr__(self):
        whole_str = "Type: {0} ".format(self.solution_type)
        whole_str += "Graph: {0} ".format(self.graph.name)
        if self.solution_type == "makespan" and self.makespan is not None:
            whole_str += "Makespan: {0} ".format(self.makespan)
        if self.solution_type == "min_sum" and self.min_sum is not None:
            whole_str += "MinSum: {0} ".format(self.min_sum)
        if self.solution_type == "local_min_sum" and self.local_min_sum is not None:
            whole_str += "LocalMinSum: {0} ".format(self.local_min_sum)
        if self.is_optimal:
            whole_str += "Is optimal: {0}".format(self.is_optimal)
        return whole_str

    def get_extra_slice_amount(self: 'AngularGraphSolution'):
        pass