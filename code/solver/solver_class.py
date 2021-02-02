import abc

from database import Graph

class Solver(abc.ABC):
    translate = {
                    1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD',
                    5: 'UNBOUNDED', 6: 'CUTOFF', 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT',
                    9: 'TIME_LIMIT', 10: 'SOLUTION_LIMIT', 11: 'INTERRUPTED', 12: 'NUMERIC',
                    13: 'SUBOPTIMAL', 14: 'INPROGRESS', 15: 'USER_OBJ_LIMIT'
                    }
    @abc.abstractproperty
    def is_multicore(self) -> bool:
        """Property if the solver uses multiple cores

        Returns:
            bool: True if solver uses multiple cores
        """
        return NotImplemented
    
    def __init__(self, params=None):
        self.params = params
    @abc.abstractmethod
    def solve(self, graph: Graph, **kwargs):
        pass

class NotOptimalError(Exception):
    pass