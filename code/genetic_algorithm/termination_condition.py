import tqdm
import time
import numpy as np
from . import GeneticAlgorithm

class IterationTerminationConditionMet():
    def __init__(self, max_iter=1, *args, **kwargs):
        self.max_iter = max_iter
        self.tqdm = None
        return

    def __call__(self, evo_alg: GeneticAlgorithm):
        if evo_alg.generation == 0:
            try:
                self.tqdm.reset()
            except AttributeError:
                self.tqdm = tqdm.tqdm(total=self.max_iter,
                                      initial=evo_alg.generation)
        else:
            try:
                self.tqdm.update()
            except AttributeError:
                self.tqdm = tqdm.tqdm(total=self.max_iter,
                                      initial=evo_alg.generation)

        if evo_alg.generation >= self.max_iter:
            self.tqdm.close()
            return True
        else:
            return False

class TimeConstraintTermination():
    """Termination condition after a set amount of time.
    """
    def __init__(self, time_limit=900, **kwargs):
        self.time_limit = time_limit
        self.start_time = None

    def __call__(self, evo_alg: GeneticAlgorithm):
        if evo_alg.generation == 0 or self.start_time is None:
            self.start_time = time.time()
            return False
        return self.start_time + self.time_limit < time.time()

class NoImprovementsTermination():
    """Termination condition where termination is signaled if no fitness improvements were found for a set amount of iterations
    """
    def __init__(self, max_iter: int):
        """Instanciate termination condition without fitness improvement for a number of iterations

        Args:
            max_iter (int): How many iterations no better fitness is found before termination
        """
        self._max_iter_w_o_improvement = max_iter
        self.best_fitness = None
        self._counter = 0

    def __call__(self, evo_alg: GeneticAlgorithm):
        if evo_alg.fitness_val is not None:
            best_fitness = max(evo_alg.fitness_val)
            if self.best_fitness is None or self.best_fitness < best_fitness:
                self._counter = 0
                self.best_fitness = best_fitness
            else:
                self._counter += 1
                return self._max_iter_w_o_improvement < self._counter
        return False

    def tear_down(self):
        """Resets the inner variables
        """
        self.best_fitness = None
        self._counter = 0



class TerminationCombination():
    """Build a termination condition out of a combination of other conditions.
       This combination condition signals termination if one inner condition signals termination.
    """
    def __init__(self, terminations: list):
        self._termintaions = terminations
        
    @property
    def tqdm(self):
        for termination in self._termintaions:
            if hasattr(termination, "tqdm"):
                return termination.tqdm
        raise AttributeError()
    def __call__(self, evo_alg: GeneticAlgorithm):
        conditions_met = [not termination(evo_alg) for termination in self._termintaions]
        terminates = not np.alltrue(conditions_met)
        if terminates:
            for termination in self._termintaions:
                try:
                    termination.tear_down()
                except AttributeError:
                    pass
        return terminates

    def add_termination(self, termination_condition):
        """Add another termination condition

        Args:
            termination_condition (callable): Termination condition to add
        """
        self._termintaions.append(termination_condition)

    def remove_termination(self, termination_condition):
        """Remove a termination condition

        Args:
            termination_condition (callable): Termination condition to remove
        """
        self._termintaions.remove(termination_condition)
