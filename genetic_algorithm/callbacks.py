import pathlib
import numpy as np

from . import GeneticAlgorithm
from database import Task

class SaveCallback:
    def __init__(self, iterations, population_amount, task: Task, session):
        #self.generations = np.empty(
        #    (iterations, population_amount), dtype=object)
        self.session = session
        self.task = task
        return

    def __call__(self, gen_algo: GeneticAlgorithm):
        if not self.session or not self.task:
            return
        try:
            self.session.add_all(gen_algo.genomes.tolist())
            self.session.commit()
            #self.generations[gen_algo.generation][:] = gen_algo.genomes[:]
            #with  open(self.savePath, 'wb') as fd_instances:
            #    pickle.dump(self.generations[:gen_algo.generation+1], fd_instances)
        except Exception as e:
            print("Exception while calling the iteration end callback:", e)

def update_callback(gen_algo: GeneticAlgorithm):
    mean = np.mean(gen_algo.fitness_val)
    max_fitness = np.max(gen_algo.fitness_val)
    min_fitness = np.min(gen_algo.fitness_val)
    gen_algo.termCon.tqdm.set_description_str(
        "Evolve feature genomes. Fitness values: Mean: {0}, max: {1}, min: {2}".format(mean, max_fitness, min_fitness)
        )
