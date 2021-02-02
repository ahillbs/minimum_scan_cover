from . import CompleteGraphGenome, GraphGenome
import numpy as np

class CompleteGraphGenomeCreator():
    def __init__(self, point_amount=7, bounds=(0, 500)):
        self.point_amount = point_amount
        self.bounds = bounds

    def __call__(self, task, generation):
        return CompleteGraphGenome.create_random_instance(self.point_amount, task, bounds=self.bounds, generation=generation)

class GraphGenomeCreator():
    def __init__(self, point_amount=7, bounds=(0, 500)):
        self.point_amount = point_amount
        self.bounds = bounds

    def __call__(self, task, generation):
        return GraphGenome.create_random_instance(self.point_amount, task, bounds=self.bounds, generation=generation, edge_chance=np.random.random())