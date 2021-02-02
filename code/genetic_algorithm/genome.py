"""Module for the abstract Genome class to help build the necessary functions
for a genome used in the standard genetic algorithm and crossover methods
"""
from abc import ABC, abstractmethod, abstractproperty

class Genome(ABC):
    """This is an abstract class to describe a genome.
    To use the genetic algorithm it is advised to inherit from this class
    to implement the needed functions.

    Arguments:
        ABC {ABC} -- Abstract class inheritance
    """

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def carry_in_next(self):
        """Just copy most information into the next gen genome
        """

    @abstractmethod
    def create_children(self, data, mother=None, **kwargs):
        """Abstract method for creating children.
        This is only used to create the new instance.
        It is not meant to perform crossover or mutation.

        Note, that father and mother do not need to be saved in the child as reference.
        Doing so will increase the shelf/pickle space needed by large.
        We recommend other unique identifier.

        Arguments:
            self {Genome} -- 'Father' of the child genome
            data -- The genome data

        Keyword Arguments:
            mother {Genome} -- 'Mother' of the child genome (default: {None})
        """
