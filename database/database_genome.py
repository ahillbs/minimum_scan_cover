from sqlalchemy import Column, Integer, String, BINARY, ForeignKey
from sqlalchemy.orm import reconstructor, relationship

from genetic_algorithm import Genome
from database import Base, Graph, AngularGraphSolution



class DatabaseGraphGenome(Genome, Base):
    """Genome that holds the database information
    """
    __tablename__ = "genomes"

    id = Column(Integer, primary_key=True, unique=True)
    graph_id = Column(Integer, ForeignKey(Graph.id))
    graph = relationship("Graph")
    generation = Column(Integer, nullable=False, default=0)
    mother_id = Column(Integer, ForeignKey("genomes.id"))
    father_id = Column(Integer, ForeignKey("genomes.id"))
    father = relationship("DatabaseGraphGenome", remote_side=[id], foreign_keys=[father_id])#, uselist=False)
    mother = relationship("DatabaseGraphGenome", remote_side=[id], foreign_keys=[mother_id])#, uselist=False)
    solution_id = Column(Integer, ForeignKey(AngularGraphSolution.id))
    solution = relationship(AngularGraphSolution)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    task = relationship("Task")
    config = relationship("Config", primaryjoin="DatabaseGraphGenome.task_id == Config.task_id", foreign_keys=[task_id])
    type = Column(String)

    __mapper_args__ = {
        'polymorphic_identity':'genomes',
        'polymorphic_on':type
    }

    def __init__(self, graph: Graph, parents=None, **kwargs):
        super().__init__(**kwargs)
        self.graph = graph
        try:
            self.father = parents[0]
            self.mother = parents[1]
        except (AttributeError, TypeError, IndexError):
            pass
        try:
            self.task = kwargs.pop("task")
        except KeyError:
            pass
        
    
    @reconstructor
    def reconstruct(self):
        self.parents = [self.father, self.mother]