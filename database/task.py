import datetime
from sqlalchemy import Column, Integer, String, BINARY, ForeignKey, DATETIME, DateTime, Table
from sqlalchemy.orm import reconstructor, relationship
from . import Base


class TaskJobs(Base):
    __tablename__ = "task_jobs"
    id = Column('id', Integer, primary_key=True)
    task_id = Column('task_id', Integer, ForeignKey('tasks.id'))
    graph_id = Column('graph_id', Integer, ForeignKey('graphs.id'))
    solution_id = Column('solution_id', Integer, ForeignKey('solutions.id'))
    prev_job_id = Column('prev_job_id', Integer, ForeignKey('task_jobs.id'))
    task = relationship("Task", back_populates="jobs")
    graph = relationship("Graph")
    solution = relationship("AngularGraphSolution")
    prev_job = relationship("TaskJobs", remote_side=[id], foreign_keys=[prev_job_id])

    def __repr__(self):
        return f"TaskJob {self.id}: Task {self.task_id} Graph {self.graph_id}"
    def __str__(self):
        return self.__repr__()

class StatusOption():
    PROCESSING = "PROCESSING"
    CREATED = "CREATED"
    ERROR = "ERROR"
    FINISHED = "FINISHED"
    ABORTED = "ABORTED"
    INTERRUPTED = "INTERRUPTED"


class Task(Base):
    __tablename__ = "tasks"
    STATUS_OPTIONS = StatusOption()

    id = Column(Integer, primary_key=True)
    name = Column(String)
    task_type = Column(String)
    status = Column(String, default=StatusOption.CREATED)
    configs = relationship("Config", back_populates="task")
    creation_date = Column(DATETIME, default=datetime.datetime.now)
    last_updated = Column(DATETIME, default=datetime.datetime.now)
    jobs = relationship("TaskJobs", back_populates="task")
    error_message = Column(String)
    parent_id = Column(Integer, ForeignKey("tasks.id"))
    parent = relationship("Task", remote_side=[id], foreign_keys=[parent_id])#, uselist=False)
    children = relationship("Task", back_populates="parent")

    def __repr__(self):
        return f"Task {self.id} Type: {self.task_type} Status: {self.status} Name: {self.name}"
    def __str__(self):
        return self.__repr__()
