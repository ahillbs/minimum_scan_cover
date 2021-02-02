import datetime
import json
from typing import Dict
from sqlalchemy import Column, Integer, String, BINARY, ForeignKey, DATETIME, DateTime
from sqlalchemy.orm import reconstructor, relationship
from database import Base, Task

class Config(Base):
    __tablename__ = "configs"
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    task = relationship("Task", back_populates="configs")
    param = Column(String, nullable=False)
    _value_str = Column(String)

    @property
    def value(self):
        return self._value
    @value.setter
    def value(self, value):
        self._value_str = json.dumps(value)
        self._value = value

    def __init__(self, **kwargs):
        try:
            val = kwargs.pop("value")
            self.value = val
        except AttributeError:
            pass
        super().__init__(**kwargs)

    @reconstructor
    def reconstruct(self):
        self._value = json.loads(self._value_str)

    def __str__(self):
        return "Task: " + str(self.task_id) + ", Param: " + str(self.param) + " Value: " + str(self.value)
    def __repr__(self):
        return str(self)



class ConfigHolder():
    @property
    def database_configs(self):
        return [self._configs[config] for config in self._configs]

    def __init__(self, task: Task):
        self._configs = {}
        if task:
            # This method also recursively loads configs from parent tasks
            # Child tasks will always overwrite parent configs if same parameter exists
            self._configs = self._get_configs(task)
            for param in self._configs:
                assert "_config" != param, "Param cant be '_config'!"
                setattr(self, param, self._configs[param])
    
    def _get_configs(self, task: Task) -> Dict:
        config = {}
        if task.parent is not None:
            config = self._get_configs(task.parent)
        config.update({c.param: c.value for c in task.configs})
        return config

    def attach_task(self, task: Task):
        for attr in self._configs:
            self._configs[attr].task = task

    def __iter__(self):
        return iter(self._configs)
    
    def __getitem__(self, key):
        return self._configs[key]

    @staticmethod
    def fromNamespace(namespace, task=None, ignored_attributes=None) -> "ConfigHolder":
        config_holder = ConfigHolder(task)
        if ignored_attributes is None:
            ignored_attributes = []
        for conf_name in namespace.__dict__:
            value = namespace.__dict__[conf_name]
            if conf_name not in ignored_attributes:
                config_holder._configs[conf_name] = Config(task=task, param=conf_name, value=value)
                setattr(config_holder, conf_name, value)
        return config_holder
