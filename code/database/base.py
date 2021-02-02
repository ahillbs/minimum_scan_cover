from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from abc import ABCMeta

class DeclarativeABCMeta(ABCMeta, DeclarativeMeta):
    pass

Base = declarative_base(metaclass=DeclarativeABCMeta)