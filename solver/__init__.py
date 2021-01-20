from .solver_class import Solver
from .bipartite_solver import AngularBipartiteMinSumSolver, AngularBipartiteMakespanSolver
from .msc_coloring_solver import MscColoringSolver
from .s_n_2_solver import GraphN2Solver
from .greedy import AngularMinSumGreedySolver, AngularMinSumGreedyConnectedSolver, AngularLocalMinSumGreedySolver, AngularMakespanGreedySolver
from .s_n_k_mirror_solver import SnkMirrorSolver

from . import mip, cp, meta_heur

MIN_SUM_SOLVER = {
    "AngularMinSumGreedySolver": AngularMinSumGreedySolver,
    "AngularMinSumGreedyConnectedSolver": AngularMinSumGreedyConnectedSolver,
    "AngularBipartiteMinSumSolver": AngularBipartiteMinSumSolver
}
LOCAL_MIN_SUM_SOLVER = {
    "AngularLocalMinSumGreedySolver": AngularLocalMinSumGreedySolver,
    "SnkMirrorSolver": SnkMirrorSolver
}
MAKESPAN_SOLVER = {
    "MscColoringSolver": MscColoringSolver,
    "AngularMakespanGreedySolver": AngularMakespanGreedySolver,
    "AngularBipartiteMakespanSolver": AngularBipartiteMakespanSolver,
}
ALL_LOCAL_SOLVER = {**MIN_SUM_SOLVER, **LOCAL_MIN_SUM_SOLVER, **MAKESPAN_SOLVER}

ALL_MIN_SUM_SOLVER = {**MIN_SUM_SOLVER, **mip.MIN_SUM_SOLVER, **cp.MIN_SUM_SOLVER, **meta_heur.MIN_SUM_SOLVER}
ALL_LOCAL_MIN_SUM_SOLVER = {**LOCAL_MIN_SUM_SOLVER, **mip.LOCAL_MIN_SUM_SOLVER, **cp.LOCAL_MIN_SUM_SOLVER, **meta_heur.LOCAL_MIN_SUM_SOLVER}
ALL_MAKESPAN_SOLVER = {**MAKESPAN_SOLVER, **mip.MAKESPAN_SOLVER, **cp.MAKESPAN_SOLVER, **meta_heur.MAKESPAN_SOLVER}
ALL_SOLVER = {**ALL_MIN_SUM_SOLVER, **ALL_LOCAL_MIN_SUM_SOLVER, **ALL_MAKESPAN_SOLVER}