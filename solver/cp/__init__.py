from .abs_makespan_solver import ConstraintAbsSolver
from .dependency_solver import ConstraintDependencySolver, ConstraintDependencyLocalMinSumSolver
MIN_SUM_SOLVER = {
    "ConstraintDependencySolver": ConstraintDependencySolver,
}
LOCAL_MIN_SUM_SOLVER = {
    "ConstraintDependencyLocalMinSumSolver": ConstraintDependencyLocalMinSumSolver
}
MAKESPAN_SOLVER = {
    "ConstraintAbsSolver": ConstraintAbsSolver
}
ALL_SOLVER = {**MIN_SUM_SOLVER, **LOCAL_MIN_SUM_SOLVER, **MAKESPAN_SOLVER}