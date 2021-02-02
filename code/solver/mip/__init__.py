from .angular_graph_scan_hamilton import AngularGraphScanMakespanHamilton, AngularGraphScanMinSumHamilton, AngularGraphScanLocalMinSumHamilton
from .angular_graph_scan_absolute import AngularGraphScanMakespanAbsolute, AngularGraphScanMakespanAbsoluteReduced
from .angular_dependency import AngularDependencySolver, AngularDependencyLocalMinSumSolver
MIN_SUM_SOLVER = {
    "AngularDependencySolver": AngularDependencySolver,
    "AngularGraphScanMinSumHamilton": AngularGraphScanMinSumHamilton,
}
LOCAL_MIN_SUM_SOLVER = {
    "AngularGraphScanLocalMinSumHamilton": AngularGraphScanLocalMinSumHamilton,
    "AngularDependencyLocalMinSumSolver": AngularDependencyLocalMinSumSolver
}
MAKESPAN_SOLVER = {
    "AngularGraphScanMakespanHamilton": AngularGraphScanMakespanHamilton,
    "AngularGraphScanMakespanAbsolute": AngularGraphScanMakespanAbsolute,
    "AngularGraphScanMakespanAbsoluteReduced": AngularGraphScanMakespanAbsoluteReduced
}
ALL_SOLVER = {**MIN_SUM_SOLVER, **LOCAL_MIN_SUM_SOLVER, **MAKESPAN_SOLVER}