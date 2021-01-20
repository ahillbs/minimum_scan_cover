from .genetic_algorithm import AngularGeneticMinSumSolver, AngularGeneticLocalMinSumSolver, AngularGeneticMakespanSolver
from .iterated_local_search import AngularIteratedMinSumSolver, AngularIteratedLocalMinSumSolver, AngularIteratedMakespanSolver
from .simulated_annealing import AngularSimulatedAnnealingMinSumSolver, AngularSimulatedAnnealingLocalMinSumSolver, AngularSimulatedAnnealingMakespanSolver

MIN_SUM_SOLVER = {
    "AngularGeneticMinSumSolver": AngularGeneticMinSumSolver,
    "AngularIteratedMinSumSolver": AngularIteratedMinSumSolver,
    "AngularSimulatedAnnealingMinSumSolver": AngularSimulatedAnnealingMinSumSolver
}
LOCAL_MIN_SUM_SOLVER = {
    "AngularGeneticLocalMinSumSolver": AngularGeneticLocalMinSumSolver,
    "AngularIteratedLocalMinSumSolver": AngularIteratedLocalMinSumSolver,
    "AngularSimulatedAnnealingLocalMinSumSolver": AngularSimulatedAnnealingLocalMinSumSolver
}
MAKESPAN_SOLVER = {
    "AngularGeneticMakespanSolver": AngularGeneticMakespanSolver,
    "AngularIteratedMakespanSolver": AngularIteratedMakespanSolver,
    "AngularSimulatedAnnealingMakespanSolver": AngularSimulatedAnnealingMakespanSolver
}
ALL_SOLVER = {**MIN_SUM_SOLVER, **LOCAL_MIN_SUM_SOLVER, **MAKESPAN_SOLVER}