import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
# from pymoo.decomposition import get_decomposition
# from pymoo.util.decomposition import Tchebycheff
# from pymoo.decomposition.tchebycheff import Tchebycheff
# decomposition = Tchebycheff()
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter

# Create interpolation models
def create_interpolators(df):
    X = df[["w1", "w2", "w3"]].values
    priority_latency_model = NearestNDInterpolator(X, df["priority_latency"].values)
    regular_latency_model = NearestNDInterpolator(X, df["regular_latency"].values)

    priority_deviation_model = NearestNDInterpolator(X, df["std_prior_latency"].values)              
    regular_deviation_model = NearestNDInterpolator(X, df["std_reg_latency"].values)

    priority_objects_model = NearestNDInterpolator(X, df["transmitted_priority_objects"].values)
    regular_objects_model = NearestNDInterpolator(X, df["transmitted_regular_objects"].values)

    travelled_distance_model = NearestNDInterpolator(X, df["travelled_distance"].values)

    return priority_latency_model, regular_latency_model,priority_deviation_model,regular_deviation_model, priority_objects_model, regular_objects_model, travelled_distance_model

# Define Multi-Objective Optimization Problem
class MultiRobotExplorationProblem(ElementwiseProblem):
    def __init__(self, priority_latency_model, regular_latency_model,priority_deviation_model,regular_deviation_model, priority_objects_model, regular_objects_model, travelled_distance_model):
        super().__init__(n_var=3, n_obj=7, n_constr=0, xl=np.array([0, 0, 0]), xu=np.array([1, 1, 1]))
        self.priority_latency_model = priority_latency_model
        self.regular_latency_model = regular_latency_model

        self.priority_deviation_model = priority_deviation_model
        self.regular_deviation_model = regular_deviation_model

        self.priority_objects_model = priority_objects_model
        self.regular_objects_model = regular_objects_model

        self.travelled_distance_model = travelled_distance_model

    def _evaluate(self, x, out, *args, **kwargs):
        w1, w2, w3 = x
        priority_latency_model = self.priority_latency_model(w1, w2, w3)
        regular_latency_model = self.regular_latency_model(w1, w2, w3)

        priority_deviation_model = self.priority_deviation_model(w1, w2, w3)
        regular_deviation_model = self.regular_deviation_model(w1, w2, w3)

        priority_objects_model = self.priority_objects_model(w1, w2, w3) 
        regular_objects_model = self.regular_objects_model(w1, w2, w3)

        travelled_distance_model = self.travelled_distance_model(w1, w2, w3) 

        out["F"] = [priority_latency_model,regular_latency_model,priority_deviation_model,regular_deviation_model,-priority_objects_model,-regular_objects_model,travelled_distance_model]  # Negative for maximization

# Run NSGA-II Optimization
def run_nsga2(problem):
    algorithm = NSGA2(
        pop_size=100,
        # sampling=get_sampling("real_random"),
        # crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        # mutation=get_mutation("real_pm", eta=20),
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    res = minimize(problem, algorithm, ('n_gen', 100), seed=1, verbose=True)
    return res

# Run MOEA/D Optimization
# def run_moead(problem):
#     # algorithm = MOEAD(
#     #     n_neighbors=15,
#     #     decomposition=get_decomposition("tchebi"),
#     #     prob_neighbor_mating=0.7
#     # )
#     algorithm = MOEAD(
#         n_neighbors=15,
#         decomposition=decomposition,
#         prob_neighbor_mating=0.7
#     )
#     res = minimize(problem, algorithm, ('n_gen', 100), seed=1, verbose=True)
#     return res

# Plot Pareto Front
def plot_pareto(res):
    Scatter().add(res.F).show()

# Main Function
def main():
    df = pd.read_csv('/home/uib/MRS_data/simulation_data/owa_data.csv')  
    priority_latency_model, regular_latency_model,priority_deviation_model,regular_deviation_model, priority_objects_model, regular_objects_model, travelled_distance_model = create_interpolators(df)
    problem = MultiRobotExplorationProblem(priority_latency_model, regular_latency_model,priority_deviation_model,regular_deviation_model, priority_objects_model, regular_objects_model, travelled_distance_model)
    
    print("Running NSGA-II Optimization...")
    res_nsga2 = run_nsga2(problem)
    print("Optimal Solutions (NSGA-II):", res_nsga2.X)
    plot_pareto(res_nsga2)
    
    print("Running MOEA/D Optimization...")
    # res_moead = run_moead(problem)
    print("Optimal Solutions (MOEA/D):", res_nsga2.X)
    plot_pareto(res_nsga2)

if __name__ == "__main__":
    main()
