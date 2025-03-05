import os
import time
from itertools import product

import numpy as np
import pandas as pd
import gurobipy as gb
from sklearn.linear_model import LinearRegression

# -----------------------------
# Setup and Data Loading
# -----------------------------

# WLS credentials
WLS_ACCESS_ID = 'ccc2c36a-db14-4956-b2e3-60adc45e9957'
WLS_SECRET = '1e0e3dbf-7933-44dc-8f81-e0482ded7ac8'
LICENSE_ID = 2586688

# Create the Gurobi environment with parameters
env = gb.Env(empty=True)
env.setParam('WLSACCESSID', WLS_ACCESS_ID)
env.setParam('WLSSECRET', WLS_SECRET)
env.setParam('LICENSEID', LICENSE_ID)
env.start()

# Load data and neighborhood matrices
df = pd.read_csv('GA_features.csv')
NEIGHBOR_INDEX_MATRIX = np.load('index_matrix.npy')
NEIGHBOR_DISTANCE_MATRIX = np.load('distance_matrix.npy')

# -----------------------------
# Constants and Columns
# -----------------------------
SOCIAL_CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
TAU_TIGHTEST = 0.4  # Tightest fairness constraint
TAU_NONE = None   # No fairness constraint

FEATURE_COLUMNS = ['frac_unem', 'n_poll', 'contribution', 'tweets']
COUNT_COLUMNS = [f'registered_{cat}' for cat in SOCIAL_CATEGORIES]
FRAC_COLUMNS = [f'frac_registered_{cat}' for cat in SOCIAL_CATEGORIES]

# -----------------------------
# Prepare Features and Targets
# -----------------------------
X = df[FEATURE_COLUMNS]
A_frac = df[FRAC_COLUMNS]
y_train = df['frac_votes'].values

# Feature values
UNEMPLOYMENT_RATE = X['frac_unem'].values
POLLING_STATIONS = X['n_poll'].values
# Use a constant vector for contribution as in the original code
CONTRIBUTION = np.ones_like(X['contribution'].values)
TWEETS = X['tweets'].values

# Neighborhood dimensions and intervention space
NUM_SCHOOLS = X.shape[0]
NUM_NEIGHBORS = NEIGHBOR_INDEX_MATRIX.shape[1]
INTERVENTION_SAMPLE_SPACES = [(0, 1)] * NUM_NEIGHBORS
POSSIBLE_INTERVENTIONS_MATRIX = np.array(list(product(*INTERVENTION_SAMPLE_SPACES)))
NUM_POSSIBLE_INTERVENTIONS = POSSIBLE_INTERVENTIONS_MATRIX.shape[0]

# -----------------------------
# Regression Model Setup
# -----------------------------
def compute_adjusted_features(feature_values, A_frac, neighbor_distance_matrix):
    # Compute the maximum influence from neighbors and multiply elementwise with A_frac
    max_neighbor_influence = np.max(neighbor_distance_matrix * feature_values[:, None], axis=1).reshape(NUM_SCHOOLS, 1)
    return A_frac * max_neighbor_influence

# Compute adjusted features for each feature
adjusted_unemployment = compute_adjusted_features(UNEMPLOYMENT_RATE, A_frac, NEIGHBOR_DISTANCE_MATRIX)
adjusted_polling = compute_adjusted_features(POLLING_STATIONS, A_frac, NEIGHBOR_DISTANCE_MATRIX)
adjusted_contribution = compute_adjusted_features(CONTRIBUTION, A_frac, NEIGHBOR_DISTANCE_MATRIX)
adjusted_tweets = compute_adjusted_features(TWEETS, A_frac, NEIGHBOR_DISTANCE_MATRIX)

# Combine features and train regression model (no intercept)
X_train = np.concatenate((adjusted_unemployment, adjusted_polling, adjusted_contribution, adjusted_tweets, A_frac), axis=1)
linear_model = LinearRegression(fit_intercept=False).fit(X_train, y_train)
model_weights = linear_model.coef_
param_dims = len(SOCIAL_CATEGORIES)

# Split regression weights into groups corresponding to each feature and demographics
weight_dict = {
    'alpha': model_weights[:param_dims],
    'beta': model_weights[param_dims:2*param_dims],
    'gamma': model_weights[2*param_dims:3*param_dims],
    'delta': model_weights[3*param_dims:4*param_dims],
    'theta': model_weights[4*param_dims:]
}
params = pd.DataFrame(weight_dict, index=SOCIAL_CATEGORIES)
ALPHA = params['alpha'].values
BETA = params['beta'].values
GAMMA = params['gamma'].values
DELTA = params['delta'].values
THETA = params['theta'].values

# -----------------------------
# Impact Calculation Functions
# -----------------------------
def calculate_expected_impact(index, intervention_array, demographic_vector):
    """
    Calculate the expected impact for a single school using the intervention decision.
    For each term, we multiply the (binary) intervention vector by the neighbor distances,
    take the maximum value, and weight it by the corresponding regression coefficient.
    """
    # Get the indices and distances of the school's neighbors
    nearest_neighbors = NEIGHBOR_INDEX_MATRIX[index, :]
    neighbor_distances = NEIGHBOR_DISTANCE_MATRIX[index, nearest_neighbors]
    
    unemployment_term = np.dot(demographic_vector, ALPHA) * np.max(neighbor_distances * intervention_array)
    polling_term = np.dot(demographic_vector, BETA) * np.max(neighbor_distances * POLLING_STATIONS[nearest_neighbors])
    contribution_term = np.dot(demographic_vector, GAMMA) * np.max(neighbor_distances * CONTRIBUTION[nearest_neighbors])
    tweets_term = np.dot(demographic_vector, DELTA) * np.max(neighbor_distances * TWEETS[nearest_neighbors])
    demographic_term = np.dot(demographic_vector, THETA)
    
    impact = unemployment_term + polling_term + contribution_term + tweets_term + demographic_term
    return max(min(impact, 1), 0)

def calculate_all_possible_impacts(index, demographic_vector):
    """
    Compute the expected impact for all possible neighbor intervention patterns.
    """
    possible_impacts = np.empty(NUM_POSSIBLE_INTERVENTIONS)
    for k, intervention_array in enumerate(POSSIBLE_INTERVENTIONS_MATRIX):
        possible_impacts[k] = calculate_expected_impact(index, intervention_array, demographic_vector)
    return possible_impacts

def calculate_total_impact(intervention_array):
    """
    Calculate the total impact over all schools given a binary intervention solution.
    """
    total_impact = 0
    for i in range(NUM_SCHOOLS):
        demographic_vector = A_frac.values[i, :]
        total_impact += calculate_expected_impact(i, intervention_array[i], demographic_vector)
    return total_impact

# -----------------------------
# Optimization Routine
# -----------------------------
def optimize_interventions(tau_value, budget):
    """
    Optimize the intervention decisions for all schools under a budget constraint.
    A fairness constraint (tau) may be imposed. For each school the model selects
    an intervention pattern (from a discrete set) that yields a factual impact,
    while the fairness constraints force the differences in impact (for each demographic group)
    to be within tau.
    """
    print(f'Running optimization for tau={tau_value} and budget={budget}')
    model = gb.Model(env=env)
    
    # Binary variables for each school
    interventions = model.addVars(NUM_SCHOOLS, vtype=gb.GRB.BINARY, name="interventions")
    model.addConstr(sum(interventions[i] for i in range(NUM_SCHOOLS)) <= budget, "budget_constraint")
    
    # For each school, create auxiliary variables for each possible intervention pattern
    for index in range(NUM_SCHOOLS):
        demographic_vector = A_frac.values[index, :]
        factual_impacts = calculate_all_possible_impacts(index, demographic_vector)
        auxiliary_vars = model.addVars(len(factual_impacts), obj=factual_impacts, vtype=gb.GRB.CONTINUOUS, name=f"aux_{index}")
        model.update()
        
        # Link the auxiliary variables to the intervention decisions of the neighbors
        for j, intervention_pattern in enumerate(POSSIBLE_INTERVENTIONS_MATRIX):
            for k, neighbor in enumerate(NEIGHBOR_INDEX_MATRIX[index, :]):
                if intervention_pattern[k] == 1:
                    model.addConstr(auxiliary_vars[j] <= interventions[int(neighbor)])
                else:
                    model.addConstr(auxiliary_vars[j] <= 1 - interventions[int(neighbor)])
        model.addConstr(sum(auxiliary_vars[j] for j in range(len(factual_impacts))) == 1)
        
        # If a fairness constraint is imposed, add constraints on group impact differences
        if tau_value is not None:
            for group_idx in range(A_frac.shape[1]):
                group_indicator = np.eye(A_frac.shape[1])[group_idx]
                group_impact_diff = calculate_all_possible_impacts(index, group_indicator) - factual_impacts
                model.addConstr(
                    sum(auxiliary_vars[j] * group_impact_diff[j] for j in range(len(factual_impacts))) <= tau_value
                )
    
    model.setObjective(model.getObjective(), gb.GRB.MAXIMIZE)
    model.optimize()
    
    if model.status == gb.GRB.OPTIMAL:
        solution = np.array([interventions[i].X for i in range(NUM_SCHOOLS)]).astype(bool)
        return solution
    else:
        raise RuntimeError("Optimization failed.")

# -----------------------------
# Comparison Routine
# -----------------------------
def compare_solutions(budget):
    """
    Run optimization for a given budget using the tight tau and no tau,
    then compare the solutions and print the corresponding impacts.
    """
    print(f"\nComparing solutions for budget = {budget}")
    try:
        tight_solution = optimize_interventions(TAU_TIGHTEST, budget)
        tight_impact = calculate_total_impact(tight_solution)
    except Exception as e:
        print(f"Optimization failed for TAU_TIGHTEST with budget {budget}: {e}")
        return None, None, None
    
    try:
        no_tau_solution = optimize_interventions(TAU_NONE, budget)
        no_tau_impact = calculate_total_impact(no_tau_solution)
    except Exception as e:
        print(f"Optimization failed for TAU_NONE with budget {budget}: {e}")
        return None, None, None
    
    print(f"Tight constraint (tau={TAU_TIGHTEST}) solution total impact: {tight_impact}")
    print(f"No tau solution total impact: {no_tau_impact}")
    different = not np.array_equal(tight_solution, no_tau_solution)
    if different:
        print("The solutions are different.")
    else:
        print("The solutions are the same.")
    return tight_solution, no_tau_solution, different

# -----------------------------
# Main Loop: Run for Tight Tau (0.4) and No Tau
# -----------------------------
BUDGET = 120  # Fixed budget for testing
TAU_VALUES = [0.1, 0.2, 0.3]  # Compare tight tau and no tau

# Store solutions for comparison
solutions = {}

# Run optimization for each tau value
for tau_value in TAU_VALUES:
    print("=" * 80)
    print(f"Running optimization for tau={tau_value} and budget={BUDGET}")
    
    try:
        # Optimize interventions
        optimal_interventions = optimize_interventions(tau_value, BUDGET)
        solutions[tau_value] = optimal_interventions
        
        # Print optimal intervention locations
        print(f"Optimal interventions (tau={tau_value}): {np.where(optimal_interventions)}")
        
        # Calculate and print total impact
        total_impact = calculate_total_impact(optimal_interventions)
        print(f"Total impact (tau={tau_value}): {total_impact}")
    except RuntimeError as e:
        print(f"Optimization failed for tau={tau_value}: {e}")

# Compare solutions
if 0.4 in solutions and None in solutions:
    print("\nComparing solutions:")
    
    # Get solutions
    tight_tau_solution = solutions[0.4]
    no_tau_solution = solutions[None]
    
    # Calculate differences
    differences = np.sum(tight_tau_solution != no_tau_solution)
    print(f"Differences between tight tau (0.4) and no tau: {differences}")
    
    # Save solutions as .npy files
    np.save('tight_tau_solution.npy', tight_tau_solution)
    np.save('no_tau_solution.npy', no_tau_solution)
    print("Solutions saved as 'tight_tau_solution.npy' and 'no_tau_solution.npy'.")
else:
    print("Not enough solutions to compare.")