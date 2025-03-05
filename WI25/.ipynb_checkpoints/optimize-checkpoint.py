import time
from itertools import product

import numpy as np
import pandas as pd
import gurobipy as gb
from sklearn.linear_model import LinearRegression

# WLS credentials
WLSACCESSID = 'ccc2c36a-db14-4956-b2e3-60adc45e9957'
WLSSECRET = '1e0e3dbf-7933-44dc-8f81-e0482ded7ac8'
LICENSEID = 2586688

# Create the Gurobi environment with parameters
env = gb.Env(empty=True)  # Start with an empty environment
env.setParam('WLSACCESSID', WLSACCESSID)
env.setParam('WLSSECRET', WLSSECRET)
env.setParam('LICENSEID', LICENSEID)
env.start()

# Load data
df = pd.read_csv('GA_features.csv')

# Define constants
SOCIAL_CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
BUDGET = 130
TAU_VALUES = [0.45, 0.55, None]  # Fairness constraints for optimization

# Define columns
X_columns = ['frac_unem', 'n_poll', 'contribution', 'tweets']
count_columns = [f'registered_{category}' for category in SOCIAL_CATEGORIES]
frac_columns = [f'frac_registered_{category}' for category in SOCIAL_CATEGORIES]

# Extract features and targets
X = df[X_columns]
A_frac = df[frac_columns]
y_train = df['frac_votes'].values

# Prepare matrices and values
CALCULUS = X['frac_unem'].values
COUNSELORS = X['n_poll'].values
FRPL = np.ones_like(X['contribution'].values)
A_MATRIX = A_frac.values
TOTAL_R = df['total_registers'].values
R_COUNTS = df[count_columns].values
R_COUNTS_TOTAL = R_COUNTS.sum(axis=0)

# Load neighborhood matrices
NEIGHBOR_INDEX_MATRIX = np.load('index_matrix.npy')
NEIGHBOR_DISTANCE_MATRIX = np.load('distance_matrix.npy')

# Dimensions and interventions
NUM_SCHOOLS = X.shape[0]
NUM_NEIGHBORS = NEIGHBOR_INDEX_MATRIX.shape[1]
intervention_sample_spaces = [(0, 1)] * NUM_NEIGHBORS
POSSIBLE_INTERVENTIONS_MATRIX = np.array(list(product(*intervention_sample_spaces)))
NUM_POSSIBLE_INTERVENTIONS = POSSIBLE_INTERVENTIONS_MATRIX.shape[0]

# Define regression model features
def compute_adjusted_features(feature_values, A_frac, neighbor_distance_matrix):
    max_neighbor_influence = np.max(neighbor_distance_matrix * feature_values[:, None], axis=1).reshape(NUM_SCHOOLS, 1)
    return A_frac * max_neighbor_influence

a_max_Sij_frac_unem = compute_adjusted_features(CALCULUS, A_frac, NEIGHBOR_DISTANCE_MATRIX)
a_max_Sij_n_poll = compute_adjusted_features(COUNSELORS, A_frac, NEIGHBOR_DISTANCE_MATRIX)
a_max_Sij_contribution = compute_adjusted_features(X['contribution'].values, A_frac, NEIGHBOR_DISTANCE_MATRIX)
a_max_Sij_tweets = compute_adjusted_features(X['tweets'].values, A_frac, NEIGHBOR_DISTANCE_MATRIX)

# Combine features for regression model
X_train = np.concatenate((a_max_Sij_frac_unem, a_max_Sij_n_poll, a_max_Sij_contribution, a_max_Sij_tweets, A_frac), axis=1)

# Train regression model
linmod = LinearRegression(fit_intercept=False).fit(X_train, y_train)
model_weights = linmod.coef_
param_dims = len(SOCIAL_CATEGORIES)

# Extract regression weights
weight_dict = {
    'alpha': model_weights[:param_dims],
    'beta': model_weights[param_dims:param_dims*2],
    'gamma': model_weights[param_dims*2:param_dims*3],
    'delta': model_weights[param_dims*3:param_dims*4],
    'theta': model_weights[param_dims*4:]
}

params = pd.DataFrame(weight_dict, index=SOCIAL_CATEGORIES)
ALPHA, BETA, GAMMA, DELTA, THETA = (
    params['alpha'].values,
    params['beta'].values,
    params['gamma'].values,
    params['delta'].values,
    params['theta'].values
)

# Helper to calculate expected impact
def calculate_expected_impact(index, intervention_array, demographic_vector):
    nearest_neighbors = NEIGHBOR_INDEX_MATRIX[index, :]
    neighbor_distances = NEIGHBOR_DISTANCE_MATRIX[index, nearest_neighbors]

    frac_unem_term = np.dot(demographic_vector, ALPHA) * np.max(neighbor_distances * intervention_array)
    n_poll_term = np.dot(demographic_vector, BETA) * np.max(neighbor_distances * COUNSELORS[nearest_neighbors])
    contribution_term = np.dot(demographic_vector, GAMMA) * np.max(neighbor_distances * X['contribution'].values[nearest_neighbors])
    tweets_term = np.dot(demographic_vector, DELTA) * np.max(neighbor_distances * X['tweets'].values[nearest_neighbors])
    demographic_term = np.dot(demographic_vector, THETA)

    impact = frac_unem_term + n_poll_term + contribution_term + tweets_term + demographic_term
    return max(min(impact, 1), 0)

# Calculate all possible impacts
def calculate_all_possible_impacts(index, demographic_vector):
    possible_impacts = np.empty(len(POSSIBLE_INTERVENTIONS_MATRIX))
    for k, intervention_array in enumerate(POSSIBLE_INTERVENTIONS_MATRIX):
        possible_impacts[k] = calculate_expected_impact(index, intervention_array, demographic_vector)
    return possible_impacts

# Optimization routine
def optimize_interventions(tau_value, A_frac):
    print(f'Running optimization for tau={tau_value}')
    model = gb.Model(env=env)

    interventions = model.addVars(NUM_SCHOOLS, vtype=gb.GRB.BINARY, name="interventions")
    model.addConstr(sum(interventions.values()) <= BUDGET, "budget_constraint")

    for index in range(NUM_SCHOOLS):
        demographic_vector = A_frac.values[index, :]
        factual_impacts = calculate_all_possible_impacts(index, demographic_vector)

        auxiliary_vars = model.addVars(len(factual_impacts), obj=factual_impacts, vtype=gb.GRB.CONTINUOUS)
        model.update()

        for j, intervention in enumerate(POSSIBLE_INTERVENTIONS_MATRIX):
            for k, neighbor in enumerate(NEIGHBOR_INDEX_MATRIX[index]):
                if intervention[k] == 1:
                    model.addConstr(auxiliary_vars[j] <= interventions[neighbor])
                else:
                    model.addConstr(auxiliary_vars[j] <= 1 - interventions[neighbor])

        model.addConstr(sum(auxiliary_vars.values()) == 1)

        if tau_value is not None:
            for group_idx in range(A_frac.shape[1]):
                group_impact_diff = calculate_all_possible_impacts(index, np.eye(A_frac.shape[1])[group_idx]) - factual_impacts
                model.addConstr(
                    sum(auxiliary_vars[j] * group_impact_diff[j] for j in range(len(factual_impacts))) <= tau_value
                )

    model.setObjective(model.getObjective(), gb.GRB.MAXIMIZE)
    model.optimize()

    if model.status == gb.GRB.OPTIMAL:
        return np.array([interventions[i].X for i in range(NUM_SCHOOLS)]).astype(bool)
    else:
        raise RuntimeError("Optimization failed.")

# Run optimization
for tau_value in TAU_VALUES:
    try:
        optimal_interventions = optimize_interventions(tau_value, A_frac)
        print(f"Optimal interventions: {np.where(optimal_interventions)}")
    except RuntimeError as e:
        print(f"Optimization failed for tau={tau_value}: {e}")
