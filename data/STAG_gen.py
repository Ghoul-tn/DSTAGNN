import numpy as np
import pandas as pd
import time
import argparse
from scipy.optimize import linprog
from joblib import Parallel, delayed  # For parallel processing

def check_data(data, name):
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError(f"Data contains NaN or Inf values: {name}")

def wasserstein_distance(p, q, D):
    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q])
    D = np.array(D)
    D = D.reshape(-1)

    # Handle NaN and Inf values
    D = np.nan_to_num(D, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    A_eq = np.nan_to_num(A_eq, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    b_eq = np.nan_to_num(b_eq, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

    check_data(D, "D")
    check_data(A_eq, "A_eq")
    check_data(b_eq, "b_eq")

    result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    myresult = result.fun

    return myresult

def spatial_temporal_aware_distance(x, y):
    x, y = np.array(x), np.array(y)
    x_norm = (x**2).sum(axis=1, keepdims=True)**0.5
    y_norm = (y**2).sum(axis=1, keepdims=True)**0.5
    p = x_norm[:, 0] / x_norm.sum()
    q = y_norm[:, 0] / y_norm.sum()
    D = 1 - np.dot(x / x_norm, (y / y_norm).T)

    # Ensure D does not contain NaN or Inf values
    D = np.nan_to_num(D, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    check_data(D, "D")

    return wasserstein_distance(p, q, D)

def spatial_temporal_similarity(x, y, normal, transpose):
    # Ensure x and y are 2D arrays
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    if normal:
        x = normalize(x)
        y = normalize(y)
    if transpose:
        x = np.transpose(x)
        y = np.transpose(y)
    return 1 - spatial_temporal_aware_distance(x, y)

def normalize(a):
    mu = np.mean(a, axis=1, keepdims=True)
    std = np.std(a, axis=1, keepdims=True)
    return (a - mu) / std

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Gambia", help="Dataset path.")
parser.add_argument("--period", type=int, default=12, help="Time series period.")
parser.add_argument("--sparsity", type=float, default=0.01, help="Sparsity of spatial graph.")
args = parser.parse_args()

# Load the multivariate time series data
data = np.load("/kaggle/input/gambia-files/Gambia_UpperRiver_multivariate_data.npz")["data"]  # Shape: (num_timesteps, num_nodes, num_features)

# Extract the time series for each node (average across features)
time_series = data.mean(axis=2)  # Shape: (num_timesteps, num_nodes)

# Debug: Check the shape of time_series
print("Shape of time_series:", time_series.shape)

# If time_series is 1D, reshape it
if len(time_series.shape) == 1:
    num_timesteps = data.shape[0]
    num_nodes = data.shape[1]
    time_series = time_series.reshape(num_timesteps, num_nodes)
    print("Reshaped time_series to:", time_series.shape)

# Compute the number of nodes
num_nodes = time_series.shape[1]

# Initialize the STAG matrix
d = np.zeros([num_nodes, num_nodes])

# Function to compute similarity for a single pair (i, j)
def compute_similarity(i, j):
    return spatial_temporal_similarity(time_series[:, i], time_series[:, j], normal=False, transpose=False)

# Use joblib to parallelize the pairwise comparisons
print("Starting parallel computation of spatial-temporal similarity...")
t0 = time.time()

# Create a list of (i, j) pairs to process
pairs = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]

# Use joblib to compute similarities in parallel
results = Parallel(n_jobs=-1)(delayed(compute_similarity)(i, j) for i, j in pairs)

# Fill the STAG matrix with the results
k = 0
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        d[i, j] = results[k]
        k += 1

# Symmetrize the matrix
stag = d + d.T

# Save the STAG matrix
np.save(f"{args.dataset}/stag_{args.sparsity}_{args.dataset}.npy", stag)
print("STAG matrix calculation is done!")
t3 = time.time()
print(f'Total time: {t3 - t0} seconds.')

# Generate the adjacency matrix from STAG
adj = np.load(f"{args.dataset}/stag_{args.sparsity}_{args.dataset}.npy")
id_mat = np.identity(num_nodes)
adjl = adj + id_mat
adjlnormd = adjl / adjl.mean(axis=0)

adj = 1 - adjl + id_mat
A_adj = np.zeros([num_nodes, num_nodes])
R_adj = np.zeros([num_nodes, num_nodes])

# Apply sparsity threshold
adj_percent = args.sparsity
top = int(num_nodes * adj_percent)

for i in range(adj.shape[0]):
    a = adj[i, :].argsort()[0:top]
    for j in range(top):
        A_adj[i, a[j]] = 1
        R_adj[i, a[j]] = adjlnormd[i, a[j]]

for i in range(num_nodes):
    for j in range(num_nodes):
        if i == j:
            R_adj[i][j] = adjlnormd[i, j]

print("Total route number: ", num_nodes)
print("Sparsity of adj: ", len(A_adj.nonzero()[0]) / (num_nodes * num_nodes))

# Save the adjacency matrix
pd.DataFrame(A_adj).to_csv(f"{args.dataset}/stag_{args.sparsity}_{args.dataset}.csv", index=False, header=None)
pd.DataFrame(R_adj).to_csv(f"{args.dataset}/strg_{args.sparsity}_{args.dataset}.csv", index=False, header=None)

print("The weighted matrix of the temporal graph is generated!")
