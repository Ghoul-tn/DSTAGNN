#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
import time
import argparse
import numpy as np
from scipy.optimize import linprog

np.seterr(divide='ignore', invalid='ignore')

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

    print("D shape:", D.shape, "Sample:", D[:10])
    print("A_eq shape:", A_eq.shape, "Sample:", A_eq[:10])
    print("b_eq shape:", b_eq.shape, "Sample:", b_eq[:10])

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
    if normal:
        x = normalize(x)
        y = normalize(y)
    if transpose:
        x = np.transpose(x)
        y = np.transpose(y)
    return 1 - spatial_temporal_aware_distance(x, y)

def gen_data(data, ntr, N):
    '''
    if flag:
        data=pd.read_csv(fname)
    else:
        data=pd.read_csv(fname,header=None)
    '''
    data=np.reshape(data,[-1,288,N])
    return data[0:ntr]

def normalize(a):
    mu=np.mean(a,axis=1,keepdims=True)
    std=np.std(a,axis=1,keepdims=True)
    return (a-mu)/std


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="PEMS03", help="Dataset path.")
parser.add_argument("--period", type=int, default=288, help="Time series perios.")
parser.add_argument("--sparsity", type=float, default=0.01, help="sparsity of spatial graph")

args = parser.parse_args()

# Load the dataset
data = np.load(args.dataset)['data']  # Load the .npz file directly

num_samples, ndim, _ = data.shape
num_train = int(num_samples * 0.6)
num_sta = int(num_train / args.period) * args.period
data = data[:num_sta, :, :1].reshape([-1, args.period, ndim])

d = np.zeros([ndim, ndim])
t0 = time.time()
for i in range(ndim):
    t1 = time.time()
    for j in range(i + 1, ndim):
        d[i, j] = spatial_temporal_similarity(data[:, :, i], data[:, :, j], normal=False, transpose=False)
        print('\r', j, end='', flush=True)
    t2 = time.time()
    print('Line', i, 'finished in', t2 - t1, 'seconds.')

sta = d + d.T

np.save("stag_001_Gambia.npy", sta)
print("The calculation of time series is done!")
t3 = time.time()
print('total finished in', t3 - t0, 'seconds.')
adj = np.load("stag_001_Gambia.npy")
id_mat = np.identity(ndim)
adjl = adj + id_mat
adjlnormd = adjl / adjl.mean(axis=0)

adj = 1 - adjl + id_mat
A_adj = np.zeros([ndim, ndim])
R_adj = np.zeros([ndim, ndim])
# A_adj = adj
adj_percent = args.sparsity

top = int(ndim * adj_percent)

for i in range(adj.shape[0]):
    a = adj[i, :].argsort()[0:top]
    for j in range(top):
        A_adj[i, a[j]] = 1
        R_adj[i, a[j]] = adjlnormd[i, a[j]]

for i in range(ndim):
    for j in range(ndim):
        if i == j:
            R_adj[i][j] = adjlnormd[i, j]

print("Total route number: ", ndim)
print("Sparsity of adj: ", len(A_adj.nonzero()[0]) / (ndim * ndim))

pd.DataFrame(A_adj).to_csv("stag_001_Gambia.csv", index=False, header=None)
pd.DataFrame(R_adj).to_csv("strg_001_Gambia.csv", index=False, header=None)

print("The weighted matrix of temporal graph is generated!")
