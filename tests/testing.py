import torch
from test_coOccurMetrics import test_data, test_metrics


data = test_data()
metrics = test_metrics()
A = data["P"]
T = data["M2"]
metric_dba = metrics["DBA"]
probs = metric_dba.computePairProbs(A, T)

# Count 1s and 0s in A
A_ones = torch.sum(A == 1).item()
A_zeros = torch.sum(A == 0).item()
A_total = A.numel()

# Count 1s and 0s in T
T_ones = torch.sum(T == 1).item()
T_zeros = torch.sum(T == 0).item()
T_total = T.numel()

print(f"A - Total elements: {A_total}")
print(f"A - Number of 1s: {A_ones}")
print(f"A - Number of 0s: {A_zeros}")
print(f"A - Shape: {A.shape}")
print(f"A - Sum check (should equal total): {A_ones + A_zeros}")

print(f"\nT - Total elements: {T_total}")
print(f"T - Number of 1s: {T_ones}")
print(f"T - Number of 0s: {T_zeros}")
print(f"T - Shape: {T.shape}")
print(f"T - Sum check (should equal total): {T_ones + T_zeros}")