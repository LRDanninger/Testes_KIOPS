# experiment_utils.py
import time
import tracemalloc
import csv
import numpy as np
from scipy import linalg as la
from scipy.sparse.linalg import LinearOperator

def time_and_mem(fn, *args, **kwargs):

    tracemalloc.start()
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return res, elapsed, peak

class MatVecCounter:
    def __init__(self, A):
        self.A = A
        self.count = 0
    def matvec(self, x):
        self.count += 1
        
        return self.A.dot(x) if hasattr(self.A, "dot") else self.A @ x
    def to_linop(self):
        n = self.A.shape[0]
        return LinearOperator(dtype=self.A.dtype, shape=(n, n), matvec=self.matvec)

def dense_expm_action(A, v, t=1.0):
    
    return la.expm(t * A) @ v

def save_csv(rows, filename):
    if len(rows) == 0:
        return
    keys = list(rows[0].keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)
