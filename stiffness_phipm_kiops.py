#!/usr/bin/env python3

"""
Adaptivity / stiffness sweep comparing KIOPS and phipm.

- Ensures correct shapes for kiops (u as (p+1, n) or (1,n) row).
- Defensive wrappers print returned types and stats previews.
- Writes results to CSV 'expB_adaptivity_results.csv'.
"""

import numpy as np
import csv
import traceback
from experiment_utils import MatVecCounter, time_and_mem, dense_expm_action
import kiops_file
import phipm_file

OUT_CSV = "expB_adaptivity_results.csv"

# -------------- CONFIG (tune as desired) --------------
n = 120                       # small by default to avoid memory problems
t_final = 1.0
m_init = 20
mmin = 10
mmax = 500
iop_default = 2               # IOP window param for kiops
tol_list = [1e-4, 1e-6]       # tolerances to sweep
alpha_list = [1, 2,4,8,16,32,64]     # stiffness scaling
runs_per_case = 1             # repeat count for medians (set higher for final runs)
# -----------------------------------------------------

def make_u_for_kiops(u):

    arr = np.asarray(u)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    elif arr.ndim == 2:
        return arr
    else:
        raise ValueError(f"Unsupported u shape for kiops: {arr.shape}")

def safe_call_kiops(tau_out, Aop, u, tol, m_init, mmin, mmax, iop, task1):

    try:
        u_ready = make_u_for_kiops(u)
        
        (res, stats), elapsed, peak = time_and_mem(kiops_file.kiops, np.array(tau_out), Aop, u_ready, tol, m_init, mmin, mmax, iop, task1)
        return res, stats, elapsed, peak
    except Exception as e:
        return None, e, None, None

def safe_call_phipm(t_final, Aop, Vcols, tol):

    try:
        res, elapsed, peak = time_and_mem(phipm_file.phipm_matrix, t_final, Aop, Vcols, tol)
        return res, elapsed, peak
    except Exception as e:
        return None, e, None

def build_laplacian_1d(n):

    e = np.ones(n)
    T = np.diag(-2*e) + np.diag(e[:-1], 1) + np.diag(e[:-1], -1)
    return T

def run_adaptivity_sweep(A_base, b, t_final, tol_list, alpha_list, out_csv=OUT_CSV):

    rows = []
    for alpha in alpha_list:
        A = alpha * A_base
        for tol in tol_list:
            for run in range(runs_per_case):
                # Instrument A for KIOPS
                mcounter_k = MatVecCounter(A)
                Aop_k = mcounter_k.to_linop()


                u = b.reshape(-1)     # flatten to 1D; make_u_for_kiops will convert to (1,n)

                res_k, stats_k, t_k, mem_k = safe_call_kiops([0.0, t_final], Aop_k, u, tol, m_init, mmin, mmax, iop_default, 1)
                matvecs_k = mcounter_k.count

                mcounter_p = MatVecCounter(A)
                Aop_p = mcounter_p.to_linop()
                Vcols = b.reshape(-1,1)
                res_p, t_p, mem_p = safe_call_phipm(t_final, Aop_p, Vcols, tol)
                matvecs_p = mcounter_p.count


                def stats_preview(stats):
                    try:
                        return str(stats)[:300]
                    except:
                        return None


                err_k = err_p = np.nan
                if A.shape[0] <= 400:
                    try:
                        ref = dense_expm_action(A, b, t_final)

                        if res_k is not None and not isinstance(res_k, Exception):
                            try:
                                
                                w = res_k
                                
                                if hasattr(w, '__len__') and np.asarray(w).ndim >= 1:
                                    w_arr = np.asarray(w)
                                    # try to pick last time index and first phi index
                                    if w_arr.ndim == 3:
                                        kiops_vec = w_arr[-1, 0, :].reshape(-1)
                                    elif w_arr.ndim == 2:
                                        kiops_vec = w_arr[-1, :].reshape(-1)
                                    else:
                                        kiops_vec = w_arr.reshape(-1)
                                else:
                                    kiops_vec = None
                            except Exception:
                                kiops_vec = None
                            if kiops_vec is not None:
                                err_k = float(np.linalg.norm(kiops_vec - ref))
                        # phipm result
                        if res_p is not None and not isinstance(res_p, Exception):
                            try:
                                phipm_vec = np.asarray(res_p).reshape(-1)
                                err_p = float(np.linalg.norm(phipm_vec - ref))
                            except Exception:
                                err_p = np.nan
                    except Exception:
                        # dense reference failed (numerical issues); leave NaN
                        err_k = err_p = np.nan

                
                rows.append({
                    'matrix_scale_alpha': alpha,
                    'tol': tol,
                    'method': 'kiops',
                    'n': A.shape[0],
                    'time_s': float(t_k) if (t_k is not None and not isinstance(t_k, Exception)) else np.nan,
                    'matvecs': int(matvecs_k),
                    'peak_mem': int(mem_k) if (mem_k is not None and not isinstance(mem_k, Exception)) else np.nan,
                    'err': err_k,
                    'stats_preview': stats_preview(stats_k)
                })
                rows.append({
                    'matrix_scale_alpha': alpha,
                    'tol': tol,
                    'method': 'phipm',
                    'n': A.shape[0],
                    'time_s': float(t_p) if (t_p is not None and not isinstance(t_p, Exception)) else np.nan,
                    'matvecs': int(matvecs_p),
                    'peak_mem': int(mem_p) if (mem_p is not None and not isinstance(mem_p, Exception)) else np.nan,
                    'err': err_p,
                    'stats_preview': None
                })
                print(f"alpha={alpha}, tol={tol}, run={run}: kiops_time={t_k}, phipm_time={t_p}, matvecs_k={matvecs_k}, matvecs_p={matvecs_p}")

    # write CSV
    keys = list(rows[0].keys()) if len(rows) > 0 else []
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote results to {out_csv}")
    return rows

if __name__ == "__main__":
    # Build a base operator (1D Laplacian) and a random RHS
    A0 = build_laplacian_1d(n)
    # choose a random b with fixed seed
    rng = np.random.default_rng(0)
    b = rng.standard_normal(n)

    try:
        _ = run_adaptivity_sweep(A0, b, t_final, tol_list, alpha_list)
    except Exception:
        print("Top-level exception in run_adaptivity_sweep:")
        traceback.print_exc()
        raise

