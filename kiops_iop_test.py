
"""
Vary the incomplete orthogonalization parameter (iop) and record runtime, matvecs, memory, and error.
"""

import numpy as np
import csv
import traceback
from experiment_utils import MatVecCounter, time_and_mem, dense_expm_action
import kiops_file

OUT_CSV = "expC_iop_results.csv"

# ---------- Config (tune) ----------
n = 200                   # keep moderate to avoid memory issues
t_final = 1.0
tol = 1e-6
m_init = 20
mmin = 10
mmax = 500
# list of IOP parameters to test; adapt to your implementation (1..k or 0..k)
iop_list = [1, 2, 4, 8, 16]
runs_per_q = 1            # repeat runs for statistics
# ------------------------------------

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

def build_diag_negative(n, scale=1.0):
    vals = - scale * np.linspace(1, n, n)
    return np.diag(vals)

def run_iop_study(A, b, t_final, tol, iop_list, out_csv=OUT_CSV):
    rows = []
    for q in iop_list:
        for run in range(runs_per_q):
            # instrument A
            mcounter = MatVecCounter(A)
            Aop = mcounter.to_linop()

                        u = b.reshape(-1)    
            res_k, stats_k, t_k, mem_k = safe_call_kiops([0.0, t_final], Aop, u, tol, m_init, mmin, mmax, q, 1)
            matvecs = mcounter.count

            # try to compute error if small n
            err = np.nan
            if A.shape[0] <= 400 and res_k is not None and not isinstance(res_k, Exception):
                try:
                    ref = dense_expm_action(A, b, t_final)
                    # extract kiops result safely
                    w = res_k
                    w_arr = np.asarray(w)
                    # 
                    if w_arr.ndim == 3:
                        kiops_vec = w_arr[-1, 0, :].reshape(-1)
                    elif w_arr.ndim == 2:
                        
                        kiops_vec = w_arr[-1, :].reshape(-1)
                    else:
                        kiops_vec = w_arr.reshape(-1)
                    err = float(np.linalg.norm(kiops_vec - ref))
                except Exception:
                    err = np.nan

            
            def sp(stats_obj):
                try:
                    return str(stats_obj)[:300]
                except Exception:
                    return None

            rows.append({
                'iop_q': q,
                'run': run,
                'n': A.shape[0],
                'tol': tol,
                'method': 'kiops',
                'time_s': float(t_k) if (t_k is not None and not isinstance(t_k, Exception)) else np.nan,
                'matvecs': int(matvecs),
                'peak_mem': int(mem_k) if (mem_k is not None and not isinstance(mem_k, Exception)) else np.nan,
                'error': err,
                'stats_preview': sp(stats_k)
            })
            print(f"q={q}, run={run}: time={t_k}, matvecs={matvecs}, err={err}")

    # write CSV
    keys = list(rows[0].keys()) if len(rows) > 0 else []
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {out_csv} with {len(rows)} rows.")
    return rows

if __name__ == "__main__":
    # build matrix and vector
    A = build_diag_negative(n, scale=1.0)
    rng = np.random.default_rng(0)
    b = rng.standard_normal(n)

    try:
        run_iop_study(A, b, t_final, tol, iop_list)
    except Exception:
        print("Top-level exception:")
        traceback.print_exc()
        raise