import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


def build_test_A(n=12):
    np.random.seed(1)
    M = np.random.randn(n, n)
    Q, _ = la.qr(M)
    eigs = np.linspace(1, 5, n)
    A =  -(Q @ np.diag(eigs) @ Q.T)
    return A


def build_augmented(A, b_list):
    N = A.shape[0]
    b0 = b_list[0]
    bp_to_b1 = b_list[1:][::-1]  # [bp,...,b1]
    p = len(bp_to_b1)

    if p == 0:
        Atilde = A
        v = b0.copy()
        return Atilde, v

    B = np.column_stack(bp_to_b1)
    K = np.zeros((p, p))
    for i in range(p - 1):
        K[i, i + 1] = 1.0

    Atilde = np.block([
        [A, B],
        [np.zeros((p, A.shape[1])), K]
    ])

    e_p = np.zeros(p)
    e_p[-1] = 1.0
    v = np.concatenate([b0, e_p])
    return Atilde, v


def compute_apriori_bound(tau, Atilde, Hm, m, beta):
    eigs_Atilde = la.eigvals(tau * Atilde)
    alpha_Atilde = np.max(np.real(eigs_Atilde))
    alpha_Hm = np.max(np.real(la.eigvals(Hm)))
    norm_tauA = la.norm(tau * Atilde, 2)
    norm_Hm = la.norm(Hm, 2)

    term1 = np.exp(alpha_Atilde) * (norm_tauA**m) / np.math.factorial(m) * beta
    term2 = np.exp(alpha_Hm) * (norm_Hm**m) / np.math.factorial(m) * beta
    return term1 + term2


# -----------------------------------------------------------
# Full Arnoldi to get Vm, Hm, h_{m+1,m}
# -----------------------------------------------------------
def arnoldi(A, v, m):
    n = A.shape[0]
    V = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))
    beta = la.norm(v)
    V[:, 0] = v / beta

    for j in range(m):
        w = A @ V[:, j]
        for i in range(j + 1):
            H[i, j] = np.dot(V[:, i], w)
            w -= H[i, j] * V[:, i]
        H[j + 1, j] = la.norm(w)
        if H[j + 1, j] == 0:
            break
        V[:, j + 1] = w / H[j + 1, j]
    return V, H, beta



def posterior_estimate_eq36(tau, H, h_next, beta, m):
    """
    Implements ε_m = β |h_{m+1,m}| * | (exp(tau * H_hat))_{m, m+1} |
    H is (m+1)x m Hessenberg.
    """
    Hm = H[:m, :m]
    h_mp1_m = h_next

    # Build H_hat (eq. 37)
    Hhat = np.zeros((m + 1, m + 1))
    Hhat[:-1, :-1] = Hm
    Hhat[:-1, -1] = np.eye(m)[:, -1]  # last column is e_m

    # Compute exp
    E = la.expm(tau * Hhat)

    entry = E[m - 1, m]   # (m, m+1) in 1-based
    eps_m = beta * abs(h_mp1_m) * abs(entry)
    return eps_m



def main():
    N = 12
    A = build_test_A(N)

    np.random.seed(3)
    b0 = np.random.randn(N)
    b1 = np.random.randn(N) * 0.1
    b2 = np.random.randn(N) * 0.01

    Atilde, v = build_augmented(A, [b0, b1, b2])
    tau = 1.0

    # Exact solution
    w_exact = la.expm(tau * Atilde) @ v
    w_exact_firstN = w_exact[:N]

    m_values = list(range(3, 16))
    errs = []
    apriori_vals = []
    posterior_vals = []

    for m in m_values:
        V, H, beta = arnoldi(Atilde, v, m)
        Hm = H[:m, :m]
        h_next = H[m, m - 1]

        # Krylov approximation
        approx = beta * V[:N, :m] @ (la.expm(tau * Hm)[:, 0])
        err = la.norm(w_exact_firstN - approx)

        # a-priori
        apr = compute_apriori_bound(tau, Atilde, Hm, m, beta)

        # a-posteriori eq. (36)
        post = posterior_estimate_eq36(tau, H, h_next, beta, m)

        errs.append(err)
        apriori_vals.append(apr)
        posterior_vals.append(post)

    # ---- Plot ----
    plt.figure(figsize=(8, 5))
    plt.loglog(m_values, errs, "o-", label="Actual error")
    plt.loglog(m_values, posterior_vals, "d--", label="Posterior estimate ")
    plt.loglog(m_values, apriori_vals, "s:", label="A priori bound")

    plt.xlabel("Krylov dimension m")
    plt.ylabel("Error / Estimate (log scale)")
    plt.title("Actual error vs A-priori  vs A-posteriori for negative eigenvalues")
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n m | actual error | posterior (36) | apriori (34)")
    for m, e, po, ap in zip(m_values, errs, posterior_vals, apriori_vals):
        print(f"{m:2d} | {e:.3e} | {po:.3e} | {ap:.3e}")


if __name__ == "__main__":
    main()
