import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator
from numpy.linalg import norm

# Este código implementa uma versão  do método phipm para calcular a # ação da exponencial de uma matriz sobre um vetor, ou seja, e^{tA}v. O método
# Ele utiliza projeção em subespaços de Krylov construídos pelo processo de Arnoldi  para reduzir o problema a uma matriz pequena de Hessenberg, cuja exponencial
# pode ser calculada de forma eficiente. O algoritmo inclui um estimador de erro  e aumenta adaptativamente a dimensão do subespaço de Krylov até que a tolerância
# especificada seja satisfeita. O código é voltado para o caso homogêneo x' = Ax  (p = 0) e pode trabalhar tanto com matrizes densas quanto com LinearOperator.
# Essa estrutura foi retirada do artigo de phipm definido na tese

def arnoldi(Aop, v, m):
   
    v = np.asarray(v).ravel()
    n = v.shape[0]
    V = np.zeros((n, m+1))
    H = np.zeros((m+1, m))
    beta = norm(v)
    if beta == 0.0:
        return V[:, :1], H[:1, :0], 0.0
    V[:, 0] = v / beta
    for j in range(m):
        w = Aop @ V[:, j]
        for i in range(j+1):
            H[i, j] = np.dot(V[:, i], w)
            w = w - H[i, j] * V[:, i]
        H[j+1, j] = norm(w)
        if H[j+1, j] == 0.0:
            return V[:, :j+1], H[:j+2, :j+1], beta
        V[:, j+1] = w / H[j+1, j]
    return V, H, beta


def phipm_matrix(t_final, A, Vcols, tol=1e-8, symmetric=False, m_init=30, debug=False, m_max=None):
    """
    Matrix-input phipm focused on the homogeneous case p=0 (x' = A x).
    - t_final: scalar final time
    - A: ndarray (n x n) or LinearOperator
    - Vcols: ndarray (n, p+1); for homogeneous p=0 -> Vcols = x0.reshape(-1,1)
    - tol: target tolerance for the Krylov approximation (absolute)
    - m_init: initial Krylov subspace size
    - m_max: optional max Krylov size (defaults to n)
    """
    # wrap A into LinearOperator if needed
    if isinstance(A, LinearOperator):
        Aop = A
        # try to infer n
        n = Aop.shape[0]
    else:
        A_mat = np.asarray(A)
        n = A_mat.shape[0]
        Aop = LinearOperator((n, n), matvec=lambda v: A_mat.dot(v))

    n_v, p1 = Vcols.shape
    assert n_v == n, "A and Vcols size mismatch"
    p = p1 - 1
    if p != 0:
        raise NotImplementedError("This simplified function currently supports only p=0 (homogeneous x'=Ax).")

    y = Vcols[:, 0].copy()
    if m_max is None:
        m_max = n

    # Single-step (attempt to reach t_final). Use inner adaptivity on m.
    tau = t_final

    # initial m
    m = max(1, int(m_init))

    # Outer attempts (increase m on rejection)
    while True:
        # Build Krylov of size m for vector y (seed = y)
        Vm, Hm, beta = arnoldi(Aop, y, m)   # Hm shape (m+1, m)
        mm = Hm.shape[1]  # actual Krylov dim (<= m)

        if mm == 0:
            # trivial case: y was zero
            return np.zeros_like(y)

        # Build small H_m (mm x mm)
        Hsmall = Hm[:mm, :mm]   # m x m
        # Compute exp(tau * Hsmall)
        E = expm(tau * Hsmall)  # mm x mm

        # Action: approximate y_new ≈ beta * Vm[:, :mm] @ (E @ e1)
        e1 = np.zeros(mm); e1[0] = 1.0
        phi = E @ e1           # column = exp(tau H) e1
        y_new = beta * (Vm[:, :mm] @ phi)

        # Error estimator: eps ≈ beta * h_{m+1,m} * (e_m^T * exp(tau H) * e1)
        h_m1m = Hm[mm, mm-1] if (Hm.shape[0] > mm and mm > 0) else 0.0
        e_m_T_E_e1 = E[mm-1, 0]  # (last row, first col)
        eps = abs(beta * h_m1m * e_m_T_E_e1)

        if debug:
            print(f"[phipm_matrix] m={m}, mm={mm}, beta={beta:.3e}, h_m1m={h_m1m:.3e}, eps={eps:.3e}")

        # Acceptance: use eps <= tol (absolute). Optionally relative test: eps <= tol * ||y_new||
        if eps <= tol:
            return y_new
        # else increase m and retry
        # grow but cap at m_max
        if m >= m_max:
            # fallback: compute full expm of dense A (guaranteed but expensive)
            try:
                A_dense = A if not isinstance(A, LinearOperator) else None
                if A_dense is None:
                    # attempt to form dense by matvecing basis vectors
                    A_dense = np.zeros((n, n))
                    for i in range(n):
                        ei = np.zeros(n); ei[i] = 1.0
                        A_dense[:, i] = Aop @ ei
                y_full = expm(t_final * A_dense) @ Vcols[:, 0]
                if debug:
                    print("phipm_matrix: reached m_max, used dense fallback.")
                return y_full
            except Exception:
                raise RuntimeError("phipm_matrix failed and dense fallback unavailable.")
        # increase m: conservative growth
        m = min(m * 2, m + 10, m_max)

