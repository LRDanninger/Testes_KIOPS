import numpy as np
import numpy
import math
import scipy.linalg
from scipy.linalg import expm
from matplotlib import pyplot as plt
from numpy import linalg as LA
import mpi4py.MPI
import time
import psutil
import os

"""
   kiops(tstops, A, u; kwargs...) -> (w, stats)

Evaluate a linear combinaton of the ``φ`` functions evaluated at ``tA`` acting on
vectors from ``u``, that is

```math
  w(i) = φ_0(t[i] A) u[0, :] + φ_1(t[i] A) u[1, :] + φ_2(t[i] A) u[2, :] + ...
```

The size of the Krylov subspace is changed dynamically during the integration.
The Krylov subspace is computed using the incomplete orthogonalization method.

Arguments:
  - `τ_out`    - Array of `τ_out`
  - `A`        - the matrix argument of the ``φ`` functions
  - `u`        - the matrix with rows representing the vectors to be multiplied by the ``φ`` functions

Optional arguments:
  - `tol`      - the convergence tolerance required (default: 1e-7)
  - `mmin`, `mmax` - let the Krylov size vary between mmin and mmax (default: 10, 128)
  - `m`        - an estimate of the appropriate Krylov size (default: mmin)
  - `iop`      - length of incomplete orthogonalization procedure (default: 2)
  - `task1`     - if true, divide the result by 1/T**p

Returns:
  - `w`      - the linear combination of the ``φ`` functions evaluated at ``tA`` acting on the vectors from ``u``
  - `stats[0]` - number of substeps
  - `stats[1]` - number of rejected steps
  - `stats[2]` - number of Krylov steps
  - `stats[3]` - number of matrix exponentials
  - `stats[4]` - Error estimate
  - `stats[5]` - the Krylov size of the last substep

`n` is the size of the original problem
`p` is the highest index of the ``φ`` functions

References:
* Gaudreault, S., Rainwater, G. and Tokman, M., 2018. KIOPS: A fast adaptive Krylov subspace solver for exponential integrators. Journal of Computational Physics. Based on the PHIPM and EXPMVP codes (http://www1.maths.leeds.ac.uk/~jitse/software.html). https://gitlab.com/stephane.gaudreault/kiops.
* Niesen, J. and Wright, W.M., 2011. A Krylov subspace method for option pricing. SSRN 1799124
* Niesen, J. and Wright, W.M., 2012. Algorithm 919: A Krylov subspace algorithm for evaluating the ``φ``-functions appearing in exponential integrators. ACM Transactions on Mathematical Software (TOMS), 38(3), p.22
"""

def kiops(τ_out, A, u, tol = 1e-7, m_init = 10, mmin = 10, mmax = 128, iop = 2, task1 = False):

   ppo, n = u.shape
   p = ppo - 1

   if p == 0:
      p = 1
      # Add extra column of zeros
      u = numpy.row_stack((u, numpy.zeros(len(u))))

   # We only allow m to vary between mmin and mmax
   m = max(mmin, min(m_init, mmax))

   # Preallocate matrix
   V = numpy.zeros((mmax + 1, n + p))
   H = numpy.zeros((mmax + 1, mmax + 1))

   step    = 0
   krystep = 0
   ireject = 0
   reject  = 0
   exps    = 0
   sgn     = numpy.sign(τ_out[-1])
   τ_now   = 0.0
   τ_end   = abs(τ_out[-1])
   happy   = False
   j       = 0

   conv    = 0.0

   numSteps = len(τ_out)

   # Initial condition
   w = numpy.zeros((numSteps, n))
   w[0, :] = u[0, :].copy()

   # compute the 1-norm of u
   local_nrmU = numpy.sum(abs(u[1:, :]), axis=1)
   normU = numpy.amax( mpi4py.MPI.COMM_WORLD.allreduce(local_nrmU) )

   # Normalization factors
   if ppo > 1 and normU > 0:
      ex = math.ceil(math.log2(normU))
      nu = 2**(-ex)
      mu = 2**(ex)
   else:
      nu = 1.0
      mu = 1.0

   # Flip the rest of the u matrix
   u_flip = nu * numpy.flipud(u[1:, :])

   # Compute and initial starting approximation for the step sizedef cn(i):

   τ = τ_end

   # Setting the safety factors and tolerance requirements
   if τ_end > 1:
      γ = 0.2
      γ_mmax = 0.1
   else:
      γ = 0.9
      γ_mmax = 0.6

   delta = 1.4

   # Used in the adaptive selection
   oldm = -1; oldτ = math.nan; ω = math.nan
   orderold = True; kestold = True

   l = 0

   while τ_now < τ_end:

      # Compute necessary starting information
      if j == 0:

         H[:,:] = 0.0

         V[0, 0:n] = w[l, :]

         # Update the last part of w
         for k in range(p-1):
            i = p - k + 1
            V[j, n+k] = (τ_now**i) / math.factorial(i) * mu
         V[j, n+p-1] = mu

         # Normalize initial vector (this norm is nonzero)
         local_sum = V[0, 0:n] @ V[0, 0:n]
         β = math.sqrt( mpi4py.MPI.COMM_WORLD.allreduce(local_sum) + V[j, n:n+p] @ V[j, n:n+p] )

         # The first Krylov basis vector
         V[j, :] /= β

      # Incomplete orthogonalization process
      while j < m:

         j = j + 1

         # Augmented matrix - vector product
         V[j, 0:n    ] = A( V[j-1, 0:n] ) + V[j-1, n:n+p] @ u_flip
         V[j, n:n+p-1] = V[j-1, n+1:n+p]
         V[j, -1     ] = 0.0

         # Classical Gram-Schmidt
         ilow = max(0, j - iop)
         local_sum = V[ilow:j, 0:n] @ V[j, 0:n]
         H[ilow:j, j-1] = mpi4py.MPI.COMM_WORLD.allreduce(local_sum) + V[ilow:j, n:n+p] @ V[j, n:n+p]
         V[j, :] = V[j, :] - V[ilow:j,:].T @ H[ilow:j, j-1]

         local_sum = V[j, 0:n] @ V[j, 0:n]
         nrm = numpy.sqrt( mpi4py.MPI.COMM_WORLD.allreduce(local_sum) + V[j, n:n+p] @ V[j, n:n+p])

         # Happy breakdown
         if nrm < tol:
            happy = True
            break

         H[j, j-1] = nrm
         V[j, :]   = (1.0 / nrm) * V[j, :]

         krystep += 1

      # To obtain the phi_1 function which is needed for error estimate
      H[0, j] = 1.0

      # Save h_j+1,j and remove it temporarily to compute the exponential of H
      nrm = H[j, j-1]
      H[j, j-1] = 0.0

      # Compute the exponential of the augmented matrix
      F = scipy.linalg.expm(sgn * τ * H[0:j + 1, 0:j + 1])
      exps += 1

      # Restore the value of H_{m+1,m}
      H[j, j-1] = nrm

      if happy is True:
         # Happy breakdown wrap up
         ω     = 0.
         err   = 0.
         τ_new = min(τ_end - (τ_now + τ), τ)
         m_new = m
         happy = False

      else:

         # Local truncation error estimation
         err = abs(β * nrm * F[j-1, j])

         # Error for this step
         oldω = ω
         ω = τ_end * err / (τ * tol)

         # Estimate order
         if m == oldm and τ != oldτ and ireject >= 1:
            order = max(1, math.log(ω/oldω) / math.log(τ/oldτ))
            orderold = False
         elif orderold is True or ireject == 0:
            orderold = True
            order = j/4
         else:
            orderold = True

         # Estimate k
         if m != oldm and τ == oldτ and ireject >= 1:
            kest = max(1.1, (ω/oldω)**(1/(oldm-m)))
            kestold = False
         elif kestold is True or ireject == 0:
            kestold = True
            kest = 2
         else:
            kestold = True

         if ω > delta:
            remaining_time = τ_end - τ_now
         else:
            remaining_time = τ_end - (τ_now + τ)

         # Krylov adaptivity

         same_τ = min(remaining_time, τ)
         τ_opt  = τ * (γ / ω)**(1 / order)
         τ_opt  = min(remaining_time, max(τ/5, min(5*τ, τ_opt)))

         m_opt = math.ceil(j + math.log(ω / γ) / math.log(kest))
         m_opt = max(mmin, min(mmax, max(math.floor(3/4*m), min(m_opt, math.ceil(4/3*m)))))

         if j == mmax:
            if ω > delta:
               m_new = j
               τ_new = τ * (γ_mmax / ω)**(1 / order)
               τ_new = min(τ_end - τ_now, max(τ/5, τ_new))
            else:
               τ_new = τ_opt
               m_new = m
         else:
            m_new = m_opt
            τ_new = same_τ

      # Check error against target
      if ω <= delta:

         # Yep, got the required tolerance; update
         reject += ireject
         step   += 1

         # Udate for τ_out in the interval (τ_now, τ_now + τ)
         blownTs = 0
         nextT = τ_now + τ
         for k in range(l, numSteps):
            if abs(τ_out[k]) < abs(nextT):
               blownTs += 1

         if blownTs != 0:
            # Copy current w to w we continue with.
            w[l+blownTs, :] = w[l, :].copy()

            for k in range(blownTs):
               τPhantom = τ_out[l+k] - τ_now
               F2 = scipy.linalg.expm(sgn * τPhantom * H[0:j, :j])
               w[l+k, :] = β * F2[:j, 0] @ V[:j, :n]

            # Advance l.
            l += blownTs

         # Using the standard scheme
         w[l, :] = β * F[:j, 0] @ V[:j, :n]

         # Update τ_out
         τ_now += τ

         j = 0
         ireject = 0

         conv += err

      else:
         # Nope, try again
         ireject += 1

         # Restore the original matrix
         H[0, j] = 0.0


      oldτ = τ
      τ    = τ_new

      oldm = m
      m    = m_new


   if task1 is True:
      for k in range(numSteps):
         w[k, :] = w[k, :] / τ_out[k]

   m_ret=m

   stats = (step, reject, krystep, exps, conv, m_ret)

   return w, stats

def resposta(t,A,vetorInicial):

  return np.dot(expm(t*A),vetorInicial)

def funcaoF(x):

  return 0 

def funcaoG(x):

    return np.sin(x)


def A(x):
  tamanhoN=int(len(x)/2)

  k=(b-a)/(tamanhoN+1)
  c=1
  vetorY=np.zeros(2*tamanhoN)
  vetorY[0:tamanhoN]=x[tamanhoN:2*tamanhoN]
  vetorY[tamanhoN]=(c**2/k**2)*(-2*x[0]+x[1])
  vetorY[tamanhoN+1:2*tamanhoN-1]=(c**2/k**2)*(-2*x[1:tamanhoN-1]+x[0:tamanhoN-2]+x[2:tamanhoN])
  vetorY[-1]=(c**2/k**2)*(-2*x[-1]+x[-2])


  return vetorY
def next_step_of_time(uold,unow, k, h, c=2.0):
    # Number of points
    nx = len(unow)
    
    # Calculate the next step of time
    un = np.zeros(nx)
    for i in range(1, nx-1):
        un[i] = 2*unow[i]-uold[i]+(c**2*h**2/k**2)*(unow[i+1]-2*unow[i]+unow[i-1])
        
    return un

def cn(i):
   return (320*np.sin(i*np.pi/2)/(i*i*i*np.pi*np.pi*np.pi))

def solucao_onda(x,t,L):

  return np.sin(x)*np.sin(t)

a=0
b=2*np.pi
n=4096
k=(b-a)/(n+1)
#h=1
discretizaçãoEixoX=np.linspace(a,b,n+2)
#print(discretizaçãoEixoX)
#print(h)
c=1


print("Running")
vetorYInicial=np.zeros(n*2)
contVetorInicial=0

for i in range (1,n+1):
  vetorYInicial[contVetorInicial]=funcaoF(discretizaçãoEixoX[i])
  vetorYInicial[contVetorInicial+n]=funcaoG(discretizaçãoEixoX[i])
  contVetorInicial=1+contVetorInicial


process = psutil.Process(os.getpid())
 
total_start_time = time.perf_counter()
total_start_mem = process.memory_info().rss  # in bytes

for tempo in range (1,2):
   start_time = time.perf_counter()
   start_mem = process.memory_info().rss
   peak_mem = start_mem
   tempo=tempo*7
   sol=solucao_onda(discretizaçãoEixoX,tempo,2*np.pi)
   U=np.zeros((2,n*2))
   U[0]=vetorYInicial
   d,stats=kiops(numpy.array([tempo,tempo]), A, U, 1e-7, 2, 2, 5, 2,False)
   expkiops=d[0]
   diff=(expkiops[0:n]-sol[1:len(discretizaçãoEixoX)-1])
   print(diff.max())
   end_time = time.perf_counter()
   end_mem = process.memory_info().rss
   print(f"Iteration {tempo+1}:")
   print(f"   Time: {(end_time - start_time):.4f} seconds")
   print(f"   Memory change: {(end_mem - start_mem)/1024:.2f} KB")
   print(f"   Peak memory during iteration: {(peak_mem - start_mem)/1024:.2f} KB above start")
   print()

total_end_time = time.perf_counter()
total_end_mem = process.memory_info().rss

print("=== TOTAL LOOP STATS ===")
print(f"Total Time: {(total_end_time - total_start_time):.4f} seconds")
print(f"Total Memory change: {(total_end_mem - total_start_mem)/1024:.2f} KB")

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

x = discretizaçãoEixoX[1:-1]
t_str = f"{tempo:.3f}"   # clean filename

# ================= KIOPS =================
plt.figure(figsize=(12, 6))
plt.plot(x, expkiops[0:n], linewidth=2)
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title(f"KIOPS approximation for t = {tempo}")
plt.grid(True)

plt.savefig(f"KIOPS_t_1D_{t_str}.png", dpi=300, bbox_inches="tight")
plt.show()


# ================= Exact solution =================
plt.figure(figsize=(12, 6))
plt.plot(x, sol[1:len(discretizaçãoEixoX)-1], linewidth=2)
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title(f"Wave Equation for t = {tempo}")
plt.grid(True)

plt.savefig(f"Result_Wave_t_1D_{t_str}.png", dpi=300, bbox_inches="tight")
plt.show()


# ================= Error =================
plt.figure(figsize=(12, 6))
plt.plot(x, diff, linewidth=2)
plt.xlabel("x")
plt.ylabel("Error")
plt.title(f"KIOPS error for t = {tempo}")
plt.grid(True)

plt.savefig(f"Error_t_1D_{t_str}.png", dpi=300, bbox_inches="tight")
plt.show()
