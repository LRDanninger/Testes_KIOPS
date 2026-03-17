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

   # Compute and initial starting approximation for the step size
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
  #matriz=np.array([[np.exp(1*t),0],[0,np.exp(10*t)]])

  return np.dot(expm(t*A),vetorInicial)

def funcaoF(x,y):
  return 0



def funcaoG(x,y):
  return np.sin(x)*np.sin(y)

def calculate_matrix(x, y, function):
    matrix = []
    for i in range(len(x)):
        row = []
        for j in range(len(y)):
            result = function(x[i], y[j])
            row.append(result)
        matrix.append(row)
    return matrix




def vector_to_square_matrix(vector):

    matrix_size = int(np.sqrt(len(vector)))
    # Reshape the vector into a matrix
    square_matrix = np.reshape(vector, (matrix_size, matrix_size))

    return square_matrix
def append_zeros_to_matrix(matrix):


    rows, cols = matrix.shape

    new_rows = rows + 2
    new_cols = cols + 2
    new_matrix = np.zeros((new_rows, new_cols), dtype=matrix.dtype)
    new_matrix[1:-1, 1:-1] = matrix

    return new_matrix

def remove_edges_from_matrix(matrix):

    rows, cols = matrix.shape
    # Remove the first and last row and column
    new_matrix = matrix[1:-1, 1:-1]

    return new_matrix

def A3(x): 
  tamanhoN=int(len(x)/2)
  N=int(np.sqrt(tamanhoN))
  raizN=int(np.sqrt(tamanhoN))
  #print(raizN)
  k1=(b-a)/(raizN+1)
  k2=(e-d)/(raizN+1)
  #print(k1)
  c=1/np.sqrt(2)
  vetorY=np.zeros(2*tamanhoN)
  vetorU=np.copy(x[tamanhoN:2*tamanhoN])
  newX=np.copy(x[0:tamanhoN])
  python_matrix=vector_to_square_matrix(newX)

  # Initialize the output matrix
  python_matrix_2 = (-2*(c**2/k1**2)*python_matrix)+(-2*(c**2/k2**2)*python_matrix)
  python_matrix_2[:-1, :] += (c**2/k2**2)*python_matrix[1:, :]   # Down
  python_matrix_2[1:, :] += (c**2/k2**2)*python_matrix[:-1, :]   # Up
  python_matrix_2[:, :-1] += (c**2/k1**2)*python_matrix[:, 1:]   # Right
  python_matrix_2[:, 1:] += (c**2/k1**2)*python_matrix[:, :-1]   # Left


  vetorV=python_matrix_2.flatten()
  vetorY=np.concatenate((vetorU,vetorV))
  #print("A3 Completo")

  return vetorY

def plot_3d_matrix(x, y, z):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(x, y, z, cmap='viridis')

    # Add labels and a color bar
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.colorbar(surf)

    # Show the plot
    plt.show()






print("Rodando")
a=0
b=np.pi
d=0
e=np.pi
c=1/np.sqrt(2)
n=1024
discretizaçãoEixoX=np.linspace(a,b,n+2)
discretizaçãoEixoY=np.linspace(d,e,n+2)
discretizaçãoEixoX = discretizaçãoEixoX[1:-1]
discretizaçãoEixoY = discretizaçãoEixoY[1:-1]
#print(discretizaçãoEixoX[0])

result_matrix = calculate_matrix(discretizaçãoEixoX, discretizaçãoEixoY, funcaoF)
vetorUInicial=np.array(result_matrix).flatten()
result_matrix_2=calculate_matrix(discretizaçãoEixoX, discretizaçãoEixoY, funcaoG)
vetorVInicial=np.array(result_matrix_2).flatten()
vetorYInicial=np.concatenate((vetorUInicial,vetorVInicial))
#print(A2(vetorYInicial)-A(vetorYInicial))

YOLD=np.copy(vetorYInicial)
YNOW=np.copy(vetorYInicial)

z=0

process = psutil.Process(os.getpid())
 
total_start_time = time.perf_counter()
total_start_mem = process.memory_info().rss  # in bytes

for tempo in range (1,2):
  start_time = time.perf_counter()
  start_mem = process.memory_info().rss
  peak_mem = start_mem

  tempo=tempo*6
  t=tempo
  U=np.zeros((2,len(vetorYInicial)))
  U[0]=vetorYInicial
  d,stats=kiops(numpy.array([tempo,tempo]), A3, U, 1e-7, 10, 10, 128, 2,False)
  expkiops=d[0]

  #plt.plot(discretizaçãoEixoX[1:len(discretizaçãoEixoX)-1],expkiops[0:n],label="KIOPS")
  #plt.plot(discretizaçãoEixoX[1:len(discretizaçãoEixoX)-1],YNOW[0:n**2],label="KIOPS")


#print(stats)

total_end_time = time.perf_counter()
total_end_mem = process.memory_info().rss


print("=== TOTAL LOOP STATS ===")
print(f"Total Time: {(total_end_time - total_start_time):.4f} seconds")
print(f"Total Memory change: {(total_end_mem - total_start_mem)/1024:.2f} KB")


tamanhoKIOPS=int(len(expkiops)/2) 
expkiops2=vector_to_square_matrix(expkiops[0:(tamanhoKIOPS)])


# Create the meshgrid
X, Y = np.meshgrid(discretizaçãoEixoX, discretizaçãoEixoY)
print(t)

U = np.sin(X) * np.sin(Y) * np.sin(t)

diff=abs(expkiops2-U)

print("Eis o erro maximo ")
print(diff.max())

scale = 1.2  # 20% bigger

plt.rcParams.update({
    "font.size": 12 * scale,
    "axes.titlesize": 14 * scale,
    "axes.labelsize": 12 * scale,
    "xtick.labelsize": 10 * scale,
    "ytick.labelsize": 10 * scale,
    "legend.fontsize": 10 * scale,
})




# Plot contour
plt.figure(figsize=(12, 10))
contour = plt.contourf(X, Y, expkiops2,cmap="viridis")
plt.colorbar(contour)

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("KIOPS approximation  for t = " +str(t) )
plt.savefig(f"A_KIOPS_2D_{t}.png",dpi=300,bbox_inches="tight")
plt.show()

plt.figure(figsize=(12, 10))
contour = plt.contourf(X, Y, U,cmap="viridis")
plt.colorbar(contour)

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Wave Equation for t = " + str(t))
plt.savefig(f"A_SOLUTION_2D_{t}.png",dpi=300,bbox_inches="tight")
plt.show()


plt.figure(figsize=(12, 10))
contour = plt.contourf(X, Y, diff,cmap="viridis")
plt.colorbar(contour)

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("KIOPS error for t = " + str(t))
plt.savefig(f"A_ERROR_2D_{t}.png",dpi=300,bbox_inches="tight")
plt.show()

