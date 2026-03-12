import numpy as np

# Neste código geramos  uma matriz A de dimensão n×n com autovalores positivos e negativos.
# Primeiro criamos um matriz diagonal com autovalores aleatórios, e introduzimos em sua diagonal autovalores de diferentes ordens de magnitude
# Em seguida, gera-se uma matriz ortogonal Q via decomposição QR de uma matriz aleatória.
#A matriz final é construída como A = Q T Q^T, preservando os autovalores de T. Por fim, a matriz A é salva no arquivo "matrizA.npy", a ser usada por KIOPS, phipm ou expm.


n = 256
# Create an array of 1000 random numbers
diagonal_values =-1* np.abs(np.random.rand(n))

# Set the first value to 0.1
diagonal_values[0] = -1*10**-2
diagonal_values[1] = -1*10**2
diagonal_values[2] = -1*10**-3
diagonal_values[3] = -1*10**3
diagonal_values[4] = -1*10**-4
diagonal_values[5] = -1*10**4
diagonal_values[6] = -1*10**-6
diagonal_values[7] = -1*10**6

# Create the diagonal matrix
T = np.diag(diagonal_values)


# Generate an orthogonal matrix Q using QR decomposition
A = np.random.randn(n, n)  # Generate a random matrix
Q, _ = np.linalg.qr(A)  # QR decomposition to get an orthogonal matrix

# Compute Q * T * Q^T
result = Q @ T @ Q.T

print("Computed Q * T * Q^(-1) successfully!")

# Optional: Check shape and a few values
print(result.shape)
print(result[:5, :5])  # Show top-left corner
np.save("matrizA.npy",result)
