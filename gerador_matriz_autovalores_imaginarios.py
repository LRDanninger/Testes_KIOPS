import numpy as np

# Neste código geramos uma matriz A de dimensão n×n cujos autovalores são puramente imaginários.
# Primeiro construímos uma matriz tridiagonal T contendo valores que correspondem às partes imaginárias dos autovalores, conforme desenvolvido na tese,
# incluindo valores de diferentes ordens de magnitude. Em seguida, gera-se uma matrizortogonal Q via decomposição QR de uma matriz aleatória. 
#A matriz final é construída como A = Q T Q^T, preservando os autovalores de T. Por fim, a matriz A é salva no arquivo "matrizA.npy",
# para ser utilizada em testes com métodos como KIOPS, phipm ou expm.


n = 512


sub_diag = np.zeros(n-1)
num_values = len(sub_diag[::2])
random_values = np.random.uniform(-5, 5, size=num_values)  # Random values
sub_diag[::2] = random_values  # Assign to even indices

sub_diag[0] = -1*10**-3
sub_diag[2] = -1*10**-3
sub_diag[4] = -1*10**-2
sub_diag[6] = -1*10**2
sub_diag[8] = -1*10**-1
sub_diag[10] =-1*10**1



super_diag = sub_diag.copy()  # Ensure symmetry
main_diag = np.full(n, 4)

T = np.diag(main_diag) + np.diag(sub_diag, k=-1) + np.diag(super_diag, k=1)

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
