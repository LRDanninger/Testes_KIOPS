from scipy.sparse.linalg import expm
import numpy as np
import time
import psutil
import os

# Este script carrega duas matrizes .npy,
# Usa-se expm para calcular a exponencial completa da matriz A, que é a matriz diagonalizada passada pelo processo de ortogonalização
# Então medimos o tempo de execução e a variação de memória do processo para comparativo usado em tese,
# Na matriz B salvamos o resultado teórico para podermos comparar o erro.

process = psutil.Process(os.getpid())
 
total_start_time = time.perf_counter()
total_start_mem = process.memory_info().rss  # in bytes



matrizA = np.load("matrizA.npy")


 
Aexp=expm(matrizA)


total_end_time = time.perf_counter()
total_end_mem = process.memory_info().rss

print("=== TOTAL LOOP STATS ===")
print(f"Total Time: {(total_end_time - total_start_time):.4f} seconds")
print(f"Total Memory change: {(total_end_mem - total_start_mem)/1024:.2f} KB")

matrizB= np.load("matrixB.npy")
diff=Aexp-matrizB
print(diff.max())