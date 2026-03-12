import numpy as np
import time
import psutil
import os


# ------------------------------------------------------------
# Função que implementa o método de Runge-Kutta de quarta ordem
# para resolver um sistema de EDOs quando a variável de estado
# é representada por um vetor. Usamos na tese para comparar o resultado com KIOPS
# ------------------------------------------------------------

def rk44_array(f, x0, h, n):
    """
    Executa o método RK4 para um sistema de EDOs.

    Parâmetros:
    f  -> função que define o sistema (lado direito da EDO)
    x0 -> condição inicial (vetor)
    h  -> passo de tempo
    n  -> número de iterações

    A função salva apenas os dois últimos estados para reduzir
    o uso de memória.
    """

    x_prev = x0.copy()
    x_curr = x0.copy()

    process = psutil.Process(os.getpid())

    total_start_time = time.perf_counter()
    total_start_mem = process.memory_info().rss

    for i in range(n):

        start_time = time.perf_counter()
        start_mem = process.memory_info().rss

        # Etapas do método Runge-Kutta de ordem 4
        k1 = f(x_curr)
        k2 = f(x_curr + 0.5 * h * k1)
        k3 = f(x_curr + 0.5 * h * k2)
        k4 = f(x_curr + h * k3)

        # Atualização da solução
        x_next = x_curr + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

        # Atualiza os estados
        x_prev = x_curr
        x_curr = x_next

        end_time = time.perf_counter()
        end_mem = process.memory_info().rss

        if i % 100 == 0:
            print(f"Iteration {i}")
            print(f"   Time: {(end_time - start_time):.4f} s")
            print(f"   Memory change: {(end_mem - start_mem)/1024:.2f} KB\n")

    total_end_time = time.perf_counter()
    total_end_mem = process.memory_info().rss

    print("=== TOTAL LOOP STATS ===")
    print(f"Total Time: {(total_end_time - total_start_time):.4f} seconds")
    print(f"Total Memory change: {(total_end_mem - total_start_mem)/1024:.2f} KB")

    return x_prev, x_curr


# ------------------------------------------------------------
# Função que define o sistema de EDOs obtido da discretização
# da equação de onda bidimensional
# ------------------------------------------------------------
def Function_A(x):

    tamanhoN = int(len(x)/2)

    N = int(np.sqrt(tamanhoN))
    raizN = int(np.sqrt(tamanhoN))

    k1 = (b-a)/(raizN+1)
    k2 = (e-d)/(raizN+1)

    c = 1/np.sqrt(2)

    vetorY = np.zeros(2*tamanhoN)

    vetorU = np.copy(x[tamanhoN:2*tamanhoN])

    newX = np.copy(x[0:tamanhoN])

    python_matrix = vector_to_square_matrix(newX)

    python_matrix_2 = (-2*(c**2/k1**2)*python_matrix)+(-2*(c**2/k2**2)*python_matrix)

    python_matrix_2[:-1, :] += (c**2/k2**2)*python_matrix[1:, :]
    python_matrix_2[1:, :] += (c**2/k2**2)*python_matrix[:-1, :]
    python_matrix_2[:, :-1] += (c**2/k1**2)*python_matrix[:, 1:]
    python_matrix_2[:, 1:] += (c**2/k1**2)*python_matrix[:, :-1]

    vetorV = python_matrix_2.flatten()

    vetorY = np.concatenate((vetorU,vetorV))

    return vetorY


# ------------------------------------------------------------
# Funções para gerar condições iniciais
# ------------------------------------------------------------
def funcaoF(x,y):
  return np.sin(x)*np.sin(y)


def funcaoG(x,y):
  return np.sin(x)*np.sin(y)


# ------------------------------------------------------------
# Gera matriz aplicando uma função f(x,y) nos pontos da malha
# ------------------------------------------------------------
def calculate_matrix(x, y, function):
    matrix = []
    for i in range(len(x)):
        row = []
        for j in range(len(y)):
            result = function(x[i], y[j])
            row.append(result)
        matrix.append(row)
    return matrix


# ------------------------------------------------------------
# Converte vetor em matriz quadrada
# ------------------------------------------------------------
def vector_to_square_matrix(vector):

    matrix_size = int(np.sqrt(len(vector)))

    square_matrix = np.reshape(vector, (matrix_size, matrix_size))

    return square_matrix


# ------------------------------------------------------------
# Programa principal
# ------------------------------------------------------------

print("Rodando")

a = 0
b = np.pi
d = 0
e = np.pi

n = 512

discretizaçãoEixoX = np.linspace(a,b,n+2)
discretizaçãoEixoY = np.linspace(d,e,n+2)

discretizaçãoEixoX = discretizaçãoEixoX[1:-1]
discretizaçãoEixoY = discretizaçãoEixoY[1:-1]


# Construção das condições iniciais
result_matrix = calculate_matrix(discretizaçãoEixoX, discretizaçãoEixoY, funcaoF)
vetorUInicial = np.array(result_matrix).flatten()

result_matrix_2 = calculate_matrix(discretizaçãoEixoX, discretizaçãoEixoY, funcaoG)
vetorVInicial = np.array(result_matrix_2).flatten()

vetorYInicial = np.concatenate((vetorUInicial,vetorVInicial))

print(vetorYInicial.shape)


# Parâmetros de integração temporal
h = 0.0001
n = 10000


# Integração usando RK4
x, z = rk44_array(Function_A, vetorYInicial, h, n)


# Reconstrução da solução final
x_final = x

tamanhoKIOPS = int(len(x_final)/2)

expkiops2 = vector_to_square_matrix(x_final[0:(tamanhoKIOPS)])


# Construção da solução analítica
X, Y = np.meshgrid(discretizaçãoEixoX, discretizaçãoEixoY)

t = h*n

U = np.sin(X) * np.sin(Y) * (np.cos(t) + np.sin(t))


# Cálculo do erro máximo
diff = abs(expkiops2 - U)

print("Eis o erro maximo ")
print(diff.max())