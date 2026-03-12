# Testes_KIOPS
Códigos da Tese

README

Este repositório contém implementações em Python utilizadas nos experimentos computacionais desenvolvidos na dissertação de mestrado Analysis and Applications of the KIOPS Method in Exponential Integration and Differential Equations: Performance Evaluation in Matrix Exponential Computation and Time-Dependent PDEs, apresentada ao Instituto de Matemática e Estatística da Universidade de São Paulo (IME-USP) por Luciano Rodrigues Danninger em 2025. A dissertação investiga métodos numéricos para o cálculo eficiente da ação da exponencial de matriz e sua aplicação na solução de sistemas diferenciais provenientes da discretização de equações diferenciais ordinárias e parciais. Em particular, o trabalho analisa métodos baseados em subespaços de Krylov, com foco no método KIOPS, avaliando seu desempenho computacional em comparação com outras abordagens clássicas e modernas para integração temporal e cálculo de exponenciais de matriz.

Os códigos presentes neste repositório correspondem às implementações utilizadas nos experimentos descritos na dissertação. A descrição matemática detalhada dos algoritmos, da construção dos operadores utilizados nos testes e da metodologia experimental encontra-se no texto da tese. Este repositório tem como objetivo disponibilizar os códigos utilizados para reproduzir os experimentos apresentados no trabalho.

O arquivo Runge_Kutta_44 corresponde à implementação do método clássico de Runge–Kutta de quarta ordem (RK44) utilizado na tese para resolver o sistema de equações diferenciais obtido a partir da discretização espacial da equação da onda.

Os dois códigos geradores de matrizes produzem as matrizes descritas na dissertação, com diferentes características espectrais. Esses scripts constroem as matrizes utilizadas nos experimentos e salvam o resultado em arquivo contendo a matrizA, que posteriormente é utilizada como entrada para os testes com os métodos KIOPS, phipm e expm.

O arquivo phipm corresponde à implementação do Algoritmo 919 descrito na dissertação, utilizado para calcular a ação da exponencial de matriz por meio de projeção em subespaços de Krylov.

O arquivo teste_expm apresenta tanto o cálculo da exponencial de matriz utilizando a função expm da biblioteca SciPy quanto o procedimento utilizado nos experimentos da tese para medir o tempo de execução e o consumo de memória durante o cálculo.

Para utilizar o método KIOPS nos experimentos é necessário obter a implementação em Python disponibilizada pelos autores do método. O código pode ser encontrado no seguinte endereço:

https://gitlab.com/stephane.gaudreault/jcp2021_highorder_sw/-/blob/master/kiops.py?ref_type=heads

O arquivo kiops.py deve ser colocado no mesmo diretório dos demais scripts do repositório. A utilização do método deve seguir as instruções e restrições definidas no README do repositório original do KIOPS.

Os códigos deste repositório utilizam principalmente as bibliotecas NumPy, SciPy e psutil, responsáveis pelas operações matriciais, pelo cálculo da exponencial de matriz e pela medição do uso de memória durante a execução dos experimentos.

Caso estes códigos sejam utilizados em trabalhos acadêmicos ou pesquisas, recomenda-se citar a dissertação de mestrado de Luciano Rodrigues Danninger apresentada ao Instituto de Matemática e Estatística da Universidade de São Paulo em 2025, onde os fundamentos teóricos, a metodologia experimental e a análise dos resultados são discutidos em detalhes.
