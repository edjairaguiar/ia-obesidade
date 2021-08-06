import numpy as np
import matplotlib.pyplot as plt

numEpocas = 2  #n de epocas
q = 18          #dados de treinamento (usando a tabela 3 do artigo)
eta = 0.01      #taxa de aprendizado
m = 3           #neuronios na camada de entrada
N = 1           #neuronios na camada escondida
L = 2           #neuronios na camada de saída

#Dados de treinamento 
peso1 = np.array([68, 102, 54, 99, 95, 91, 118, 118, 93, 95, 131, 112, 50, 97, 101, 117, 80, 74])
peso2 = np.array([541, 576, 528, 435, 454, 300, 362, 450, 378, 403, 349, 467, 463, 374, 352, 362, 408, 573])
peso3 = np.array([257, 322, 315, 357, 115, 140, 144, 233, 121, 117, 80, 257, 134, 159, 133, 172, 216, 173])

#classificação desejada
d = np.array([[72.9, 84.0], [73.6, 84.8], [73.6, 84.0], [73.7, 84.0], [73.1, 83.5], [73.5, 82.5], [73.6, 85.0], [73.4, 83.5], [73.7, 84.0], [73.3, 84.0], [73.4, 85.0], 
              [74.0, 83.5], [73.2, 83.5], [73.3, 85.0], [73.6, 83.0], [71.7, 82.0], [72.2, 82.0], [72.5, 83.0]])

#inicia aleatoriamente as matrizes de pesos
W1 = np.random.random((N, m+1))
W2 = np.random.random((L, N+1)) #tratamento matricial para evitar loops

#erros
E = np.zeros(q)
Etm = np.zeros(numEpocas) #erro total médio p acompanhar evolucao do treinamento

bias = 1

#entrada
X = np.vstack((peso1, peso2, peso3))

#TREINAMENTO

for i in range(numEpocas):
    for j in range(q):
        
        #bias no vetor de entrada
        Xb = np.hstack((bias, X[:, j])) #bias + peso 1 + peso 2 + peso 3
        
        #saida da camada escondida
        O1 = np.tanh(W1.dot(Xb))
        
        #incluindo o bias; saída da camada escondida eh a entrada da saida:
        O1b = np.insert(O1, 1, bias)
        
        #saída
        Y = np.tanh(W2.dot(O1b))
        
        e = d[j] - Y #valor d - saída da rede 
        
        #erro total, quadratico
        #E[(j)] = (e.transpose().dot(2))/2 
        E[j] = np.dot(np.transpose(e), [2,2])/2
        
        #propagacao do erro 
        #gradiente da camada de saida
        delta2 = np.diag(e).dot((1 - Y*Y))
        vdelta2 = (W2.transpose()).dot(delta2)
        delta1 = np.diag(1 - O1b*O1b).dot(vdelta2)
        
        #atualizacao dos pesos
        W1 = W1 + eta*(np.outer(delta1[1:], Xb))
        W2 = W2 + eta*(np.outer(delta2, O1b))
        
        Etm[i] = E.mean() #media dos erros
        
        
plt.plot(Etm) #plota grafico do erro medio
plt.show()

#TESTE DA REDE        
        
Error_Test = np.zeros(q)

for i in range(q):
    #bias no vetor de entrada, usando os mesmos dados de treinamento para teste
    Xb = np.hstack((bias, X[:, j])) #bias + peso 1 + peso 2 + peso 3 + pesocorporal + cabdominal
        
    #saida da camada escondida
    O1 = np.tanh(W1.dot(Xb))
        
    #incluindo o bias; saída da camada escondida eh a entrada da saida:
    O1b = np.insert(O1, 0, bias)
        
    #neural network output
    Y = np.tanh(W2.dot(O1b))
    
    Error_Test = d[i] - Y

#vetor teste de erro - d; se acertar tudo, o vetor deve aparecer zerado
print(Error_Test)
print(np.round(Error_Test) - d)
