import pymp
import pandas as pd
import sklearn.datasets
import sklearn.cluster
import scipy.cluster.vq
import matplotlib.pyplot as plot
import time


elem = 100000 #numero de elementos presente para a criação do dataframe
num_grupos = 4 #numero de grupos/cluster a serem distribuidos pelos elementos

ini = time.time() #calcula o tempo de execução do trecho

# Gera o dataframe passando o numero de elementos/amostras(samples) e cria centroides para as amostras de acordo com o numero de grupos
Data, labels = sklearn.datasets.make_blobs(n_samples = elem, n_features = 2, centers = num_grupos)

fim = time.time()
print(fim-ini," tempo 1")

ini = time.time()

#atribuo o dataframe criado anteriormente como um dataframe nativo da biblioteca pandas para poder "particionar" depois
data = pd.DataFrame(data=Data)
with pymp.Parallel(4) as p: #paralelizando com 4 threads
  for i in p.range(0,4): #chamada de cada thread
    with p.lock: #evita que acessem o mesmo trecho do dataframe
      Data_paralelo = data.iloc[i * 25000:(i+ 1) * 25000] #calculo para efetuar a divisao do dataframe de forma igual para cada thread
      #algoritmo da biblioteca scipy que faz o calculo dos vizinhos proximos de acordo com a quantia de grupos e os dados de entrada
      means, _ = scipy.cluster.vq.kmeans(Data_paralelo, num_grupos, iter=300)

fim = time.time()
print(fim-ini," tempo 2")

ini = time.time()

#chamando a biblioteca da sklearn que faz o calculo de Kmeans após termos definido os vizinhos e os gruposs a serem utilizados
kmeans = sklearn.cluster.KMeans(num_grupos, max_iter=300) #Low-level parallelism
kmeans.fit(Data)
means = kmeans.cluster_centers_ #gera o calculo dos centróides da região de um grupo determinado

fim = time.time()
print(fim-ini," tempo 3")

ini = time.time()

#chamada da biblioteca matplotlib para fazer a plotagem(exibição) do gráfico com os dados calculoados através do dataframe
plot.scatter(Data[:, 0], Data[:, 1], c = labels) #plota os dados gerados de forma que atribui as labels de acordo com a definição gerada anteriormente 
plot.scatter(means[:, 0], means[:, 1], linewidths=2)

fim = time.time()
print(fim-ini," tempo 4")
plot.show() #Exibi a pilha de plot calculada

