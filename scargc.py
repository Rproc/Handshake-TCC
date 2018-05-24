import numpy as np
import random
import math
import time
from sklearn.cluster import KMeans


# Criar função para obter datasets
# Porcentagem treino = porcentagem de elementos rotulados

def criar_datasets(porcentagem_treino):
    # d_treino = np.zeros((n_elem, n_features), dtype=np.float)
    # indices_teste = []
    # i_arranjo = 0


    texto = open("datasets/1CDT.txt","r")
    linhas = texto.readlines()
    linhas = list(map(str.strip,linhas))
    n_features = len(linhas[0].split(','))
    withLabel =	len(linhas)
    limite_treino = math.floor(withLabel * (porcentagem_treino/100))


    dataset = np.zeros((withLabel, n_features), dtype=np.float)
    d_treino = np.zeros((limite_treino, n_features), dtype=np.float)
    d_stream = np.zeros(( (withLabel - limite_treino), n_features), dtype=np.float)
    # cria vetor de permutacao aleatoria de 0 a n_ratings-1
    # arranjo = np.random.permutation(withLabel)
    i = 0
    j = 0

    for linha in linhas:
        info = linha.split(",")
        n_t = len(info)

        # print(info)
        for k in range(0, n_t - 1):
            dataset[i, k] = float(info[k])

        dataset[i, (n_t - 1)] = int(info[n_t - 1])

        if(i < limite_treino):
            d_treino[i,:] = dataset[i,:]

        else:
            d_stream[j,:] = dataset[i,:]
            j += 1
        i = i + 1

    labeled = dataset[:, (n_t -1)]
    # print(dataset)
    return dataset, d_treino, d_stream, labeled, n_features


def start_scargc(dataset, d_treino, dataset_label, pool_size, num_clusters, n_features):

    classes = set(dataset_label)
    num_class = len(classes)

    centroid_past = []
    centroid_temp = []

    if num_clusters == num_class:
        for class_label in range(0, num_class): # labels

            # print(list(classes)[class_label])
            a = list(np.where(d_treino[:, (n_features - 1)] == list(classes)[class_label])[0])
            # size = len(a)
            # print(len(a))
            i = 0
            aux = np.zeros((len(a), n_features), dtype=np.float)

            for var in range(0, len(a)):
                aux[var, :] = d_treino[a[i], :]
                i += 1

            g = np.reshape(aux, (-1, n_features) )

            aux = np.median(g, axis=0)
            # print(aux)
            centroid_past.append(aux)
            #median will return all elements of the centroids


    else:
        kmeans = KMeans(n_clusters=num_clusters).fit(d_treino)
        print(kmeans.cluster_centers_)
    # b = np.reshape(centroid_past, (-1, num_class))
    # print(centroid_past)

def main():
    dataset, dataset_train, stream, dataset_label, n_features = criar_datasets(5)

    start_scargc(dataset, dataset_train, dataset_label, 50, 2, n_features)

if __name__ == '__main__':
	main()
