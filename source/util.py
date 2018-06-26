# Criar função para obter datasets
# Porcentagem treino = porcentagem de elementos rotulados
import math
import numpy as np



class utils:

    def criar_datasets(porcentagem_treino):
        # d_treino = np.zeros((n_elem, n_features), dtype=np.float)
        # indices_teste = []
        # i_arranjo = 0


        texto = open("/home/procopio/Documents/tcc/datasets/1CDT.txt","r")
        linhas = texto.readlines()
        linhas = list(map(str.strip,linhas))
        n_features = len(linhas[0].split(','))
        withLabel =	len(linhas)
        limite_treino = math.floor(withLabel * (porcentagem_treino/100))


        dataset = np.zeros((withLabel, n_features), dtype=np.float)
        d_treino = np.zeros((limite_treino, n_features), dtype=np.float)
        l_train = np.zeros(limite_treino, dtype=np.int)
        d_stream = np.zeros(( (withLabel - limite_treino), n_features), dtype=np.float)
        l_stream = np.zeros( withLabel - limite_treino, dtype=np.int)
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
                l_train[i] = dataset[i, (n_t - 1)]

            else:
                d_stream[j,:] = dataset[i,:]
                j += 1
            i = i + 1

        l_stream = d_stream[:, (n_t -1)]
        data_labeled = dataset[:, (n_t -1)]
        # print(d_treino[0])
        # print(l_train[0])
        return dataset, data_labeled, d_treino, l_train, d_stream, l_stream, n_features

    
