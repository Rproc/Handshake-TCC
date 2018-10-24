import numpy as np
import random
import math
import time
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
# from source import util
import sys


def handshake_kde(dataset, data_labeled, d_treino, l_train, stream, l_stream, n_features, band, episilon, percent_init):

    # classes = set(data_labeled)
    # num_class = len(classes)
    d_treino = np.delete(d_treino, np.s_[n_features-1], axis=1)

    percent_pool = int( len(d_treino)/100 * percent_init )

    kde = KernelDensity(bandwidth=band).fit(d_treino)

    KNN = KNeighborsClassifier(n_neighbors=1)
    KNN.fit(d_treino, l_train)


    pdf = np.exp(kde.score_samples(np.reshape(d_treino[0,:], (-1, 2))))
    # print(pdf)
    aux = np.hstack([d_treino[0, :], pdf, l_train[0]] )
    inicial_pool = aux


    for i in range(1, len(d_treino)):
        # print(pdf)

        pdf = np.exp(kde.score_samples(np.reshape(d_treino[i,:], (-1, 2))))
        aux = np.hstack([d_treino[i, :], pdf, l_train[i]] )
        inicial_pool = np.vstack([inicial_pool, aux])

    # FILTER
    # print(inicial_pool)

    inicial_pool = inicial_pool[inicial_pool[:,-2].argsort()[::-1]]
    # print(inicial_pool[:,-2])

    pool1 = [inicial_pool[i,:] for i in range(0, len(inicial_pool)) if inicial_pool[i, -1] == 1]
    pool2 = [inicial_pool[i,:] for i in range(0, len(inicial_pool)) if inicial_pool[i, -1] == 2]

    half_percent = int(percent_pool/2)
    inicial_pool = np.vstack([pool1[0:half_percent], pool2[0:half_percent] ])

    pool = []#inicial_pool[:, :-3]
    labels = inicial_pool[:, -1]

    for i in range(0, len(inicial_pool)):
        pool.append(np.hstack([inicial_pool[i, :-2], labels[i]]) )

    pool = np.asarray(pool)

    # print(inicial_pool)
    # sys.exit(0)
    data_labeled = []
    poolsize = len(inicial_pool)
    count = 0
    updt = 0

    for i in range(0, len(stream)):

        x = stream[i,:-1]
        y = l_stream[i]

        x = np.reshape(x, (-1, 2))
        predicted = KNN.predict(x)
        trust = kde.score_samples(x)

        # print('x', x)

        temp = np.column_stack((x, predicted))

        data_labeled.append(predicted)
        pool = np.vstack([pool, temp])

        count += 1

        if trust <= episilon:

            kde = KernelDensity(bandwidth=band).fit(pool[:,:(n_features - 1)])
            KNN.fit(pool[:, 0:(n_features - 1)], pool[:, -1])
            pdf = np.exp(kde.score_samples(np.reshape(pool[0,:(n_features - 1)], (-1, 2))))

            new_pool = []

            aux = np.hstack( [pool[0, 0:(n_features -1 )], pdf , pool[0, -1]] )
            new_pool = aux

            for k in range(1, len(pool)):

                pdf = np.exp(kde.score_samples(np.reshape(pool[k,:(n_features - 1)], (-1, 2))))
                aux = np.hstack( [pool[k, 0:(n_features -1 )], pdf, pool[k, -1]] )
                new_pool = np.vstack([new_pool, aux])

            new_pool = new_pool[new_pool[:,-2].argsort()[::-1]]

            pool1 = [new_pool[i,:] for i in range(0, len(new_pool)) if new_pool[i, -1] == 1]
            pool2 = [new_pool[i,:] for i in range(0, len(new_pool)) if new_pool[i, -1] == 2]
            tam = 0

            if len(pool1) <= half_percent:
                tam = len(pool1)
                # t = percent_pool -
                inicial_pool = pool1[0:tam]
                inicial_pool = np.vstack([inicial_pool, pool2[0:percent_pool-tam]])

            elif len(pool2) <= half_percent:
                tam = len(pool2)
                inicial_pool = pool2[0:tam]
                inicial_pool = np.vstack([inicial_pool, pool1[0:percent_pool-tam]])

            pool = []
            labels = inicial_pool[:, -1]

            for i in range(0, len(inicial_pool)):
                pool.append(np.hstack([inicial_pool[i, :-2], labels[i]]) )

            pool = np.asarray(pool)

            updt+= 1

    return data_labeled, updt
