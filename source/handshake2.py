import numpy as np
import random
import math
import time
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from source import util
import sys


def handshake2(dataset, data_labeled, d_treino, l_train, stream, l_stream, num_components, n_features, episilon):

    classes = set(data_labeled)
    num_class = len(classes)
    d_treino = np.delete(d_treino, np.s_[n_features-1], axis=1)


    gmm = GaussianMixture(n_components=num_components).fit(d_treino)

    KNN = KNeighborsClassifier(n_neighbors=1)
    KNN.fit(d_treino, l_train)

    pred = gmm.predict(np.reshape(d_treino[0,:], (-1, 2)))
    # print(pred_s)
    cl = int(pred)
    pred_proba = gmm.predict_proba(np.reshape(d_treino[0,:], (-1, 2)))
    aux = np.hstack([d_treino[0, :], pred, pred_proba[0][cl], l_train[0]] )

    inicial_pool = aux
    # print(aux)

    for i in range(1, len(d_treino)):

        pred = gmm.predict(np.reshape(d_treino[i,:], (-1, 2)))
        cl = int(pred)
        pred_proba = gmm.predict_proba(np.reshape(d_treino[i,:], (-1, 2)))
        aux = np.hstack([d_treino[i, :], pred, pred_proba[0][cl], l_train[i]] )

        inicial_pool = np.vstack([inicial_pool, aux])


    # FILTER

    # for i in range(0, len(inicial_pool)):

    inicial_pool = inicial_pool[inicial_pool[:,3].argsort()[::-1]]
    inicial_pool = inicial_pool[0:80]
    pool = []#inicial_pool[:, :-3]
    labels = inicial_pool[:, -1]

    for i in range(0, len(inicial_pool)):
        pool.append(np.hstack([inicial_pool[i, :-3], labels[i]]) )

    pool = np.asarray(pool)

    # print(pool)

    data_labeled = []
    poolsize = len(inicial_pool)
    count = 0
    print(poolsize)

    # sys.exit(0)
    for i in range(0, len(stream)):

        x = stream[i,:-1]
        y = l_stream[i]

        x = np.reshape(x, (-1, 2))

        predicted = KNN.predict(x)
        index_s = (int(predicted) - 1)
        trust = KNN.predict_proba(x)
        trust_s = int(trust[:, index_s])

        pred = gmm.predict(x)
        cl = int(pred)
        pred_proba = gmm.predict_proba(np.reshape(x, (-1, 2)))
        trust_u = pred_proba[0][cl]

        delta = abs(trust_u - trust_s)

        temp = np.column_stack((x, predicted))

        # print(temp)

        if len(pool) > 0:
            pool = np.vstack([pool, temp])
        else:
            pool = temp

        count += 1
        # sys.exit(0)
        if delta > episilon:

            gmm = GaussianMixture(n_components=num_components).fit(pool[:,:-1])
            pred_all = gmm.predict(pool[:, 0:(n_features -1 )])

            new_labels = pred_all[0:len(inicial_pool)]

            concordant_labels = np.nonzero(inicial_pool[:,-3] == new_labels[:])[0]

            print(concordant_labels)
            sys.exit(0)
