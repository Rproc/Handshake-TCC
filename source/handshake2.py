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
    # print(type(cl))
    pred_proba = gmm.predict_proba(np.reshape(d_treino[0,:], (-1, 2)))
    aux = np.hstack([d_treino[0, :], cl, pred_proba[0][cl], l_train[0]] )
    inicial_pool = aux
    # print(aux)

    for i in range(1, len(d_treino)):

        pred = gmm.predict(np.reshape(d_treino[i,:], (-1, 2)))
        cl = int(pred)
        pred_proba = gmm.predict_proba(np.reshape(d_treino[i,:], (-1, 2)))
        aux = np.hstack([d_treino[i, :], cl, pred_proba[0][cl], l_train[i]] )

        inicial_pool = np.vstack([inicial_pool, aux])


    # FILTER

    # for i in range(0, len(inicial_pool)):

    print(inicial_pool[0])

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
    b = 0

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

        pool = np.vstack([pool, temp])

        # print(pool[:, 0:(n_features - 1)])

        count += 1
        # sys.exit(0)
        # if delta > episilon:

        gmm = GaussianMixture(n_components=num_components).fit(pool[:,:-1])
        pred_all = gmm.predict(pool[:, 0:(n_features - 1)])

        init_pool = np.asarray(inicial_pool[:, -3], dtype=int)
        new_labels = pred_all[0:len(inicial_pool)]

        concordant_labels = np.nonzero(inicial_pool[:,-3] == new_labels[:] )[0]

        # if len(concordant_labels)/poolsize < 1:
        KNN.fit(pool[:, 0:(n_features - 1)], pool[:, -1])

        pred_proba_all = gmm.predict_proba(pool[:, :(n_features - 1)])

        new_pool = []
        cl = int(pred_all[0])
        print(pred_proba_all[0][cl])
        aux = np.hstack( [pool[0, 0:(n_features -1 )], cl, pred_proba_all[0][cl], pool[0, -1]] )
        new_pool = aux
        for k in range(1, len(pool)):
            cl = int(pred_all[k])
            aux = np.hstack( [pool[k, 0:(n_features -1 )], cl, pred_proba_all[k][cl], pool[k, -1]] )
            new_pool = np.vstack([new_pool, aux])

        print(new_pool)
        sys.exit(0)
        b+=1

        if(b == 1):
            sys.exit(0)
