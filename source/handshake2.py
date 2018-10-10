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

    pred = gmm.predict(np.reshape(d_treino[0,:], (-1, 2)))
    cl = int(pred)
    pred_proba = gmm.predict_proba(np.reshape(d_treino[0,:], (-1, 2)))
    aux = np.hstack([d_treino[0, :], pred, pred_proba[0][cl]] )

    inicial_pool = aux
    print(aux)


    for i in range(1, len(d_treino)):

        pred = gmm.predict(d_treino[i, :])
        cl = int(pred)
        pred_proba = gmm.predict_proba(np.reshape(d_treino[i,:], (-1, 2)))
        aux = np.hstack([d_treino[i, :], pred, pred_proba[0][cl]] )

        inicial_pool = np.vstack([inicial_pool, aux])


    # FILTER


    for i in range(0, len(stream)):

        x = stream[i,:]
        y = l_stream[i]

        x_1d = x
        KNN = KNeighborsClassifier(n_neighbors=1)
        KNN.fit(d_treino, l_train)

        x = np.reshape(x, (-1, 2))
