import numpy as np
import random
import math
import time
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
# from source import util
import sys


def handshake2(dataset, data_labeled, d_treino, l_train, stream, l_stream, num_components, n_features, episilon, percent_init, k):

    classes = set(data_labeled)
    class_list = list(classes)
    class_list = np.asarray(class_list[:], dtype=int)

    # print(class_list[0], 'hey', class_list[1])
    # sys.exit(0)
    # num_class = len(classes)
    d_treino = np.delete(d_treino, np.s_[n_features-1], axis=1)

    percent_pool = int( len(d_treino)/100 * percent_init )

    gmm = GaussianMixture(n_components=num_components).fit(d_treino)

    KNN = KNeighborsClassifier(n_neighbors=k)
    KNN.fit(d_treino, l_train)

    # print(n_features)

    pred = gmm.predict(np.reshape(d_treino[0,:], (-1, (n_features - 1))))
    # print(pred_s)
    cl = int(pred)
    # print(type(cl))
    pred_proba = gmm.predict_proba(np.reshape(d_treino[0,:], (-1, (n_features - 1))))
    aux = np.hstack([d_treino[0, :], cl, pred_proba[0][cl], l_train[0]] )
    inicial_pool = aux
    # print(aux)

    for i in range(1, len(d_treino)):

        pred = gmm.predict(np.reshape(d_treino[i,:], (-1, (n_features - 1))))
        cl = int(pred)
        pred_proba = gmm.predict_proba(np.reshape(d_treino[i,:], (-1, (n_features - 1))))
        aux = np.hstack([d_treino[i, :], cl, pred_proba[0][cl], l_train[i]] )

        inicial_pool = np.vstack([inicial_pool, aux])


    # FILTER

    inicial_pool = inicial_pool[inicial_pool[:,-2].argsort()[::-1]]

    pool_mix = {}

    for j in range(0, len(class_list)):
        pool_mix[j] = [inicial_pool[i,:] for i in range(0, len(inicial_pool)) if inicial_pool[i, -1] == class_list[j]]

    some_percent = int(percent_pool/len(class_list))

    for key, value in pool_mix.items():
        if key == 0:
            inicial_pool = value[0:some_percent]
        else:
            inicial_pool = np.vstack([inicial_pool, value[0:some_percent]])

    # print(inicial_pool)
    # sys.exit(0)
    pool = []#inicial_pool[:, :-3]
    labels = inicial_pool[:, -1]

    for i in range(0, len(inicial_pool)):
        pool.append(np.hstack([inicial_pool[i, :-3], labels[i]]) )

    pool = np.asarray(pool)


    data_labeled = []
    poolsize = len(inicial_pool)
    data_gmm = []
    count = 0
    updt = 0
    # print(poolsize)
    for i in range(0, len(stream)):

        x = stream[i,:-1]
        y = l_stream[i]

        x = np.reshape(x, (-1, (n_features - 1)))

        predicted = KNN.predict(x)
        index_s = (int(predicted) - 1)
        trust = KNN.predict_proba(x)
        trust_s = int(trust[:, index_s])

        pred = gmm.predict(x)
        cl = int(pred)
        pred_proba = gmm.predict_proba(np.reshape(x, (-1, (n_features - 1))))
        trust_u = pred_proba[0][cl]

        delta = abs(trust_u - trust_s)

        temp = np.column_stack((x, predicted))

        data_labeled.append(predicted)
        data_gmm.append(cl)
        pool = np.vstack([pool, temp])

        count += 1

        if delta >= episilon:
            gmm = GaussianMixture(n_components=num_components).fit(pool[:,:-1])
            pred_all = gmm.predict(pool[:, 0:(n_features - 1)])

            init_pool = np.asarray(inicial_pool[:, -3], dtype=int)
            new_labels = pred_all[0:len(inicial_pool)]

            concordant_labels = np.nonzero(inicial_pool[:,-3] == new_labels[:] )[0]

            if len(concordant_labels)/poolsize < 1:

                KNN.fit(pool[:, 0:(n_features - 1)], pool[:, -1])
                # pred_proba = gmm.predict_proba(pool[:, :(n_features - 1)])
                pred_proba = gmm.predict_proba(np.reshape(pool[0,:(n_features - 1)], (-1, (n_features - 1))))

                new_pool = []

                cl = int(pred_all[0])
                aux = np.hstack( [pool[0, 0:(n_features -1 )], cl, pred_proba[0][cl], pool[0, -1]] )
                new_pool = aux

                for k in range(1, len(pool)):
                    cl = int(pred_all[k])
                    pred_proba = gmm.predict_proba(np.reshape(pool[k,:(n_features - 1)], (-1, (n_features - 1))))
                    aux = np.hstack( [pool[k, 0:(n_features -1 )], cl, pred_proba[0][cl], pool[k, -1]] )
                    new_pool = np.vstack([new_pool, aux])

                new_pool = new_pool[new_pool[:,-2].argsort()[::-1]]
                pool_mix = {}

                for j in range(0, len(class_list)):
                    pool_mix[j] = [new_pool[i,:] for i in range(0, len(new_pool)) if new_pool[i, -1] == class_list[j]]

                some_percent = int(percent_pool/len(class_list))
                inicial_pool = []

                more_elements = np.zeros(len(class_list), dtype=int)

                for key, value in pool_mix.items():
                    if len(value) < some_percent:
                        # more_elements[int(key)] = 0
                        if key == 0:
                            inicial_pool = value[0:len(value)]
                        else:
                            inicial_pool = np.vstack([inicial_pool, value[0:len(value)]])
                    else:
                        if len(value) > some_percent:
                            # print('key', key)
                            more_elements[int(key)] = 1
                        if key == 0:
                            inicial_pool = value[0:some_percent]
                        else:
                            inicial_pool = np.vstack([inicial_pool, value[0:some_percent]])


                tam = percent_pool - len(inicial_pool)
                # print(tam, some_percent)
                # sys.exit(0)
                if tam > 0:
                    for elem in range(0, len(more_elements)):
                        if more_elements[elem] == 1:
                            value = pool_mix[elem]
                            # print(value)
                            # print(value[6])
                            inicial_pool = np.vstack([inicial_pool, value[(some_percent):(some_percent + tam)]])

                pool = []
                labels = inicial_pool[:, -1]

                for i in range(0, len(inicial_pool)):
                    pool.append(np.hstack([inicial_pool[i, :-3], labels[i]]) )

                pool = np.asarray(pool)

                # print(pool[:, -1])

                updt+= 1

    return data_labeled, updt, data_gmm
