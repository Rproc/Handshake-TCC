import numpy as np
import random
import math
import time
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from source.util import utils
import sys


def handshake(dataset, data_labeled, d_treino, l_train, stream, l_stream, pool_size, num_clusters, n_features, episilon):

    classes = set(data_labeled)
    num_class = len(classes)

    centroid_past = []
    centroid_temp = []

    # print(d_treino.shape[0])

    # d_train = np.delete(d_treino, np.s_[n_features-1], axis=1)

    # if num_clusters == num_class:
    #     for class_label in range(0, num_class): # labels
    #
    #         a = list(np.where(d_treino[:, (n_features - 1)] == list(classes)[class_label])[0])
    #         aux = np.zeros((len(a), n_features), dtype=np.float)
    #
    #         i = 0
    #         for var in range(0, len(a)):
    #             aux[var, :] = d_treino[a[i], :]
    #             i += 1
    #
    #         g = np.reshape(aux, (-1, n_features) )
    #
    #         aux = np.median(g, axis=0) #median will return all elements of the centroids
    #         centroid_past.append(aux)

    # else:
    kmeans = KMeans(n_clusters=num_clusters).fit(d_treino[:,0:-1])
    centroid_past = kmeans.cluster_centers_
    centroid_past = np.asarray(centroid_past)

    KNN = KNeighborsClassifier(n_neighbors=1)
    KNN.fit(d_treino[:,:-1], l_train)
    centroid_past_lab = []
    centroid_past_lab = KNN.predict(np.reshape(centroid_past[0,:], (-1, 2)))

    kmeans_lab = {}
    p = int(kmeans.predict(np.reshape(centroid_past[0,:], (-1, 2))))

    kmeans_lab[p] = int(centroid_past_lab)

    for core in range(1, centroid_past.shape[0]):
        pred = KNN.predict(np.reshape(centroid_past[core,:], (-1, 2)))
        p = int(kmeans.predict(np.reshape(centroid_past[core,:], (-1, 2))))
        kmeans_lab[p] = int(pred)
        centroid_past_lab = np.vstack([centroid_past_lab, pred])

    centroid_past = np.hstack([centroid_past, centroid_past_lab])

    ##################################### End of Init Data ###########################
    stream = np.delete(stream, np.s_[n_features-1], axis=1)
    d_treino = np.delete(d_treino, np.s_[n_features-1], axis=1)

    pool = []
    updt = 0
    b = 0
    data_x = []
    data_ys = []
    data_yu = []
    data_lab = []
    data_labels = []
    knn_labels = []

    for i in range(0, len(stream)):
        x = stream[i,:]
        y = l_stream[i]

        x_1d = x
        KNN = KNeighborsClassifier(n_neighbors=1)
        KNN.fit(d_treino, l_train)

        x = np.reshape(x, (-1, 2))

        predicted = KNN.predict(x)
        index_s = (int(predicted) - 1)
        trust = KNN.predict_proba(x)
        trust_s = int(trust[:, index_s])

        pred_u = kmeans.predict(x)
        class_u = int(pred_u)
        sim = 0
        trust_u = -1

        sim = utils.similarity(centroid_past, class_u, kmeans_lab, x_1d)
        trust_u = utils.dist_centroid(centroid_past, class_u, kmeans_lab, x_1d)

        # print(i)
        # print('pred_u', kmeans_lab[class_u], 'trust_u', trust_u, 'sim', sim)
        # print('pred_s', int(predicted), 'trust_s', trust_s)

        delta = abs(trust_s - trust_u)

        # print('delta', delta)
        # print('\n')
        # Save stream data
        data_x.append(x_1d)
        data_ys.append(y)
        data_lab.append(d_treino)
        data_labels.append(l_train)
        # Save predictions
        knn_labels.append(predicted)
        # data_acc.append(predicted)

        temp = np.column_stack((x, predicted))

        print('hs', temp)

        sys.exit(0)
        predicted = []

        if len(pool) > 0:
            pool = np.vstack([pool, temp])
        else:
            pool = temp


        if len(pool) == pool_size or delta > episilon:

            centroid_past = np.asarray(centroid_past)
            # c_old = np.delete(centroid_past, np.s_[n_features - 1], axis=1)
            kmeans = KMeans(n_clusters=num_clusters, init=centroid_past[-num_clusters:, :-1]).fit(pool[:,0:-1])
            centroid_cur = kmeans.cluster_centers_

            print('centroid past ', centroid_past)

            KNN.fit(centroid_past[:,:-1], centroid_past[:,-1])
            clab = KNN.predict(centroid_cur)
            nearest = KNN.kneighbors(centroid_cur, return_distance=False)

            intermed = []
            centroid_label = []


            a = np.median(np.vstack([centroid_past[nearest[0],0:-1], centroid_cur[0,:]]), axis=0)
            intermed = np.hstack( (a, clab[0]))
            centroid_label = clab[0]

            for p in range(1, centroid_cur.shape[0]):
                a = np.median(np.vstack([centroid_past[nearest[p], 0:-1], centroid_cur[p,:]]), axis=0)
                aux = np.hstack( (a, clab[p]))
                intermed = np.vstack([intermed, aux])
                centroid_label = np.vstack([centroid_label, clab[p]])

            centroid_cur = np.column_stack([centroid_cur, centroid_label])
            centroid_past = intermed

            print('centroid current ', centroid_cur , '\n')

            KNN.fit( np.vstack([centroid_cur[:,:-1], centroid_past[:,:-1]]), np.hstack([centroid_cur[:,-1], centroid_past[:,-1]]) )
            pred_all = KNN.predict(pool[:,0:-1])

            for p in range(0, pool.shape[0]):
                if p == 0:
                    new_pool = np.hstack([pool[0,0:-1], pred_all[0]])
                else:
                    new_pool = np.vstack([new_pool, np.hstack([pool[p,0:-1], pred_all[p]])])


            concordant_labels = np.nonzero(pool[:,-1] == new_pool[:,-1])[0]

            print('pool', pool[:,-1])
            print('pool', new_pool[:,-1], '\n\n')
            # print('pool', l_stream[:21])

            # print('\npoolsize)

            if len(concordant_labels)/pool.shape[0] < 1 or len(data_ys) < pool.shape[0]:
                pool[:,-1] = new_pool[:, -1]
                centroid_past = np.vstack([centroid_cur, intermed])
                d_treino = pool[:, 0:-1]
                l_train = pool[:, -1]

                updt = updt + 1

            # break
            # print(c[:,:-1])
            # print(c[:,-1])
            # print(l_train)
            print(concordant_labels)
            print('updt', updt)

            # sys.exit(0)
            b += 1
            pool = []



    return d_treino, l_train, data_lab, data_labels, data_x, data_y, knn_labels, updt
    # print(pool)
