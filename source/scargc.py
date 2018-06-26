import numpy as np
import random
import math
import time
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
# from source import util



def start_scargc(dataset, data_labeled, d_treino, l_train, stream, l_stream, pool_size, num_clusters, n_features):

    classes = set(data_labeled)
    num_class = len(classes)

    centroid_past = []
    centroid_temp = []

    print(d_treino.shape[0])

    # d_train = np.delete(d_treino, np.s_[n_features-1], axis=1)

    if num_clusters == num_class:
        for class_label in range(0, num_class): # labels

            a = list(np.where(d_treino[:, (n_features - 1)] == list(classes)[class_label])[0])
            aux = np.zeros((len(a), n_features), dtype=np.float)

            i = 0
            for var in range(0, len(a)):
                aux[var, :] = d_treino[a[i], :]
                i += 1

            g = np.reshape(aux, (-1, n_features) )

            aux = np.median(g, axis=0) #median will return all elements of the centroids
            centroid_past.append(aux)

    else:
        kmeans = KMeans(n_clusters=num_clusters).fit(d_treino[:,0:-1])
        centroids = kmeans.cluster_centers_
        centroid_past = centroids

    # print(len(centroid_past))

    ##################################### End of Init Data ###########################
    stream = np.delete(stream, np.s_[n_features-1], axis=1)
    d_treino = np.delete(d_treino, np.s_[n_features-1], axis=1)

    # print(stream)
    # print(l_stream)
    pool = []
    updt = 0

    data_x = []
    data_y = []
    data_lab = []
    data_labels = []
    knn_labels = []
    data_acc = []

    for i in range(0, len(stream)):
        x = stream[i,:]
        y = l_stream[i]

        KNN = KNeighborsClassifier(n_neighbors=1)
        KNN.fit(d_treino, l_train)

        x = np.reshape(x, (-1, 2))
        predicted = KNN.predict(x)

        # Save stream data
        data_x.append(x)
        data_y.append(y)
        data_lab.append(d_treino)
        data_labels.append(l_train)
        # Save predictions
        knn_labels.append(predicted)
        data_acc.append(predicted)

        # knn_labels.append(predicted)
        temp = np.column_stack((x, predicted))

        predicted = []
        # print('temp ', temp)

        if len(pool) < 1:
            pool = temp
        else:
            pool = np.vstack([pool, temp])


        if len(pool) == pool_size:

            centroid_past = np.asarray(centroid_past)
            # c_old = np.delete(centroid_past, np.s_[n_features - 1], axis=1)
            kmeans = KMeans(n_clusters=num_clusters, init=centroid_past[-num_clusters:, :-1]).fit(pool[:,0:-1])
            centroid_cur = kmeans.cluster_centers_

            KNN.fit(centroid_past[:,:-1], centroid_past[:,-1])
            clab = KNN.predict(centroid_cur)
            nearest = KNN.kneighbors(centroid_cur, return_distance=False)

            intermed = []
            centroid_label = []

            # print(centroid_past.shape)

            a = np.median(np.vstack([centroid_past[nearest[0],0:-1], centroid_cur[0,:]]), axis=0)
            intermed = np.hstack( (a, clab[0]))
            centroid_label = clab[0]
            # print('before loop', intermed)

            for p in range(1, centroid_cur.shape[0]):
                a = np.median(np.vstack([centroid_past[nearest[p], 0:-1], centroid_cur[p,:]]), axis=0)
                aux = np.hstack( (a, clab[p]))
                intermed = np.vstack([intermed, aux])
                centroid_label = np.vstack([centroid_label, clab[p]])

            centroid_cur = np.column_stack([centroid_cur, centroid_label])
            centroid_past = intermed

            KNN.fit( np.vstack([centroid_cur[:,:-1], centroid_past[:,:-1]]), np.hstack([centroid_cur[:,-1], centroid_past[:,-1]]) )
            pred_all = KNN.predict(pool[:,0:-1])

            for p in range(0, pool.shape[0]):
                if p == 0:
                    new_pool = np.hstack([pool[0,0:-1], pred_all[0]])
                else:
                    new_pool = np.vstack([new_pool, np.hstack([pool[p,0:-1], pred_all[p]])])


            concordant_labels = np.nonzero(pool[:,-1] == new_pool[:,-1])[0]

            if len(concordant_labels)/pool_size < 1 or len(data_y) < pool.shape[0]:
                pool[:,-1] = new_pool[:, -1]
                centroid_past = np.vstack([centroid_cur, intermed])
                d_treino = pool[:, 0:-1]
                l_train = pool[:, -1]

                updt = updt + 1

            # print(concordant_labels)
            # break
            # print(c[:,:-1])
            # print(c[:,-1])
            pool = []

    return data_acc, d_treino, l_train, data_lab, data_labels, data_x, data_y, knn_labels
    # print(pool)
