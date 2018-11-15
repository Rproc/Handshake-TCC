# Criar função para obter datasets
# Porcentagem treino = porcentagem de elementos rotulados
import math
import numpy as np
import os.path
from sklearn.decomposition import PCA
from source import handshake2 as hs2

class utils:

    def getDataset(name):
        print('oi')


    def criar_datasets(porcentagem_treino, path):
        # d_treino = np.zeros((n_elem, n_features), dtype=np.float)
        # indices_teste = []
        # i_arranjo = 0


        texto = open(path,"r")
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


    def chunkify(lst, n):
        return [lst[i::n] for i in range(n)]    

    def cossine_similarity(u1, u2):

        try:
            similarity = float(np.sum(u1 * u2))/( math.sqrt(np.sum(u1**2)) * math.sqrt(np.sum(u2**2)) )
        except:
            similarity = 0

        return similarity

    def similarity(centroid_past, class_u, kmeans_lab, x):

        sim = 0
        trust_u = -1
        for k in range(0, centroid_past.shape[0]):
            if int(centroid_past[k, -1]) == kmeans_lab[class_u]:
                sim = utils.cossine_similarity(x, centroid_past[k,:-1])
            if sim > trust_u:
                trust_u = sim

        return trust_u


    def dist_centroid(centroid_past, class_u, kmeans_lab, x):

        dist_y = 99999999
        dist_not_y = 99999999
        dist = 0
        for k in range(0, centroid_past.shape[0]):
            if int(centroid_past[k, -1]) == kmeans_lab[class_u]:
                dist = utils.dist_euc(x, centroid_past[k,:-1])
                if dist < dist_y:
                    dist_y = dist
            else:
                dist = utils.dist_euc(x, centroid_past[k,:-1])
                if dist < dist_not_y:
                    dist_not_y = dist

        dist_euc = 1 - (dist_y/(dist_y + dist_not_y) )

        return dist_euc

    def dist_euc(u1,u2):
        return math.sqrt(np.sum(pow(u1-u2,2)))


    def saveLog(name_dataset, acc_percent, score, f1, mcc, updt):

        save_path = '/home/localuser/Documentos/procopio/tcc/experiments'
        name = name_dataset + '_SCARGC' + '.log'
        completeName = os.path.join(save_path, name)

        f = open(completeName, 'w')
        ab = name_dataset + '\nnumber of updates: ' + str(updt) + '\n'+ 'acc_percent: '+ str(acc_percent) + '\n'+ 'score: '+ str(score) + '\n'+ 'f1: '+ str(f1)+ '\n'+ 'mcc: '+ str(mcc) + '\n'
        f.write(ab)
        f.close()

    def saveLog2(name_dataset, ep, percent, updt, acc_percent, score, f1, mcc, time, mem):

        save_path = '/home/localuser/Documentos/procopio/tcc/experiments/'
        name = name_dataset + '.log'
        completeName = os.path.join(save_path, name)

        f = open(completeName, 'a')
        ab = name_dataset + '\n' + 'Executed in: ' + str(time) + ' seconds\n' + 'ep: ' + str(ep) + ' percent_pool: ' + str(percent) + ' number of updts: ' + str(updt) + '\nacc_percent: '+ str(acc_percent) + '\n'+ 'score: '+ str(score) + '\nf1: '+ str(f1) + '\n'+ 'mcc: ' + str(mcc) + '\nMemory peak: ' + str(mem) + '\n\n'
        f.write(ab)
        f.close()

    def saveLogKDE(name_dataset, ep, percent, updt, acc_percent, score, f1, mcc, time):

        save_path = '/home/localuser/Documentos/procopio/tcc/experiments/KDE'
        name = name_dataset + '.log'
        completeName = os.path.join(save_path, name)

        f = open(completeName, 'a')
        ab = name_dataset + '\n' + 'Executed in: ' + str(time) + ' seconds\n' + 'ep: ' + str(ep) + ' percent_pool: ' + str(percent) + ' number of updts: ' + str(updt) + '\nacc_percent: '+ str(acc_percent) + '\n'+ 'score: ' + str(score) + '\n' + 'f1: ' + str(f1)+ '\n'+ 'mcc: ' + str(mcc) + '\n\n'
        f.write(ab)
        f.close()


    def pca(X, numComponents):
        pca = PCA(n_components=numComponents)
        pca.fit(X)
        PCA(copy=True, iterated_power='auto', n_components=numComponents, random_state=None, svd_solver='auto', tol=0.0, whiten=False)

        return pca.transform(X)
