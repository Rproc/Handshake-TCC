# from source import scargc
from source.util import utils as u
from source import metrics, plots
from source import handshake, handshake2, scargc, hs
import sys
import time
import os
import psutil
import resource
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score


def doTrain(data, label, step, n_features, percent, epsilon, n_components):

    label_ = label

    size = len(data)
    mod = size % 10

    if mod != 0:
        data = data[0:(size - mod), :]
        label = label[0:(size - mod)]

    tam = int((size - mod)/10)
    # print(tam)
    # sys.exit(0)

    data = np.vsplit(data, 10)
    label = u.chunkify(label, 10)
    label_ = label_[0:(size-mod)]
    # print(len(data))
    # print(len(label))

    # percent = int(percent)
    # print(percent)
    # print(epsilon)
    acc_percent = []
    f1_percent = []
    mcc_percent = []
    predicted = []
    updt = []

    pred, updt = handshake2.handshake2(data, label_, data[0], label[0], data[1], label[1], n_components, n_features, epsilon, percent)
    pred = np.array(pred, dtype=int)
    # print(label[i])
    pred = pred.flatten()
    predicted = pred
    acc_percent, f1_percent, mcc_percent = metrics.makeBatches(label[0], predicted, len(label[0]), step)

    # print(acc_percent, f1_percent, mcc_percent)
    # sys.exit(0)

    for i in range(1, len(data)-1):
        # print(i)
        pred, upd = handshake2.handshake2(data, label_, data[i], label[i], data[i+1], label[i+1], n_components, n_features, epsilon, percent)
        pred = np.array(pred, dtype=int)
        # print(label[i])
        pred = pred.flatten()

        # print(pred)
        acc, f1, mcc = metrics.makeBatches(label[i], pred, len(label[i]), step)

        predicted = np.vstack([predicted, pred])
        # print(predicted)
        # sys.exit(0)

        updt += upd
        acc_percent = np.vstack([acc_percent, acc])
        f1_percent = np.vstack([f1_percent, f1])
        mcc_percent = np.vstack([mcc_percent, mcc])


    # print(acc_percent)

    predicted = predicted.flatten()
    lab = label_[tam:]


    score = accuracy_score(lab, predicted)
    f1 = f1_score(lab, predicted, average = 'macro')
    mcc = matthews_corrcoef(lab, predicted)
    # score, f1, mcc = metrics.metrics(acc_percent, lab, predicted, step, f1_type = 'macro')
    # print(a,score, f1, mcc)
    return score, f1, mcc

def main():

    poolsize = 300
    clusters = 2
    n_components = 2
    # episilon = 0.15
    step = 10

    # n_features = 8
    # band = 0.4
    # base = '/home/localuser/Documentos/procopio/tcc/datasets/'
    # base = '/home/procopio/Documents/tcc/datasets/'
    base = '/home/god/Documents/ccomp/tcc/datasets/'
    list = ['1CSurr.txt', '2CDT.txt', '2CHT.txt']# 'NOAA.txt', 'elec.txt', 'keystroke.txt']
    # list = ['keystroke.txt']
    # list = ['NOAA.txt']#, 'elec.txt']
    database = {}

    for i in range(0, len(list)):
        database[i] = base + list[i]

    array_ep = [0.05, 0.10, 0.15]
    # array_ep = [0.1]
    array_p = [10, 20, 30]
    # array_p = [30]

    result = [0.0, 0.0, 0.0]
    aux = []
    result = np.array(result, dtype=float)

    dic = {}
    for key, value in database.items():
        dic[key] = {}
        for ep in range(0, len(array_ep)):
            for p in range(0, len(array_p)):

                if key == 7:
                    n_components = 4
                    step = 2
                adr = value
                dataset, data_labeled, dataset_train, l_train, stream, l_stream, n_features = u.criar_datasets(5, adr)

                score, f1, mcc = doTrain(dataset_train, l_train, step, n_features, array_p[p], array_ep[ep], n_components)

                aux = np.hstack([score, f1, mcc])
                result = np.vstack([result, aux])

                a = str(array_ep[ep])
                b = str(array_p[p])
                name = a +' / '+ b
                dic[key][name] = aux

    # dic[name] = result



    for p_id, p_info in dic.items():
        print('dataset :', p_id)

        for key in p_info:
            print(key, ':', p_info[key])

                #
                # dataset, data_labeled, dataset_train, l_train, stream, l_stream, n_features = u.criar_datasets(5, adr)
                #
                # start = time.time()
                # # if key == 0:
                # predicted, updt = handshake2.handshake2(dataset, data_labeled, dataset_train, l_train, stream, l_stream, n_components, n_features, array_ep[ep], array_p[p])
                # # else:
                # #     n_components = 4
                # #     predicted, updt = hs.handshake2(dataset, data_labeled, dataset_train, l_train, stream, l_stream, n_components, n_features, array_ep[ep], array_p[p])
                #
                # end = time.time()
                # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # # print('mem', mem)
                # startScargc = time.time()
                #
                # predictedS, updtS = scargc.scargc_1NN(dataset, data_labeled, dataset_train, l_train, stream, l_stream, poolsize, clusters, n_features)
                #
                # endScargc = time.time()
                # memS = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                #
                # tempoS = endScargc - startScargc
                # tempo = end - start
                # name = list[int(key)]
                #
                # acc_percent, f1_percent, mcc_percent = metrics.makeBatches(l_stream, predicted, len(stream))
                # score, f1, mcc, std = metrics.metrics(acc_percent, l_stream, predicted, f1_type = 'macro')
                #
                # acc_percentScargc, f1_percentS, mcc_percentS = metrics.makeBatches(l_stream, predictedS, len(stream))
                # scoreS, f1S, mccS, stdS = metrics.metrics(acc_percentScargc, l_stream, predictedS, f1_type = 'macro')
                # # print(f1_percentS[0])
                # # u.saveLog2(name, array_ep[ep], array_p[p], updt, acc_percent, score, f1, mcc, tempo, mem)
                # matrixAcc = [acc_percent[0], acc_percentScargc[0]]
                # matrixF1 = [f1_percent[0], f1_percentS[0]]
                #
                # # print(matrixF1)
                # listTime = [tempo, tempoS]
                # listAcc = [score, scoreS]
                # listMethod = ['Handshake', 'SCARGC']
                # print('memory peak: ', mem)
                # print('Acc: ', score)
                # print('Macro-F1: ', f1)
                # print('MCC: ', mcc)
                # print('Desvio Padrão: ', std)
                # print('Numero de atualizações: ', updt)
                # # plots.plotF1(f1_percent, 100, '1CDT_Handshake')
                # # plots.plotF1(f1_percentS, 100, '1CDT_SCARGC')
                #
                # plots.plotAcc(acc_percent, 100, 'keystroke')
                # # plots.plotAccuracyCurves(matrixAcc, listMethod)
                #
                # plots.plotPerBatches(stream, predicted, l_stream, len(stream))


if __name__ == '__main__':
	main()
