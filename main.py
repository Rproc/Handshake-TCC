# from source import scargc
from source.util import utils as u
from source import metrics, plots
from source import handshake, handshake2, scargc, hs
import sys
import time
import os
import psutil
import resource

def main():

    poolsize = 150
    clusters = 2
    n_components = 2
    episilon = 0.15

    # n_features = 8
    # band = 0.4
    base = '/home/localuser/Documentos/procopio/tcc/datasets/'
    # base = '/home/procopio/Documents/tcc/datasets/'
    # base = '/home/god/Documents/ccomp/tcc/datasets/'
    # list = ['1CDT.txt']#, '1CHT.txt', '1CSurr.txt', '2CDT.txt', '2CHT.txt', 'NOAA.txt', 'elec.txt', 'keystroke.txt']
    # list = ['keystroke.txt']
    list = ['NOAA.txt']#, 'elec.txt']
    database = {}

    for i in range(0, len(list)):
        database[i] = base + list[i]

    # array_ep = [0.05, 0.10, 0.15]
    array_ep = [0.1]
    # array_p = [10, 20, 30]
    array_p = [30]

    for key, value in database.items():
        # if (key != 2):
        #     continue
        for ep in range(0, len(array_ep)):
            for p in range(0, len(array_p)):
                adr = value
                dataset, data_labeled, dataset_train, l_train, stream, l_stream, n_features = u.criar_datasets(5, adr)

                start = time.time()
                # if key == 0:
                predicted, updt = handshake2.handshake2(dataset, data_labeled, dataset_train, l_train, stream, l_stream, n_components, n_features, array_ep[ep], array_p[p])
                # else:
                #     n_components = 4
                #     predicted, updt = hs.handshake2(dataset, data_labeled, dataset_train, l_train, stream, l_stream, n_components, n_features, array_ep[ep], array_p[p])

                end = time.time()
                mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # print('mem', mem)
                startScargc = time.time()

                predictedS, updtS = scargc.scargc_1NN(dataset, data_labeled, dataset_train, l_train, stream, l_stream, poolsize, clusters, n_features)

                endScargc = time.time()
                memS = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

                tempoS = endScargc - startScargc
                tempo = end - start
                name = list[int(key)]

                acc_percent, f1_percent, mcc_percent = metrics.makeBatches(l_stream, predicted, len(stream))
                score, f1, mcc, std = metrics.metrics(acc_percent, l_stream, predicted, f1_type = 'macro')

                acc_percentScargc, f1_percentS, mcc_percentS = metrics.makeBatches(l_stream, predictedS, len(stream))
                scoreS, f1S, mccS, stdS = metrics.metrics(acc_percentScargc, l_stream, predictedS, f1_type = 'macro')
                # print(f1_percentS[0])
                # u.saveLog2(name, array_ep[ep], array_p[p], updt, acc_percent, score, f1, mcc, tempo, mem)
                matrixAcc = [acc_percent[0], acc_percentScargc[0]]
                matrixF1 = [f1_percent[0], f1_percentS[0]]

                # print(matrixF1)
                listTime = [tempo, tempoS]
                listAcc = [score, scoreS]
                listMethod = ['Handshake', 'SCARGC']
                print('memory peak: ', mem)
                print('Acc: ', score)
                print('Macro-F1: ', f1)
                print('MCC: ', mcc)
                print('Desvio Padrão: ', std)
                print('Numero de atualizações: ', updt)
                # plots.plotF1(f1_percent, 100, '1CDT_Handshake')
                # plots.plotF1(f1_percentS, 100, '1CDT_SCARGC')

                plots.plotAcc(acc_percent, 100, 'keystroke')
                # plots.plotAccuracyCurves(matrixAcc, listMethod)

                plots.plotPerBatches(stream, predicted, l_stream, len(stream))

if __name__ == '__main__':
	main()
