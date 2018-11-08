# from source import scargc
from source.util import utils as u
from source import metrics
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

    n_features = 8
    # band = 0.4
    base = '/home/localuser/Documentos/procopio/tcc/datasets/'
    # base = '/home/procopio/Documents/tcc/datasets/'
    # base = '/home/god/Documents/ccomp/tcc/datasets/'
    list = ['1CDT.txt', '1CHT.txt', '1CSurr.txt', '2CDT.txt', '2CHT.txt', 'NOAA.txt', 'elec.txt', 'keystroke.txt']
    # list = ['keystroke.txt']
    # list = ['NOAA.txt', 'elec.txt']
    database = {}

    for i in range(0, len(list)):
        database[i] = base + list[i]

    array_ep = [0.05, 0.10, 0.15]
    # array_ep = [0.02]
    # array_ep = [0.15, 0.20, 0.25]
    array_p = [10, 20, 30]

    for key, value in database.items():
        # if (key != 2):
        #     continue
        for ep in range(0, len(array_ep)):
            for p in range(0, len(array_p)):
                adr = value
                dataset, data_labeled, dataset_train, l_train, stream, l_stream, n_features = u.criar_datasets(5, adr)

                start = time.time()
                if key != 7:
                    predicted, updt = handshake2.handshake2(dataset, data_labeled, dataset_train, l_train, stream, l_stream, n_components, n_features, array_ep[ep], array_p[p])
                else:
                    predicted, updt = hs.handshake2(dataset, data_labeled, dataset_train, l_train, stream, l_stream, n_components, n_features, array_ep[ep], array_p[p])

                end = time.time()
                mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                print('mem', mem)
                tempo = end - start
                name = list[int(key)]
                acc_percent = metrics.makeBatches(l_stream, predicted, len(stream))
                score, f1, mcc = metrics.metrics(acc_percent, l_stream, predicted, f1_type = 'macro')
                u.saveLog2(name, array_ep[ep], array_p[p], updt, acc_percent, score, f1, mcc, tempo, mem)
                # u.saveLog(list[int(key)], acc_percent, score, f1, mcc, updt)
                print(key, 'episilon: ', ep, 'percentage: ', p)


if __name__ == '__main__':
	main()
