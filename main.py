# from source import scargc
from source.util import utils as u
from source import metrics
from source import handshake, handshake2
import sys

def main():

    # poolsize = 50
    # clusters = 2
    n_components = 2
    episilon = 0.15
    n_features = 2
    base = '/home/localuser/Documentos/procopio/tcc/datasets/'
    # base = '/home/procopio/Documents/tcc/datasets/'
    # base = '/home/god/Documents/ccomp/tcc/datasets/'
    list = ['1CDT.txt', '1CHT.txt', '1CSurr.txt', '2CDT.txt', '2CHT.txt']
    database = {}

    for i in range(0, len(list)):
        database[i] = base + list[i]


    array_ep = [0.15, 0.20, 0.25]
    array_p = [10, 20, 30]
    # arr_predict = []
    # arr_updt = []

    for key, value in database.items():
        if (key != 2):
            continue
        for ep in range(0, len(array_ep)):
            for p in range(0, len(array_p)):
                adr = value
                dataset, data_labeled, dataset_train, l_train, stream, l_stream, n_features = u.criar_datasets(5, adr)
                predicted, updt = handshake2.handshake2(dataset, data_labeled, dataset_train, l_train, stream, l_stream, n_components, n_features, array_ep[ep], array_p[p])
                # arr_predict.append(predicted)
                # arr_updt.append(updt)
                acc_percent = metrics.makeBatches(l_stream, predicted, len(stream))
                score, f1, mcc = metrics.metrics(l_stream, predicted)
                u.saveLog2(list[int(key)], array_ep[ep], array_p[p], updt, acc_percent, score, f1, mcc)
                print(key, 'episilon: ', ep, 'percentage: ', p)
        # arr_predicted = []
        # arr_updt = []

    # dataset, data_labeled, dataset_train, l_train, stream, l_stream, n_features = u.utils.criar_datasets(5, adr)

    # d_treino, l_train, data_lab, data_labels, data_x, data_y, predicted, updt = scargc.scargc_1NN(dataset, data_labeled, dataset_train, l_train, stream, l_stream, poolsize, clusters, n_features)
    # d_treino, l_train, data_lab, data_labels, data_x, data_y, predicted, updt = handshake.handshake(dataset, data_labeled, dataset_train, l_train, stream, l_stream, poolsize, clusters, n_features, episilon)

    # predicted, updt = handshake2.handshake2(dataset, data_labeled, dataset_train, l_train, stream, l_stream, n_components, n_features, episilon)
    # acc_percent = metrics.makeBatches(l_stream, predicted, len(stream))
    # score, f1, mcc = metrics.metrics(l_stream, predicted)

    # print(acc_percent)
    # print('Accuracy:', score, '\n', 'F1:', f1, '\n', 'Matthews Correlation:', mcc, '\n')
    # print('Numero de updates', updt)

    # u.utils.saveLog('1CSurr_handshake2', acc_percent, score, f1, mcc)



if __name__ == '__main__':
	main()
