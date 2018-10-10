# from source import scargc
from source import util as u
from source import metrics
from source import handshake, handshake2

def main():

    poolsize = 50
    clusters = 2
    n_components = 2
    episilon = 0.34

    # adr = '/home/procopio/Documents/tcc/datasets/2CDT.txt'
    adr = '/home/god/Documents/ccomp/tcc/datasets/2CDT.txt'

    # adr = '/home/localuser/Documentos/procopio/tcc/datasets/2CDT.txt'
    dataset, data_labeled, dataset_train, l_train, stream, l_stream, n_features = u.utils.criar_datasets(5, adr)

    # d_treino, l_train, data_lab, data_labels, data_x, data_y, predicted, updt = scargc.scargc_1NN(dataset, data_labeled, dataset_train, l_train, stream, l_stream, poolsize, clusters, n_features)
    # d_treino, l_train, data_lab, data_labels, data_x, data_y, predicted, updt = handshake.handshake(dataset, data_labeled, dataset_train, l_train, stream, l_stream, poolsize, clusters, n_features, episilon)

    handshake2.handshake2(dataset, data_labeled, dataset_train, l_train, stream, l_stream, n_components, n_features, episilon)
    # acc_percent = metrics.makeBatches(l_stream, predicted, len(stream))
    # score, f1, mcc = metrics.metrics(l_stream, predicted)
    #
    # print(acc_percent)
    # print('Accuracy:', score, '\n', 'F1:', f1, '\n', 'Matthews Correlation:', mcc, '\n')
    # print('Numero de updates', updt)
    #
    # util.utils.saveLog('2CDT_handshake', acc_percent, score, f1, mcc)



if __name__ == '__main__':
	main()
