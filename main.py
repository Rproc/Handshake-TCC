from source import scargc
from source import util
from source import metrics


def main():

    poolsize = 50
    clusters = 2

    adr = '/home/procopio/Documents/tcc/datasets/1CDT.txt'
    dataset, data_labeled, dataset_train, l_train, stream, l_stream, n_features = util.utils.criar_datasets(5, adr)

    d_treino, l_train, data_lab, data_labels, data_x, data_y, predicted, updt = scargc.scargc_1NN(dataset, data_labeled, dataset_train, l_train, stream, l_stream, poolsize, clusters, n_features)

    acc_percent = metrics.makeBatches(l_stream, predicted, len(stream))
    score, f1, mcc = metrics.metrics(l_stream, predicted)

    print(acc_percent)
    print('Accuracy:', score, '\n', 'F1:', f1, '\n', 'Matthews Correlation:', mcc, '\n')
    print('Numero de updates', updt)




if __name__ == '__main__':
	main()
