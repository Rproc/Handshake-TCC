from source import scargc
from source import util
from sklearn.metrics import accuracy_score


def main():

    poolsize = 50
    clusters = 2

    dataset, data_labeled, dataset_train, l_train, stream, l_stream, n_features = util.utils.criar_datasets(5)

    data_acc, d_treino, l_train, data_lab, data_labels, data_x, data_y, knn_labels = scargc.start_scargc(dataset, data_labeled, dataset_train, l_train, stream, l_stream, poolsize, clusters, n_features)

    score = accuracy_score(l_stream, knn_labels)
    score_quant = accuracy_score(l_stream, knn_labels, normalize=False)
    print(score, '\n', score_quant)



if __name__ == '__main__':
	main()
