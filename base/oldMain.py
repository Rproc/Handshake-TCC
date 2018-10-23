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
