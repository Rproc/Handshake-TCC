import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from math import sqrt
import sys

# Retorna avaliação de acertos == dividos em 10% do tamanho da stream
def makeBatches(label_stream, predicted_label, size_stream, step):

    newlist_original = label_stream[int((0/step)*size_stream):int(((0+1)/step)*size_stream)]
    newlist_predicted = predicted_label[int((0/step)*size_stream):int(((0+1)/step)*size_stream)]

    # print(len(newlist_original))
    # print(len(newlist_predicted))
    score = accuracy_score(newlist_original, newlist_predicted)
    f1 = f1_score(newlist_original, newlist_predicted, average='macro')
    mcc = matthews_corrcoef(newlist_original, newlist_predicted)
    data_acc = score
    data_f1 = f1
    data_mcc = mcc

    for i in range(1, step):
        newlist_original = label_stream[int((i/step)*size_stream)+1:int(((i+1)/step)*size_stream)]
        newlist_predicted = predicted_label[int((i/step)*size_stream)+1:int(((i+1)/step)*size_stream)]

        score = accuracy_score(newlist_original, newlist_predicted)
        f1 = f1_score(newlist_original, newlist_predicted, average='macro')
        mcc = matthews_corrcoef(newlist_original, newlist_predicted)

        data_acc = np.column_stack((data_acc, score))
        data_f1 = np.column_stack((data_f1, f1))
        data_mcc = np.column_stack((data_mcc, mcc))


    return data_acc, data_f1, data_mcc


def metrics(data_acc, l_stream, predicted, step, f1_type = 'binary'):

    # print(l_stream)
    # print(predicted)
    # predicted = predicted.flatten()
    score = np.sum(data_acc)
    score = score/step
    f1 = f1_score(l_stream, predicted, average = f1_type)
    mcc = matthews_corrcoef(l_stream, predicted)
    std = np.std(data_acc)

    return score, f1, mcc, std
