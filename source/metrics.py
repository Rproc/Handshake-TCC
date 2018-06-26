import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from math import sqrt


# Retorna avaliação de acertos == dividos em 10% do tamanho da stream
def makeBatches(label_stream, predicted_label, size_stream):

    newlist_original = label_stream[int((0/10)*size_stream):int(((0+1)/10)*size_stream)]
    newlist_predicted = predicted_label[int((0/10)*size_stream):int(((0+1)/10)*size_stream)]

    score = accuracy_score(newlist_original, newlist_predicted)
    data_acc = score

    for i in range(1, 10):
        newlist_original = label_stream[int((i/10)*size_stream)+1:int(((i+1)/10)*size_stream)]
        newlist_predicted = predicted_label[int((i/10)*size_stream)+1:int(((i+1)/10)*size_stream)]

        score = accuracy_score(newlist_original, newlist_predicted)
        data_acc = np.column_stack((data_acc, score))

    return data_acc


def metrics(l_stream, predicted, f1_type = 'binary'):

    score = round(accuracy_score(l_stream, predicted), 4)
    f1 = f1_score(l_stream, predicted, average = f1_type)
    mcc = matthews_corrcoef(l_stream, predicted)

    return score, f1, mcc
