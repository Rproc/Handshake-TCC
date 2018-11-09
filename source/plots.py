import numpy as np
from source import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from source.util import utils


def plotAcc(data_acc, steps, label):
    data_acc = np.multiply(data_acc[0], 100)
    # print(data_acc)
    c = range(len(data_acc))
    # print(c)
    fig = plt.figure()
    fig.add_subplot(122)
    ax = plt.axes()
    ax.plot(c, data_acc, 'k')
    plt.yticks(range(0, 101, 10))
    plt.xticks(range(0, steps+1, 10))
    plt.title(label)
    plt.ylabel("Acurácia")
    plt.xlabel("Step")
    plt.grid()
    plt.show()


# def finalEvaluation(arrAcc, steps, label):
#     # print("Average Accuracy: ", np.mean(arrAcc))
#     print("Standard Deviation: ", np.std(arrAcc))
#     plotAccuracy(arrAcc, steps, label)
