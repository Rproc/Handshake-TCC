import numpy as np
from sklearn.decomposition import PCA
from source.util import utils as u
from source import metrics, plots
from source import scargc, hs







base = '/home/test/Documentos/Handshake-TCC/datasets/1CSurr.txt'

dataset, data_labeled, dataset_train, l_train, stream, l_stream, n_features = u.criar_datasets(5, base)

pca = PCA(.95)
pca.fit(dataset_train)
d_treino_pca = pca.transform(dataset_train)

print(pca.explained_variance_ratio_)

print(pca.n_components_)
