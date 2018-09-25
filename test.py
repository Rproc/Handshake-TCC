X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
x = neigh.predict([[1.7]])
p = neigh.predict_proba([[1.6]])

print(x)
i = int(x)

print(p.shape, p)

print(i)
print(p[:, i])
