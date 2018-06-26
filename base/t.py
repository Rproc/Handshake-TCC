# import numpy as np
#
#
# a = [5, 2, 3, 8, 9, -2]
# b = []
# pool = []
# for i in range(0, 10):
#     b.append(i)
#     temp = np.hstack((a, b))
#     b = []
#     print(temp)
#
#     if i < 1:
#         pool = temp
#     else:
#         pool = np.vstack([pool, temp])
#
# print(pool)
#
# # g = np.reshape(a, (1, -1) )
# #
# # print(g)
# #
# # c = np.median(g, axis=0)
#
# # print(c)
#
#
# # x = "12, 212, 121, 212, white"
# # l = len( x.split(",") )
# #
# # print(l)
# #
# #
# # for y in range(0, l):
# #     print ("We're on time %d" % (y))
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
c = kmeans.labels_
a = kmeans.predict([[0, 0], [4, 4]])

b = kmeans.cluster_centers_

print(b)
