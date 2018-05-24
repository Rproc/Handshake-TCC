import numpy as np


a = [5, 2, 3, 8, 9, -2]

g = np.reshape(a, (-1, 2) )

print(g)

c = np.median(g, axis=0)

print(c)


# x = "12, 212, 121, 212, white"
# l = len( x.split(",") )
#
# print(l)
#
#
# for y in range(0, l):
#     print ("We're on time %d" % (y))
