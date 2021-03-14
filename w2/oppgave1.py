import numpy as np
a = np.ones((3,1,3))
b = np.ones((2,3,3))
c = np.ones((4,1))
d = np.ones((3,1,1,5))
e = np.ones((3))
f = np.ones((3,1,1,5))
g = np.ones((1,4))
h = np.ones((7,1))
i = np.ones((6,3,1,7))
j = np.ones((2,7))
k = np.ones((6,3,1,7))
l = np.ones((2,1,7))
m = np.ones((1,2,3,1,6))
n = np.ones((8,1,3,2,6))
o = np.ones((2,5,1,7))
p = np.ones((9,2,3,2,1))

values = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p]

"""
ab = false
cd = (3,1,4,5)
ef = false
gh = (7,4)
ij = (6,3,2,7)
kl = false
mn = (8,2,3,2,6)
op = false
"""

for i in range(int(len(values)/2)):
    try:
        result = values[i*2]*values[i*2+1]
        print("successfull broadcast for shapes {} and {} \nResulting shape: {}\n".format(values[i*2].shape,values[i*2+1].shape, result.shape))
    except ValueError:
        print("Broadcasting failed for shapes {} and {}\n".format(values[i*2].shape,values[i*2+1].shape))
