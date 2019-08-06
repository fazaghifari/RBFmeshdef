import numpy as np
import kernel
from numpy.linalg import solve as mldivide

def interp (node_bound,g,r,mesh):

    m, dimen = np.shape(node_bound)
    n = np.size(mesh,0)
    p = np.zeros(shape=[m,dimen+1])
    p[:,0] = 1
    for i in range(0,m):
        p[i,1:dimen+1] = node_bound[i,0:dimen]

    node_new = node_bound+g;
    M = np.zeros(shape=[m,m])
    sigma = 1


    for i in range(0,m):
        for j in range(0,m):
            x = np.sqrt(np.sum((node_new[i,:]-node_new[j,:])**2))
            M[i,j] = kernel.ctpsc2b(x,r)

    intpM = np.zeros(shape=[m+dimen+1,m+dimen+1])

    for i in range(0,m+dimen+1):
        for j in range(0,m+dimen+1):
            if i<=(m-1) and j<=(m-1):
                intpM[i,j] = M[i,j]
            elif i<=(m-1) and j>(m-1):
                intpM[i,j] = p[i,j-(m-1)]
            elif i>(m-1) and j<=(m-1):
                pt = np.transpose(p)
                intpM[i,j] = pt[i-(m-1),j]

    g0 = np.zeros(shape=[m+dimen+1,dimen])
    g0[0:m,:] = g
    gamma_beta = mldivide(intpM,g0)

    gamma = gamma_beta[0:m,:]
    beta = gamma_beta[m:m+dimen+1,:]
    v = np.zeros(n,dimen)

    for i in range(0,n):
        temp = np.zeros(shape=[1,dimen])
        for j in range(0,m):
            x = np.sqrt(np.sum((mesh[i,0:dimen] - node_new[j, :]) ** 2))
            temp = gamma[j,:]*kernel.ctpsc2b(x,r)+temp

        if temp == 0:
            v[i,:] = temp
        else:
            v[i,:] = temp + (np.dot(np.array([1, mesh[i,0:dimen]]),beta))

    return v