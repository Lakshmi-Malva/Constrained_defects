import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh as seigh
from numpy.linalg import eig, eigh

A = np.array([[[[7, 2], [2, 9]], [[3, 4], [6, 1]]],
[[[7, 2], [2, 9]], [[3, 4], [6, 1]]]]).astype(np.float64)
B = np.array([A,A])#.reshape(8,2,2)
C = np.array([[[7, 2], [2, 9]], [[3, 4], [6, 1]]])

def largest_eignval(mat):

    reshape_done = False
    if len(mat.shape) > 3: 
        reshape_done = True
        reshape_list = mat.shape[:-2]
        mat = mat.reshape(prod(reshape_list), 2, 2)

    val, vec = eigh(mat)
    val_max = np.array(list(map(max, list(val))))
    val_max_ind = np.array(list(map(np.argmax, val)))
    vec_max = np.array(list(map(lambda mat, i: mat[:,i], vec, val_max_ind)))

    if reshape_done: 
        val_max = val_max.reshape(*reshape_list)
        vec_max = vec_max.reshape(*reshape_list, 2)
    
    return val_max, vec_max

