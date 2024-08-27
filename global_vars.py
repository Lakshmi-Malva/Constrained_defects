import numpy as np

grid_points = 17#257
L = 1 #box_length
del_x = 1/(grid_points-1); del_y = del_x #grid-space
X = np.arange(0,L+del_x,del_x); Y = np.arange(0,L+del_y,del_y) #grids
T = 5
dt = 1e-1
rho = 1 #density 
lam = 1 #tumbling parameter
zeta = 0.1
eta = 4/3
I = np.identity(2) #2-D identity matrix