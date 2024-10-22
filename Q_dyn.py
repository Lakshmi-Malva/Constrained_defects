import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from math import prod
from cfpack import stop
import time
from numba import jit

#Global constants
grid_points = 17
L = 1 #box_length
del_x = 1/(grid_points-1); del_y = del_x #grid-space
X = np.arange(0,L+del_x,del_x); Y = np.arange(0,L+del_y,del_y) #grids
#for an extra grid: spatial derivative purpose
X_u = np.arange(-del_x,L+2*del_x,del_x); Y_v = np.arange(-del_y,L+2*del_y,del_y) 
T = 1; lam = 1; I = np.identity(2)
dt = 1e-3

# (Ignore this) Q-tensor definition 
def Q_ij_form(n_ij,order_param = 1):
    Q_ij = 2*order_param*(np.outer(n_ij,n_ij)-I/2)
    return Q_ij

#This is the velocity field
def U_centre_time_ind(X_u,Y_v):
    xv, yv = np.meshgrid(X_u,Y_v)
    mod_ = xv**2 + yv**2
    sigma = 1e-2
    u = -xv*np.exp(-mod_/sigma**2)
    v = yv*np.exp(-mod_/sigma**2)
    return u, v

def U_cons_strain_rate(X_u,Y_v):
    xv, yv = np.meshgrid(X_u,Y_v)
    a, b, c = 1,1,2
    d = -a
    u = a*xv + b*yv
    v = c*xv + d*yv
    return u, v
    
#This function calculates derivatives of velocity field (first-order)
def dyn_terms_non_period(u_global,v_global):
    u_global, v_global = np.array(u_global), np.array(v_global)
    rows, columns = u_global.shape
    row_ind, column_ind = np.arange(rows), np.arange(columns)

    u = np.delete(u_global, -1, 0); v = np.delete(v_global, -1, 0)
    u = np.delete(u, 0, 0); v = np.delete(v, 0, 0)
    u = np.delete(u, -1, 1); v = np.delete(v, -1, 1)
    u = np.delete(u, 0, 1); v = np.delete(v, 0, 1)
    
    #u[:,-1] = 0; v[-1,:] = 0; u[:,0] = 0; v[0,:] = 0

    x_next_ind, x_beh_ind = column_ind+1, column_ind-1
    
    del_u_del_x = (u_global[1:-1, x_next_ind[1:-1]] - u) / del_x
    del_v_del_x = (v_global[1:-1, x_next_ind[1:-1]] - v) / del_x

    y_next_ind, y_beh_ind = row_ind+1, row_ind-1

    del_u_del_y = (u_global[y_next_ind[1:-1], 1:-1] - u) / del_y
    del_v_del_y = (v_global[y_next_ind[1:-1], 1:-1] - v) / del_y

    #To store velocity related functions in a dictionary
    U_dict_ij = {}
    
    U_dict_ij['u'] = u; U_dict_ij['v'] = v 
    U_dict_ij['du/dx'] = del_u_del_x; U_dict_ij['du/dy'] = del_u_del_y
    U_dict_ij['dv/dx'] = del_v_del_x; U_dict_ij['dv/dy'] = del_v_del_y
   
    #Vorticity
    vort_ = (del_u_del_y - del_v_del_x)/2

    #Rate of strain tensor
    E_ = (del_u_del_y + del_v_del_x)/2

    U_dict_ij['e'] = E_; U_dict_ij['w'] = vort_

    return U_dict_ij

def Q_dyn_t(t,Q_terms,U_dict_ij):
    
    q1 = np.array(Q_terms[:grid_points**2]).reshape((grid_points, grid_points))
    q2 = np.array(Q_terms[grid_points**2:]).reshape((grid_points, grid_points))

    grad_U_diff = U_dict_ij['dv/dy'] - U_dict_ij['du/dx']
    q1q2 = q1*q2
    q1_sq = q1**2
    q2_sq = q2**2

    S_1 = 2*lam*(q2_sq*grad_U_diff) - 4*lam*(U_dict_ij['e']*q1q2) - 2*(U_dict_ij['w']*q2) - (lam/2)*grad_U_diff
    
    S_2 = 2*lam*(q1q2*grad_U_diff) + (lam*U_dict_ij['e'])*(1 - 4*q2_sq) + 2*U_dict_ij['w']*q1 
    
    return [S_1, S_2]

if __name__ == "__main__":

    X_mesh, Y_mesh = np.meshgrid(X,Y)

    #Initial conditions
    n_0 = np.array([[0,1]]).T
    Q_0 = Q_ij_form(n_0)
    q1_0 = Q_0[0][0]; q2_0 = Q_0[1][0]
    Q1_0 = np.ones((len(X),len(Y)),dtype=float)*q1_0
    Q2_0 = np.ones((len(X),len(Y)),dtype=float)*q2_0

    #initial velocity: time-independent
    u_global, v_global = U_centre_time_ind(X_u,Y_v)
    U_dict_ij = dyn_terms_non_period(u_global,v_global)
    '''
    plt.quiver(X_u,Y_v,u_global,v_global)
    plt.show()
    '''

    #solve for Q
    Q_term_0 = (list(Q1_0.flatten()) + list(Q2_0.flatten()))
    start_time = time.time()
    sol = solve_ivp(Q_dyn_t, t_span=(0,T), y0 = Q_term_0, args=(U_dict_ij,), vectorized=True, max_step=dt, atol=1e-9, rtol=1e-9)
    print("--- %s seconds ---" % (time.time() - start_time))
    stop()
