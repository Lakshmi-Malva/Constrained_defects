import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.linalg import eigh, det
from global_vars import *
from math import prod

#Q-tensor definition
def Q_ij_form(n_ij,order_param = 1):
    #dot_prod = (n_ij.T@n_0)[0][0]
    Q_ij = 2*order_param*(np.outer(n_ij,n_ij)-I/2)
    return Q_ij

#define velocity form
def U_DG_time_ind(X,Y):
    A = 0.1; omega = 2*np.pi/10; epsilon = 0.25; pi = np.pi

    u_form = lambda x,y: -A*pi*np.sin(pi*x)*np.cos(pi*y)
    v_form = lambda x,y: A*pi*np.cos(pi*x)*np.sin(pi*y)

    xv, yv = np.meshgrid(X,Y)
    u = u_form(xv, yv); v = v_form(xv, yv)
    return u, v

def U_xy_time_ind(X,Y):
    xv, yv = np.meshgrid(X,Y)
    return -xv, yv

def U_DG_time_dep(t,X=X,Y=Y):
    A = 0.1; omega = 2*np.pi/10; epsilon = 0.25; pi = np.pi

    a = lambda t: epsilon*np.sin(omega*t)
    b = lambda t: 1 - 2*a(t)
    f = lambda x,t: a(t)*x**2 + b(t)*x
    df = lambda x,t: 2*a(t)*x + b(t)

    u_form = lambda x,y,t: -A*pi*np.sin(pi*f(x,t))*np.cos(pi*y)
    v_form = lambda x,y,t: -A*pi*np.cos(pi*f(x,t))*np.sin(pi*y)*df(x,t)

    xv, yv = np.meshgrid(X,Y)
    u = u_form(xv, yv, t); v = v_form(xv, yv, t)
    return u, v

def calc_ind(index):
    if index+1 > grid_points-1: next_ind = 1
    else: next_ind = index+1
    if index-1 < 0: beh_ind = grid_points-1
    else: beh_ind = index-1
    return next_ind, beh_ind

#dynamic_terms
def dyn_terms(u_global,v_global,incompressible=True):
    u_global, v_global = np.array(u_global), np.array(v_global)
    rows, columns = u_global.shape
    calc_ind_vect = np.vectorize(calc_ind)

    x_next_ind, x_beh_ind = calc_ind_vect(np.arange(columns))
    del_u_del_x = (u_global[:, x_next_ind] - u_global) / del_x
    del_v_del_x = (v_global[:, x_next_ind] - v_global) / del_x
    del2u_delx2 = (u_global[:, x_next_ind] + u_global[:, x_beh_ind] - 2*u_global) / (del_x**2)
    del2v_delx2 = (v_global[:, x_next_ind] + v_global[:, x_beh_ind] - 2*v_global) / (del_x**2)

    y_next_ind, y_beh_ind = calc_ind_vect(np.arange(rows))

    #du/dx + dv/dy = 0
    if incompressible: del_v_del_y = -del_u_del_x
    else: del_v_del_y = (v_global[y_next_ind, :] - v_global) / del_y

    del_u_del_y = (u_global[y_next_ind, :] - u_global) / del_y
    del2u_dely2 = (u_global[y_next_ind, :] + u_global[y_beh_ind, :] - 2*u_global) / (del_y**2)
    del2v_dely2 = (v_global[y_next_ind, :] + v_global[y_beh_ind, :] - 2*v_global) / (del_y**2)

    if incompressible: 
        del2u_delxdely = -del2v_dely2
        del2v_delydelx = -del2u_delx2
    else:
        del2u_delxdely = (u_global[np.ix_(y_next_ind, x_next_ind)] - 
                  u_global[np.ix_(y_beh_ind ,x_next_ind)] - 
                  u_global[np.ix_(y_next_ind, x_beh_ind)] + 
                  u_global[np.ix_(y_beh_ind, x_beh_ind)]) / (4*del_x*del_y)
        del2v_delydelx = (v_global[np.ix_(y_next_ind, x_next_ind)] - 
                  v_global[np.ix_(y_beh_ind ,x_next_ind)] - 
                  v_global[np.ix_(y_next_ind, x_beh_ind)] + 
                  v_global[np.ix_(y_beh_ind, x_beh_ind)]) / (4*del_x*del_y)
    
    #U.grad
    u_grad = u_global/del_x; v_grad = v_global/del_y

    #To store velocity related functions in a dictionary
    U_dict_ij = {}
    U_dict_ij['du/dx'] = del_u_del_x; U_dict_ij['du/dy'] = del_u_del_y
    U_dict_ij['dv/dx'] = del_v_del_x; U_dict_ij['dv/dy'] = del_v_del_y
    U_dict_ij['ud/dx'] = u_grad; U_dict_ij['vd/dy'] = v_grad

    #Vorticity
    vort_ = (del_u_del_y - del_v_del_x)/2

    #Rate of strain tensor
    E_ = (del_u_del_y + del_v_del_x)/2

    U_dict_ij['e'] = E_; U_dict_ij['w'] = vort_

    return U_dict_ij

#time-evolution eq. of Q (nematodynamic eq.)
#Time_independent velocity field
def Q_dyn_t(t,Q_terms,U_dict_ij):
    
    q1 = np.array(Q_terms[:grid_points**2]).reshape((grid_points, grid_points))
    q2 = np.array(Q_terms[grid_points**2:]).reshape((grid_points, grid_points))
    
    S_1 = U_dict_ij['du/dx']*lam*(-2*q1**2 + q1 + 1) + U_dict_ij['dv/dy']*lam*q1*(2*q1 + 1) - 4*lam*U_dict_ij['e']*q1*q2 + 2*U_dict_ij['w']*q2
    f_q1 = S_1 - (U_dict_ij['ud/dx'] + U_dict_ij['vd/dy'])*q1

    S_2 = U_dict_ij['du/dx']*lam*q2*(1 - 2*q1) + U_dict_ij['dv/dy']*lam*q2*(1 + 2*q1) - 4*lam*U_dict_ij['e']*q2**2 - 2*U_dict_ij['w']*q1
    f_q2 = S_2 - (U_dict_ij['ud/dx'] + U_dict_ij['vd/dy'])*q2

    return [f_q1, f_q2]

def Q_dyn_time_t(t,Q_terms):
    
    u_global, v_global = U_form_time_dep(t)
    U_dict_ij = dyn_terms(u_global,v_global)
    q1 = np.array(Q_terms[:grid_points**2]).reshape((grid_points, grid_points))
    q2 = np.array(Q_terms[grid_points**2:]).reshape((grid_points, grid_points))
    
    S_1 = U_dict_ij['du/dx']*lam*(-2*q1**2 + q1 + 1) + U_dict_ij['dv/dy']*lam*q1*(2*q1 + 1) - 4*lam*U_dict_ij['e']*q1*q2 + 2*U_dict_ij['w']*q2
    f_q1 = S_1 - (U_dict_ij['ud/dx'] + U_dict_ij['vd/dy'])*q1

    S_2 = U_dict_ij['du/dx']*lam*q2*(1 - 2*q1) + U_dict_ij['dv/dy']*lam*q2*(1 + 2*q1) - 4*lam*U_dict_ij['e']*q2**2 - 2*U_dict_ij['w']*q1
    f_q2 = S_2 - (U_dict_ij['ud/dx'] + U_dict_ij['vd/dy'])*q2

    return [f_q1, f_q2]

#Incompressible NVS
def NVS(u_global,v_global):
    U_dict_ij, div_Pi_ij = dyn_terms(Q_global,u_global,v_global,dyn='NVS')

    #(U.grad)U
    u_grad_u_ij = np.array(
        [
            [
                u_global[i][j]*U_dict_ij['du/dx'] + v_global[i][j]*U_dict_ij['du/dy'],
                u_global[i][j]*U_dict_ij['dv/dx'] + v_global[i][j]*U_dict_ij['dv/dy']
            ]
        ]
    )

    f_U_ij = div_Pi_ij/rho - u_grad_u_ij

    return f_U_ij

def get_Q(sol):
    Q_1 = []; Q_2 = []
    rows, columns = sol.y.shape
    for col in range(columns):
        q1 = np.array(sol.y[:grid_points**2,col:col+1]).reshape((grid_points, grid_points))
        q2 = np.array(sol.y[grid_points**2:,col:col+1]).reshape((grid_points, grid_points))
        Q_1.append(q1); Q_2.append(q2)
    
    Q_1, Q_2 = np.array(Q_1), np.array(Q_2)
    
    def create_Q(Q1, Q2):
        rows, cols = Q1.shape
        Q_mat = np.zeros((rows, cols, 2, 2))
        
        Q_mat[:, :, 0, 0] = Q1
        Q_mat[:, :, 0, 1] = Q2
        Q_mat[:, :, 1, 0] = Q2
        Q_mat[:, :, 1, 1] = -Q1
        return Q_mat

    Q_t = np.zeros(len(sol.t)).astype('object')
    for ind in range(len(sol.t)):
        Q_t[ind] = create_Q(Q_1[ind], Q_2[ind])
    
    return Q_1, Q_2, Q_t

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

    vec_max_x = vec_max[:,0]
    vec_max_y = vec_max[:,1]

    if reshape_done: 
        val_max = val_max.reshape(*reshape_list)
        vec_max_x = vec_max_x.reshape(*reshape_list)
        vec_max_y = vec_max_y.reshape(*reshape_list)
    
    return val_max, vec_max_x, vec_max_y

def det_mat(mat):
    reshape_done = False
    if len(mat.shape) > 3: 
        reshape_done = True
        reshape_list = mat.shape[:-2]
        mat = mat.reshape(prod(reshape_list), 2, 2)

    det_ = np.array(list(map(det, mat)))

    if reshape_done: det_ = det_.reshape(*reshape_list)
    return det_

def get_E(u_global,v_global):
    U_dict_ij = dyn_terms(u_global,v_global)

    def create_E(e, ux, vy):
        rows, cols = e.shape
        E_mat = np.zeros((rows, cols, 2, 2))
        
        E_mat[:, :, 0, 0] = ux
        E_mat[:, :, 0, 1] = e
        E_mat[:, :, 1, 0] = e
        E_mat[:, :, 1, 1] = vy
        return E_mat

    E_mat = create_E(U_dict_ij['e'], U_dict_ij['du/dx'], U_dict_ij['dv/dy'])

    return largest_eignval(E_mat)

def rk4singlestep(fun, dt, t0, y0, *args):
    f1 = fun(t0, y0, *args)
    #print(f1)
    f2 = fun(t0 + dt/2, y0 + f1*(dt/2), *args)
    #print(f2)
    f3 = fun(t0 + dt/2, y0 + f2*(dt/2), *args)
    f4 = fun(t0 + dt, y0 + f3*dt, *args)
    y_out = y0 + (dt/6)*(f1 + 2*(f2 + f3) + f4)
    return y_out