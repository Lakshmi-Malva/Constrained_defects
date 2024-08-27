import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from global_vars import *

#Q-tensor definition
def Q_ij_form(n_ij,order_param):
    #dot_prod = (n_ij.T@n_0)[0][0]
    Q_ij = 2*order_param*(np.outer(n_ij,n_ij)-I/2)
    return Q_ij

#define velocity form
def U_form_time_ind(X,Y):
    A = 0.1; omega = 2*np.pi/10; epsilon = 0.25; pi = np.pi

    u_form = lambda x,y: -A*pi*np.sin(pi*x)*np.cos(pi*y)
    v_form = lambda x,y: A*pi*np.cos(pi*x)*np.sin(pi*y)

    xv, yv = np.meshgrid(X,Y)
    u = u_form(xv, yv); v = v_form(xv, yv)
    return u, v

def U_form_time_dep(t,X=X,Y=Y):
    u = np.array([[0]*len(X)]*len(Y)).T
    v = np.array([[0]*len(X)]*len(Y)).T

    A = 0.1; omega = 2*np.pi/10; epsilon = 0.25; pi = np.pi

    a = lambda t: epsilon*np.sin(omega*t)
    b = lambda t: 1 - 2*a(t)
    f = lambda x,t: a(t)*x**2 + b(t)*x
    df = lambda x,t: 2*a(t)*x + b(t)

    u_form = lambda x,y,t: -A*pi*np.sin(pi*f(x,t))*np.cos(pi*y)
    v_form = lambda x,y,t: -A*pi*np.cos(pi*f(x,t))*np.sin(pi*y)*df(x,t)

    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            u[i][j] = u_form(x,y,t)
            v[i][j] = v_form(x,y,t)
    return u, v

def calc_ind(i,j):
    if i+1 > grid_points-1: x_next_ind = 0
    else: x_next_ind = i+1
    if j+1 > grid_points-1: y_next_ind = 0
    else: y_next_ind = j+1
    if i-1 < 0: x_beh_ind = grid_points-1
    else: x_beh_ind = i-1
    if j-1 < 0: y_beh_ind = grid_points-1
    else: y_beh_ind = j-1
    return x_next_ind, y_next_ind, x_beh_ind, y_beh_ind

#dynamic_terms
def dyn_terms(Q_global,u_global,v_global,i,j,dyn='Q'):
    x_next_ind, y_next_ind, x_beh_ind, y_beh_ind = calc_ind(i,j)
    #Q-tensor
    Q_ij = Q_global[i][j]; Q_i1j = Q_global[x_next_ind][j]; Q_ij1 = Q_global[i][y_next_ind]

    #dU/dr
    del_u_del_x = (u_global[x_next_ind][j] - u_global[i][j])/del_x
    del_v_del_y = (v_global[i][y_next_ind] - v_global[i][j])/del_y
    del_u_del_y = (u_global[i][y_next_ind] - u_global[i][j])/del_y
    del_v_del_x = (v_global[x_next_ind][j] - v_global[i][j])/del_x

    #d2U/dr2
    del2u_delx2 = (u_global[x_next_ind][j] + u_global[x_beh_ind][j] - 2*u_global[i][j])/(del_x**2)
    del2u_dely2 = (u_global[i][y_next_ind] + u_global[i][y_beh_ind] - 2*u_global[i][j])/(del_y**2)
    del2u_delxdely = (u_global[x_next_ind][y_next_ind] - u_global[x_next_ind][y_beh_ind] - u_global[x_beh_ind][y_next_ind] + u_global[x_beh_ind][y_beh_ind]) / (4*del_x*del_y)
    del2v_delx2 = (v_global[x_next_ind][j] + v_global[x_beh_ind][j] - 2*v_global[i][j])/(del_x**2)
    del2v_dely2 = (v_global[i][y_next_ind] + v_global[i][y_beh_ind] - 2*v_global[i][j])/(del_y**2)
    del2v_delydelx = (v_global[x_next_ind][y_next_ind] - v_global[x_next_ind][y_beh_ind] - v_global[x_beh_ind][y_next_ind] + v_global[x_beh_ind][y_beh_ind]) / (4*del_x*del_y)

    #gradU
    grad_u_ij = np.array([[del_u_del_x,del_u_del_y],[del_v_del_x,del_v_del_y]])

    #grad.U = 0
    grad_dot_u_ij = del_u_del_x + del_v_del_y 

    #U.grad
    u_grad = u_global[i][j]/del_x; v_grad = v_global[i][j]/del_y

    #To store velocity related functions in a dictionary
    U_dict_ij = {}
    U_dict_ij['du/dx'] = del_u_del_x; U_dict_ij['du/dy'] = del_u_del_y
    U_dict_ij['dv/dx'] = del_v_del_x; U_dict_ij['dv/dy'] = del_v_del_y
    U_dict_ij['gradU'] = grad_u_ij; U_dict_ij['grad.U'] = grad_dot_u_ij
    U_dict_ij['ud/dx'] = u_grad; U_dict_ij['vd/dy'] = v_grad

    #Vorticity
    vort_ = (del_u_del_y - del_v_del_x)/2
    vort_ij = np.array([[0,vort_],[-vort_,0]])

    #Rate of strain tensor
    E_ = (del_u_del_y + del_v_del_x)/2
    E_ij = np.array([[del_u_del_x,E_],[E_,del_v_del_y]])

    #Stress tensor
    Pi_vis_ij = 2*eta*E_ij
    Pi_act_ij = -zeta*Q_ij
    Pi_ij = Pi_vis_ij + Pi_act_ij

    #div of Pi
    div_E_ij = np.array(
        [
            [
                del2u_delx2 + (del2u_dely2 + del2v_delydelx)/2,
                (del2u_delxdely + del2v_delx2)/2 + del2v_dely2
            ]
        ]
        ).T
    
    div_Q_ij = np.array(
        [
            [
                (Q_i1j[0][0] - Q_ij[0][0])/del_x + (Q_ij1[0][1] - Q_ij[0][1])/del_y,
                (Q_i1j[1][0] - Q_ij[1][0])/del_x + (Q_ij1[1][1] - Q_ij[1][1])/del_y
            ]
        ]
    ).T

    div_Pi_ij = 2*eta*div_E_ij - zeta*div_Q_ij

    if dyn == 'Q': return U_dict_ij, vort_ij, E_ij
    if dyn == 'NVS': return U_dict_ij, div_Pi_ij
    if dyn == 'NVS+Q': return U_dict_ij, vort_ij, E_ij, div_Pi_ij

#time-evolution eq. of Q (nematodynamic eq.)
def Q_dyn_ij(t,Q_ij,Q_global,u_global,v_global,i,j):
    #Q_ij = Q_ij.reshape(2,2)
    #u_global, v_global = U_form_time_dep(t)
    U_dict_ij, vort_ij, E_ij = dyn_terms(Q_global,u_global,v_global,i,j)
    grad_u_ij = U_dict_ij['gradU']
    x_next_ind, y_next_ind, x_beh_ind, y_beh_ind = calc_ind(i,j)
    
    #S: co-rotation term
    S1 = (lam*E_ij+vort_ij)@(Q_ij+I/2)
    S2 = (lam*E_ij-vort_ij)@(Q_ij+I/2)
    S3 = 2*lam*np.trace(np.outer(Q_ij,grad_u_ij.T))*(Q_ij+I/2)
    S_ij = S1 + S2 - S3
    print(S_ij)

    #(U.grad)Q
    u_grad = U_dict_ij['ud/dx']; v_grad = U_dict_ij['vd/dy']
    Q_i1j = Q_global[x_next_ind][j]; Q_ij1 = Q_global[i][y_next_ind]
    grad_Q_11 = u_grad*(Q_i1j[0][0]-Q_ij[0][0]) + v_grad*(Q_ij1[0][0]-Q_ij[0][0])
    grad_Q_12 = u_grad*(Q_i1j[0][1]-Q_ij[0][1]) + v_grad*(Q_ij1[0][1]-Q_ij[0][1])
    grad_Q_21 = u_grad*(Q_i1j[1][0]-Q_ij[1][0]) + v_grad*(Q_ij1[1][0]-Q_ij[1][0])
    grad_Q_22 = u_grad*(Q_i1j[1][1]-Q_ij[1][1]) + v_grad*(Q_ij1[1][1]-Q_ij[1][1])
    u_grad_Q_ij = [[grad_Q_11,grad_Q_12],[grad_Q_21,grad_Q_22]]

    #delQ/delt = S - (U.grad)Q
    f_Q_ij = S_ij - u_grad_Q_ij
    #print(f_Q_ij)

    return f_Q_ij

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

def rk4singlestep(fun, dt, t0, y0, *args):
    f1 = fun(t0, y0, *args)
    #print(f1)
    f2 = fun(t0 + dt/2, y0 + f1*(dt/2), *args)
    #print(f2)
    f3 = fun(t0 + dt/2, y0 + f2*(dt/2), *args)
    f4 = fun(t0 + dt, y0 + f3*dt, *args)
    y_out = y0 + (dt/6)*(f1 + 2*(f2 + f3) + f4)
    return y_out

'''for t in tqdm(T_steps,desc=f'Time', leave=True):
    for i in trange(len(X),desc=f'X Iters:{i}', leave=False):
        for j in trange(len(Y),desc=f'Y Iters:{j}', leave=False):
            Q_0_ij = Q_global_sol[i][j].reshape(4,)
            Q_sol_ij = solve_ivp(Q_dyn_ij, t_span=(t,t+dt), y0=Q_0_ij, vectorized=True, args=(Q_global_sol,u_global,v_global,i,j))
            Q_global_sol[i][j] = Q_sol_ij.y.T[-1].reshape(2,2)
    print(Q_global_sol)'''