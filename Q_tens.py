import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Nemat_dyn_func import *
from global_vars import *
from tqdm import tqdm, trange
from colour import Color
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from cfpack import stop

#Output path
out_folder = './Results/Q_tens/Time_Ind/'
#out_folder = './Results/Q_tens/Time_Dep/'
try: os.makedirs(out_folder)
except: pass

#Initial conditions
n_0 = np.array([[0,1]]).T
Q_0 = Q_ij_form(n_0)
q1_0 = Q_0[0][0]; q2_0 = Q_0[1][0]
Q1_0 = np.ones((len(X),len(Y)),dtype=float)*q1_0
Q2_0 = np.ones((len(X),len(Y)),dtype=float)*q2_0

X_mesh, Y_mesh = np.meshgrid(X,Y)
#initial velocity: time-independent
u_global, v_global = U_xy_time_ind(X,Y)
U_dict_ij = dyn_terms(u_global,v_global)
T_steps = np.arange(0,T+dt,dt)

#Eig of rate of strain tensor E
E_eigVal, E_eigVec_x, E_eigVec_y = get_E(u_global,v_global)

#solve for Q
Q_term_0 = (list(Q1_0.flatten()) + list(Q2_0.flatten()))
stop()
sol = solve_ivp(Q_dyn_t, t_span=(0,T), y0 = Q_term_0, vectorized=True, args=(U_dict_ij,))
#sol = solve_ivp(Q_dyn_time_t, t_span=(0,T), y0 = Q_term_0, vectorized=True)

T_sol = sol.t
green = Color("green")
colors = list(green.range_to(Color("orange"),len(T_sol)))

Q_1, Q_2, Q_t = get_Q(sol)

Q_eigVal, Q_eigVec_x, Q_eigVec_y,  Q_det = [], [], [], []
out_file = lambda ind: out_folder + f'E_Q_{ind}.pdf'

for ind, qt in enumerate(Q_t):

    val, vec_x, vec_y = largest_eignval(qt)
    Q_eigVal.append(val)
    Q_eigVec_x.append(vec_x); Q_eigVec_y.append(vec_x)
    det_ = -(Q_1[ind]**2 + Q_2[ind]**2)
    Q_det.append(det_)

    fig = plt.figure(figsize =(4, 4)) 
    plt.xlabel('x'); plt.ylabel('y')

    plt.contourf(X_mesh, Y_mesh, det_)
    plt.colorbar()

    plt.quiver(X_mesh, Y_mesh, E_eigVec_x, E_eigVec_y, color = 'k')
    plt.quiver(X_mesh, Y_mesh, vec_x, vec_x, color = 'white')

    plt.savefig(out_file(ind),format = "pdf",dpi=300,bbox_inches='tight')
    plt.close()






