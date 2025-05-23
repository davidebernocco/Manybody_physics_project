"""
Heisemberg S=1/2 model on a ladder.
(PBC on the two parallel legs "==")

    1==3==5==7==9      "==" : J_parallel = J*cos(theta)
    |  |  |  |  |      " | " : J_perpendicular = J*sin(theta)
    0==2==4==6==8


The Hilbert space associated to the full system is (C^2)tensor(C^2)...(C^2)

@author: david
"""

import numpy as np
import math
import time
from scipy.sparse.linalg import eigsh

from Funz_ladder import generate_binary_arrays, array_of_integers
from Funz_ladder import Block_Hamiltonian_sparse, blocks_GS, magnetisation

"""
Nr = 5    # Number of ladder rungs
N = 2*Nr  # Total number of sites on the ladder

# Interaction parameters
h = 0
J = 1                 
th = 0.5
J_par = J*math.cos(th)
J_perp = J*math.sin(th)
"""


# -----------------------------------------------------------------------------
# 1)  
# -----------------------------------------------------------------------------
"""

Sz_fix = np.asarray([i for i in range(int(-N/2), int(N/2) +1)], dtype=int)
N_1 = np.asarray([i for i in range(N+1)], dtype=int)
dict_Sz = dict(zip(Sz_fix, N_1))

# Generate the arrays associated to the sector Sz = fixed (ex. 0)
list_Sz_fixed = generate_binary_arrays(N, dict_Sz[2])

vett_m_Sz = array_of_integers(list_Sz_fixed, N)

# Zip, sort by v_m_Sz0, and unzip
paired = sorted(zip(vett_m_Sz, list_Sz_fixed))      
v_m_Sz_sorted, list_Sz_fixed_sorted = zip(*paired)

# Convert back to arrays if needed
v_m_Sz_sorted = np.asarray(v_m_Sz_sorted, dtype=np.int32)
list_Sz_fixed_sorted = np.asarray(list_Sz_fixed_sorted)



# -----------------------------------------------------------------------------
# 2) 
# -----------------------------------------------------------------------------


start_time2 = time.time()
# With sparse Hamiltonian
Block_H_Sz_sparse = Block_Hamiltonian_sparse(v_m_Sz_sorted, list_Sz_fixed_sorted, N, J_par, J_perp)
eigenvalues_sparse, eigenvectors_sparse = eigsh(Block_H_Sz_sparse, k=1, which='SA')
print("GS energy for SPARSE MATRIX", eigenvalues_sparse[0])
end_time2 = time.time()
elapsed_time2 = end_time2 - start_time2
print("Elapsed time2:", elapsed_time2)

# ////////////////////////////////////////////////////////////////////////////


# -----------------------------------------------------------------------------
# 3) Extract the m eigenvalues with a built-in efficient function
# -----------------------------------------------------------------------------
"""
#autovalori = blocks_GS(N, J_par, J_perp)




# ////////////////////////////////////////////////////////////////////////////


# -----------------------------------------------------------------------------
# 4) Obtain the "magnetization vs h" plot, starting from energy eigenvalues
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False
"""
# Discontinuity points (edges of steps)
x_steps, y_heights = magnetisation(autovalori)


x_plot = np.insert(x_steps, 0, 0)  
y_plot = y_heights              
y_plot =  np.append(y_plot, y_plot[-1])  


# Plot of step function
fig_m, ax_m = plt.subplots(figsize=(6.2, 4.5))
ax_m.step(x_plot, y_plot, where='post')
ax_m.set_xlabel(r'$ h $', fontsize=15)
ax_m.set_ylabel(r'$ m $', fontsize=15)
ax_m.grid(True)
plt.show()
"""

# ///////////////////////////////////////////////////////////////////////////

# Size scaling at fixed interaction parameters

def multiple_plot(n_min, n_max, param_h, param_J, param_th ):
   
    # Interaction parameters
    param_J_par = param_J * math.cos(param_th)
    param_J_perp = param_J * math.sin(param_th)
    
    fig_m, ax_m = plt.subplots(figsize=(6.2, 4.5))
    
    for i in range(n_min, n_max+1):
        Num = 2*i  # Total number of sites on the ladder
        autovalori = blocks_GS(Num, param_J_par, param_J_perp)
        
        # Discontinuity points (edges of steps)
        x_steps, y_heights = magnetisation(autovalori)


        x_plot = np.insert(x_steps, 0, 0)  
        y_plot = y_heights              
        y_plot =  np.append(y_plot, y_plot[-1])  


        # Plot of step function
        
        ax_m.step(x_plot, y_plot, where='post', label=f'{i} rungs ')
    ax_m.set_xlabel(r'$ h $', fontsize=15)
    ax_m.set_ylabel(r'$ m $', fontsize=15)
    ax_m.set_title(fr'Magnetization for ladder Hamiltonian ($J={param_J:.2f}$, $\theta={param_th:.2f}$)')
    ax_m.legend(loc='upper left')
    ax_m.grid(True)
    plt.show()


justapposed_plots = multiple_plot(3, 6, 0, 1, 0)







