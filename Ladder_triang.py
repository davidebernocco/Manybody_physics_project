"""
Heisemberg S=1/2 model on a TRIANGULAR ladder.
(PBC on the two parallel legs "==")

      1==3==5==7==9      "==" : J_parallel = J*cos(theta)
     / \/ \/ \/ \/      " | " : J_perpendicular = J*sin(theta)
    0==2==4==6==8


The Hilbert space associated to the full system is (C^2)tensor(C^2)...(C^2)

@author: david
"""

import numpy as np
import math
from scipy.sparse import lil_matrix
from scipy.linalg import eigh

from Funz_ladder import generate_binary_arrays, array_of_integers
from Funz_ladder import Block_Hamiltonian_sparse, magnetisation
from scipy.sparse.linalg import eigsh



Nr = 6    # Number of ladder rungs
N = 2*Nr  # Total number of sites on the ladder

# Interaction parameters
h = 0
J = 1                 
th = math.pi/2
J_par = J*math.cos(th)
J_perp = J*math.sin(th)



def Block_Hamiltonian_sparse_TR(v, lst, n, Jpar, Jperp):
    d = len(lst)
    H_Sz = lil_matrix((d, d), dtype=np.float32)
    
    v_index = {val: idx for idx, val in enumerate(v)}  # dictionary for fast lookup

    for row, lst_j in enumerate(lst):
        # Off-diagonal J_par
        for i in range(n):
            if lst_j[i] != lst_j[(i+2)%n]:
                vet = lst_j.copy()
                vet[i], vet[(i+2)%n] = vet[(i+2)%n], vet[i]
                m = sum(vet[k] * (2**k) for k in range(n))
                column = v_index[m]
                H_Sz[row, column] -= Jpar / 2

        # Diagonal J_par
        vet = lst_j - 0.5
        h1 = sum(vet[i] * vet[(i+2)%n] for i in range(n))
        H_Sz[row, row] += Jpar * h1

        # Off-diagonal J_perp
        for i in range(n):
            if lst_j[i] != lst_j[(i+1)%n]:
                vet = lst_j.copy()
                vet[i], vet[(i+1)%n] = vet[(i+1)%n], vet[i]
                m = sum(vet[k] * (2**k) for k in range(n))
                column = v_index[m]
                H_Sz[row, column] -= Jperp / 2

        # Diagonal J_perp
        vet = lst_j - 0.5
        h2 = sum(vet[i] * vet[(i+1)%n] for i in range(n))
        H_Sz[row, row] += Jperp * h2

    return H_Sz.tocsr()

"""
Sz_fix_TR = np.asarray([i for i in range(int(-N/2), int(N/2) +1)], dtype=int)
N_1_TR = np.asarray([i for i in range(N+1)], dtype=int)
dict_Sz_TR = dict(zip(Sz_fix_TR, N_1_TR))

# Generate the arrays associated to the sector Sz = fixed
list_Sz_fixed_TR = generate_binary_arrays(N, dict_Sz_TR[0])

vett_m_Sz_TR = array_of_integers(list_Sz_fixed_TR, N)

# Zip, sort by v_m_Sz0, and unzip
paired_TR = sorted(zip(vett_m_Sz_TR, list_Sz_fixed_TR))      
v_m_Sz_sorted_TR, list_Sz_fixed_sorted_TR = zip(*paired_TR)

# Convert back to arrays if needed
v_m_Sz_sorted_TR = np.asarray(v_m_Sz_sorted_TR, dtype=np.int32)
list_Sz_fixed_sorted_TR = np.asarray(list_Sz_fixed_sorted_TR)

# Sparse Hamiltonian
Block_H_Sz_sparse_TR = Block_Hamiltonian_sparse_TR(v_m_Sz_sorted_TR, list_Sz_fixed_sorted_TR, N, J_par, J_perp)

# Convert to dense
A_dense_TR = Block_H_Sz_sparse_TR.toarray()   

# Compute all eigenvalues (interesting for the degeneracy)
eigenvalues_TR, _ = eigh(A_dense_TR)
"""



def blocks_GS_TR(n, Jpar, Jperp):
    Sz_fix = np.asarray([i for i in range(int(-n/2), int(n/2) +1)], dtype=int)
    n_1 = np.asarray([i for i in range(n+1)], dtype=int)
    dict_Sz = dict(zip(Sz_fix, n_1))
    d = len(Sz_fix)
    E_gs = np.zeros(d, dtype=np.float32)
    
    j = 0
    for sz in Sz_fix:
        if sz == int(-n/2) or sz == int(n/2):
            # Manually add the energy of the 1x1 sectors of |Sz| max
            E_gs[j] += np.sign(sz) * (Jpar*n/4 + Jperp*n/4)
        else:
            # Generate the arrays associated to the sector Sz = fixed (ex. 0)
            list_Sz_fixed = generate_binary_arrays(n, dict_Sz[sz])
            
            # Convert vectors to integers
            vett_m_Sz = array_of_integers(list_Sz_fixed, n)
                    
            # Zip, sort by v_m_Sz0, and unzip
            paired = sorted(zip(vett_m_Sz, list_Sz_fixed))      
            v_m_Sz_sorted, list_Sz_fixed_sorted = zip(*paired)
            
            # Convert back to arrays if needed
            v_m_Sz_sorted = np.asarray(v_m_Sz_sorted, dtype=np.int32)
            list_Sz_fixed_sorted = np.asarray(list_Sz_fixed_sorted)
                    
            Block_H_Sz_sparse = Block_Hamiltonian_sparse_TR(v_m_Sz_sorted, list_Sz_fixed_sorted, n, Jpar, Jperp)
            eigenvalues_sparse, _ = eigsh(Block_H_Sz_sparse, k=1, which='SA')
            
            E_gs[j] += eigenvalues_sparse[0]
        j += 1
        
    return E_gs



autovalori = blocks_GS_TR(N, J_par, J_perp)


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False

# Discontinuity points (edges of steps)
x_steps, y_heights = magnetisation(autovalori)


x_plot = np.insert(x_steps, 0, 0)  
y_plot = y_heights / Nr  # NORMALIZE by magnetization max (= n°sites/2)
y_plot =  np.append(y_plot, y_plot[-1])  


# Plot of step function
fig_m, ax_m = plt.subplots(figsize=(6.2, 4.5))
ax_m.step(x_plot, y_plot, where='post', label=f'{Nr} rungs ')
ax_m.set_xlabel(r'$ h $', fontsize=15)
ax_m.set_ylabel(r'$ m $', fontsize=15)
ax_m.set_title(fr'Magnetization for ladder Hamiltonian ($J={J:.2f}$, $\theta={th:.2f}$)')
ax_m.legend(loc='upper left')
ax_m.grid(True)
plt.show()




# -----------------------------------------------------------------------------
# 5) Size scaling at fixed interaction parameters
# -----------------------------------------------------------------------------
"""
def multiple_plot(n_min, n_max, param_h, param_J, param_th ):
   
    # Interaction parameters
    param_J_par = param_J * math.cos(param_th)
    param_J_perp = param_J * math.sin(param_th)
    
    fig_m, ax_m = plt.subplots(figsize=(6.2, 4.5))
    
    for i in range(n_min, n_max+1, 2):
        Num = 2*i  # Total number of sites on the ladder
        autovalori = blocks_GS_TR(Num, param_J_par, param_J_perp)
        
        # Discontinuity points (edges of steps)
        x_steps, y_heights = magnetisation(autovalori)


        x_plot = np.insert(x_steps, 0, 0)  
        y_plot = y_heights / i   # NORMALIZE by magnetization max (= n°sites/2)           
        y_plot =  np.append(y_plot, y_plot[-1])  


        # Plot of step function
        
        ax_m.step(x_plot, y_plot, where='post', label=f'{i} rungs ')
    ax_m.set_xlabel(r'$ h $', fontsize=15)
    ax_m.set_ylabel(r'$ m $', fontsize=15)
    ax_m.set_title(fr'Magnetization for ladder Hamiltonian ($J={param_J:.2f}$, $\theta={param_th:.2f}$)')
    ax_m.legend(loc='upper left')
    ax_m.grid(True)
    plt.show()


justapposed_plots_TR = multiple_plot(2, 8, 0, 1, 0)
"""
