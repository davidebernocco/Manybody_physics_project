"""
Old / extra functions for the study of the Heisemberg S=1/2 model on a ladder

@author: david
"""


import numpy as np
import math
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
from itertools import combinations

# Parameters
Nr = 5
N = 2 * Nr
h = 0
J = 1
theta = 0
J_par = J * math.cos(theta)
J_perp = J * math.sin(theta)


"""
# -----------------------------------------------------------------------------
# 1) Build full spin matrix operator associated to the H 
   # (useless for our current approach)  
# -----------------------------------------------------------------------------

# Define the Pauli spin matrices plus the 2x2 identity matrix
Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


# Precompute single-particle operators defined on the full system space,
# returning a list of operators acting on each site of the ladder.
def build_site_operators(op, N):
    ops = []
    for i in range(N):
        op_list = [I2] * N
        op_list[i] = op
        full_op = op_list[0]
        for o in op_list[1:]:
            full_op = np.kron(full_op, o)
        ops.append(full_op)
    return ops

# Precompute all spin operators
Sx_list = build_site_operators(Sx, N)
Sy_list = build_site_operators(Sy, N)
Sz_list = build_site_operators(Sz, N)


# Build the Hamiltonian matrix
def H_matrix(N): 
    
    H = np.zeros((2**N, 2**N), dtype=complex)
    
    for i in range(N):
        # Coupling with the magnetic field
        H += -h * Sz_list[i]    

        for op_list in [Sx_list, Sy_list, Sz_list]:
            
            # For even i
            if i % 2 == 0:
                # Rung interaction
                j1 = (i + 1) % N
                H += J_perp * op_list[i] @ op_list[j1]

                # Leg interaction
                j2 = (i + 2) % N
                H += J_par * op_list[i] @ op_list[j2]
                
            # For odd i only leg interaction
            else:
                j2 = (i + 2) % N
                H += J_par * op_list[i] @ op_list[j2]
                
    return H

# Construct Hamiltonian
ham = H_matrix(N)

# Check Hermiticity
print("Hamiltonian shape:", ham.shape)
print("Hermitian check (H - H†):", np.linalg.norm(ham - ham.conj().T))

"""


# ///////////////////////////////////////////////////////////////////////////



"""
# Just for educational purpose: not efficient as built-in functions!

# Use this LANCZOS algorithm to get an mxm tridiagonal real matrix 
# from the block hamiltonian (fixed Sz). 
# Then diagonalize it to get the approximation of the lowest energy eigenvalue
# for that spin sector


def lanczos(sp_H, sp_row, sp_col, m):
    n = sp_row[-1] + 1
    v = np.random.randn(n)
    v /= np.linalg.norm(v)

    V = np.zeros((n, m), dtype=np.float64)
    alpha = np.zeros(m, dtype=np.float64)
    beta = np.zeros(m - 1, dtype=np.float64)

    V[:, 0] = v
    w = sparse_matvec(sp_H, sp_row, sp_col, v)
    alpha[0] = np.dot(v, w)
    w -= alpha[0] * v  # First orthogonalization step

    for j in range(1, m):
        beta[j - 1] = np.linalg.norm(w)

        v = w / beta[j - 1]
        V[:, j] = v

        w = sparse_matvec(sp_H, sp_row, sp_col, v)
        w -= beta[j - 1] * V[:, j - 1]  # Subtract previous component
        alpha[j] = np.dot(v, w)
        w -= alpha[j] * v          # Subtract current component

    eigs, _ = eigh_tridiagonal(alpha, beta)
    gs_energy = eigs[0]

    return gs_energy


"""



# ////////////////////////////////////////////////////////////////////////////



# Block hamiltonian for a single chain
"""
def Block_Hamiltonian(v, lst):
    d = len(lst)
    H_Sz = np.zeros((d,d), dtype=np.float32)
    
    row = 0
    column = 0
    for lst_j in lst:
        
        # Off-diagonal part of H_Sz
        for i in range(N):
            if lst_j[i] != lst_j[(i+1)%N]:
                vet = lst_j.copy()
                vet[i], vet[(i+1)%N] = vet[(i+1)%N], vet[i]
                m = 0
                for k in range(N):
                    m += vet[k] * (2 ** (k))
                column = np.where(v == m)[0][0]
                H_Sz[row, column] -= J/2
        del vet
        
        # Diagonal part of H_Sz
        vet = lst_j.copy() - 0.5
        h1 = 0
        for i in range(N):
            h1 += vet[i] * vet[(i+1)%N]
        H_Sz[row, row] += J *  h1
           
        row += 1
    
    return H_Sz


Block_H_Sz = Block_Hamiltonian(v_m_Sz_sorted, list_Sz_fixed_sorted)
"""



# ////////////////////////////////////////////////////////////////////////////



# LADDER HAMILTONIAN (block of defined Sz)
# Once the sector is chosen, and v_m_sorted is built along with list_Sz_sorted,
# we can build the corresponding hamiltonian block! (Here with dense matrices)
# (First function built: less efficient compared to the sparse matrix approach)

"""
def Block_Hamiltonian(v, lst):
    d = len(lst)
    H_Sz = np.zeros((d,d), dtype=np.float64)
    
    row = 0
    column = 0
    for lst_j in lst:
        
        # Off-diagonal part of H_Sz, J_par
        for i in range(N):
            if lst_j[i] != lst_j[(i+2)%N]:
                vet = lst_j.copy()
                vet[i], vet[(i+2)%N] = vet[(i+2)%N], vet[i]
                m = 0
                for k in range(N):
                    m += vet[k] * (2 ** (k))
                column = np.where(v == m)[0][0]
                H_Sz[row, column] -= J_par/2
        del vet
        
        # Diagonal part of H_Sz, J_par
        vet = lst_j.copy() - 0.5
        h1 = 0
        for i in range(N):
            h1 += vet[i] * vet[(i+2)%N]
        H_Sz[row, row] += J_par *  h1
        
        # Off-diagonal part of H_Sz, J_perp
        for i in range(0,N,2):
            if lst_j[i] != lst_j[(i+1)%N]:
                vet = lst_j.copy()
                vet[i], vet[(i+1)%N] = vet[(i+1)%N], vet[i]
                m = 0
                for k in range(N):
                    m += vet[k] * (2 ** (k))
                column = np.where(v == m)[0][0]
                H_Sz[row, column] -= J_perp/2
        del vet
        
        # Diagonal part of H_Sz, J_perp
        vet = lst_j.copy() - 0.5
        h2 = 0
        for i in range(0,N,2):
            h2 += vet[i] * vet[(i+1)%N]
        H_Sz[row, row] += J_perp *  h2
           
        row += 1
    
    return H_Sz



start_time1 = time.time()
from scipy.linalg import eigh
Block_H_Sz = Block_Hamiltonian(v_m_Sz_sorted, list_Sz_fixed_sorted)
eigenvalues, eigenvectors = eigh(Block_H_Sz)
print('GS energy',eigenvalues[0])
end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
print("Elapsed time1:", elapsed_time1)
"""

# -----------------------------------------------------------------------------
# 2)  Ladder Hamiltonian as EFFECTIVE HAMILTONIAN of a bosonic particle (triplet)
#     hopping into void neighbours (singlets)
# -----------------------------------------------------------------------------
"""

Sz_fix = np.asarray([i for i in range(int(-N/2), int(N/2) +1)], dtype=int)
N_1 = np.asarray([i for i in range(N+1)], dtype=int)
dict_Sz = dict(zip(Sz_fix, N_1))

# Generate the arrays associated to the sector Sz = fixed
list_Sz_fixed = generate_binary_arrays(N, dict_Sz[1])

vett_m_Sz = array_of_integers(list_Sz_fixed, N)

# Zip, sort by v_m_Sz0, and unzip
paired = sorted(zip(vett_m_Sz, list_Sz_fixed))      
v_m_Sz_sorted, list_Sz_fixed_sorted = zip(*paired)

# Convert back to arrays if needed
v_m_Sz_sorted = np.asarray(v_m_Sz_sorted, dtype=np.int32)
list_Sz_fixed_sorted = np.asarray(list_Sz_fixed_sorted)



# Takes all the binary configurations associated to a specific Sz sector of the Ham
# and returns a dictionary of the configurations(through their m) with n_tr 
# number of triplets (dimers (1,1) or (0,0)) among the singlets (dimers (1,0) or (0,1))
def triplet_position(list_Sz, list_m, n_tr):
    tr_dict = {}
    d = len(list_Sz_fixed_sorted[0])
    for k in range(len(list_Sz)):
        lst_half = list_Sz[k] - 0.5
        s = 0
        pos_tr = []
        for i in range(0, d-1, 2):
            val = abs(lst_half[i] + lst_half[i+1])
            s += val
            if val == 1:
                pos_tr.append(i)
        if s == n_tr:
            tr_dict[list_m[k]] = pos_tr
    return tr_dict
            
            
# Configurations with n_tr triplets (tipically 1)      
vet_tr = triplet_position(list_Sz_fixed_sorted, v_m_Sz_sorted, 1)

keys_list = list(vet_tr.keys())
values_list = list(vet_tr.values())

picked_values = keys_list

# Get indices of these values in v
indices = [i for i, val in enumerate(v_m_Sz_sorted) if val in picked_values]

# Select corresponding rows in A
selected_rows = list_Sz_fixed_sorted[indices]



# Takes all the configurations with fixed n_tr number of triplets and returns 
# the m of the configurations with the n_tr triplets hopped with respect to the
# starting arrays because of the presence of the J_par interaction.
# Ex:
#     1==0==0==0==0==0      
# =>  |  |  |  |  |  |     equiv to    T==S==S==S==S==S
#     1==1==1==1==1==1
#
# =>  Ladder_Hamiltonian   equiv to    Eff_Hamiltonian that makes Triplets hop
#
def Hopping_check(lst, dictionary):
    n = len(lst[0])
    keys_list = list(dictionary.keys())
    values_list = list(dictionary.values())
    
    my_dict = {}
    my_dict['initial m with 1 triplet'] = keys_list
    my_dict['triplet position'] = values_list
    my_dict['m after hopping'] = []
    my_dict['new triplet position'] = []

    for row, lst_j in enumerate(lst):
        # Off-diagonal J_par
        lst_conf = []
        lst_m = []
        lst_pos = []
        for i in range(n):
            if lst_j[i] != lst_j[(i+2)%n]:
                vet = lst_j.copy()
                vet[i], vet[(i+2)%n] = vet[(i+2)%n], vet[i]
                lst_conf.append(vet)
                m = sum(vet[k] * (2**k) for k in range(n))
                lst_m.append(m)
        my_dict['m after hopping'].append(lst_m)
        moment_dict = triplet_position(lst_conf, lst_m, 1)
        lst_pos = [item for sublist in moment_dict.values() for item in sublist]
        my_dict['new triplet position'].append(lst_pos)

    return my_dict


Ham_eff = Hopping_check(selected_rows, vet_tr)
print(Ham_eff)

"""



# -----------------------------------------------------------------------------
# 2) MAGNETIZATION vs h, COMPARISON between builtin function and self made Lanczos
# -----------------------------------------------------------------------------
"""

import math

import numpy as np
from scipy.linalg import eigh_tridiagonal
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = False


from Funz_ladder import generate_binary_arrays, array_of_integers
from Funz_ladder import Block_Hamiltonian_sparse, blocks_GS, magnetisation


# ////////////////////////////////////////////////////////////////////////////

# BUILT-IN FUNCTION to calculate lowest-lying eigenstates for each Sz sector


start_time11 = time.time()

lowest_eigs = blocks_GS(N, J_par, J_perp)

# Discontinuity points (edges of steps)
x_steps, y_heights = magnetisation(lowest_eigs)
y_heights = np.append(y_heights, N)

# Plot edges
x_plot = np.insert(x_steps, 0, 0) 
x_plot = np.append(x_plot, 5*x_plot[-1]/4) 
y_plot = y_heights / Nr            
y_plot =  np.append(y_plot, y_plot[-1])  


# Plot of step function
fig_m, ax_m = plt.subplots(figsize=(6.2, 4.5))
ax_m.step(x_plot, y_plot, where='post', label=f'{Nr} rungs ')
ax_m.set_xlabel(r'$ h $', fontsize=15)
ax_m.set_ylabel(r'$ m $', fontsize=15)
ax_m.set_title(fr'Magnetization for ladder Hamiltonian ($J={J:.2f}$, $\theta={theta:.2f}$)')
ax_m.legend(loc='upper left')
ax_m.grid(True)
plt.show()

end_time11 = time.time()
elapsed_time11 = end_time11 - start_time11
print("Elapsed time11:", elapsed_time11)


# /////////////////////////////////////////////////////////////////////////////

# SELF-MADE LANCZOS FUNCTION to calculate lowest-lying eigenstates for each Sz sector


def lanczos_sp(H_sparse, m):
    
    n = H_sparse.shape[0]
    v = np.random.randn(n)
    v /= np.linalg.norm(v)

    V = np.zeros((n, m), dtype=np.float64)
    alpha = np.zeros(m, dtype=np.float64)
    beta = np.zeros(m - 1, dtype=np.float64)

    V[:, 0] = v
    w = H_sparse @ v  # Use sparse matrix-vector product
    alpha[0] = np.dot(v, w)
    w -= alpha[0] * v

    for j in range(1, m):
        beta[j - 1] = np.linalg.norm(w)

        v = w / beta[j - 1]
        V[:, j] = v

        w = H_sparse @ v
        w -= beta[j - 1] * V[:, j - 1]
        alpha[j] = np.dot(v, w)
        w -= alpha[j] * v

    eigs, _ = eigh_tridiagonal(alpha, beta)
    gs_energy = eigs[0]

    return gs_energy



def blocks_GS_lan(n, Jpar, Jperp):
    Sz_fix = np.asarray([i for i in range(int(-n/2), int(n/2) +1)], dtype=int)
    n_1 = np.asarray([i for i in range(n+1)], dtype=int)
    dict_Sz = dict(zip(Sz_fix, n_1))
    d = len(Sz_fix)
    E_gs = np.zeros(d, dtype=np.float32)
    
    j = 0
    for sz in Sz_fix:
        if sz == int(-n/2) or sz == int(n/2):
            # Manually add the energy of the 1x1 sectors of |Sz| max
            E_gs[j] += np.sign(sz) * (Jpar*n/4 + Jperp*n/8)
        else:
            # Generate the arrays associated to the sector Sz = fixed
            list_Sz_fixed = generate_binary_arrays(n, dict_Sz[sz])
            
            # Convert vectors to integers
            vett_m_Sz = array_of_integers(list_Sz_fixed, n)
                    
            # Zip, sort by v_m_Sz0, and unzip
            paired = sorted(zip(vett_m_Sz, list_Sz_fixed))      
            v_m_Sz_sorted, list_Sz_fixed_sorted = zip(*paired)
            
            # Convert back to arrays if needed
            v_m_Sz_sorted = np.asarray(v_m_Sz_sorted, dtype=np.int32)
            list_Sz_fixed_sorted = np.asarray(list_Sz_fixed_sorted)
                    
            Block_H_Sz_sparse = Block_Hamiltonian_sparse(v_m_Sz_sorted, list_Sz_fixed_sorted, n, Jpar, Jperp)
            eigenvalues_sparse = lanczos_sp(Block_H_Sz_sparse, 15)
            
            E_gs[j] += eigenvalues_sparse
        j += 1
        
    return E_gs



start_time22 = time.time()

lowest_eigs = blocks_GS_lan(N, J_par, J_perp)

# Discontinuity points (edges of steps)
x_steps, y_heights = magnetisation(lowest_eigs)
y_heights = np.append(y_heights, N)

# Plot edges
x_plot = np.insert(x_steps, 0, 0) 
x_plot = np.append(x_plot, 5*x_plot[-1]/4) 
y_plot = y_heights / Nr            
y_plot =  np.append(y_plot, y_plot[-1])  


# Plot of step function
fig_m, ax_m = plt.subplots(figsize=(6.2, 4.5))
ax_m.step(x_plot, y_plot, where='post', label=f'{Nr} rungs ')
ax_m.set_xlabel(r'$ h $', fontsize=15)
ax_m.set_ylabel(r'$ m $', fontsize=15)
ax_m.set_title(fr'Magnetization for ladder Hamiltonian ($J={J:.2f}$, $\theta={theta:.2f}$) Lanczos')
ax_m.legend(loc='upper left')
ax_m.grid(True)
plt.show()

end_time22 = time.time()
elapsed_time22 = end_time22 - start_time22
print("Elapsed time22:", elapsed_time22)
"""




