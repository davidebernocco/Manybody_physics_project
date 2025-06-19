"""
Lanczos algorithm for the study of the Heisemberg S=1/2 model on a ladder
(sparse matrix formalism implemented)

@author: david
"""

# Ex) Application of the Lanczos algorithm for the diagonalization of a 
#     block of the hamiltonian matrix of fixed Sz.


import numpy as np
import math
import time
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh_tridiagonal
from Funz_ladder import generate_binary_arrays, array_of_integers

Nr = 4
N = 2 * Nr
h = 0
J = 1
theta = math.pi/2
J_par = J * math.cos(theta)
J_perp = J * math.sin(theta)


# LADDER HAMILTONIAN with self made SPARSE MATRIX approach, used along with 
# self made LANCZOS algorithm to provide GS energy.
def Block_Hamiltonian(v, lst, n, Jpar, Jperp):
    
    H_dict = {}  # key: (row, col), value: matrix entry
    v_index = {val: idx for idx, val in enumerate(v)}  # dictionary for fast lookup
    vet_H = []
    vet_row = []
    vet_col = []
    
    row = 0
    column = 0
    for lst_j in lst:
        
        # Off-diagonal part of H_Sz, Jpar
        for i in range(n):
            if lst_j[i] != lst_j[(i+2)%n]:
                vet = lst_j.copy()
                vet[i], vet[(i+2)%n] = vet[(i+2)%n], vet[i]
                m = 0
                for k in range(n):
                    m += vet[k] * (2 ** (k))
                column = v_index[m]
                H_dict[(row, column)] = H_dict.get((row, column), 0.0) - Jpar / 2
                
        del vet
        
        # Diagonal part of H_Sz, Jpar
        vet = lst_j.copy() - 0.5
        h1 = 0
        for i in range(n):
            h1 += vet[i] * vet[(i+2)%n]
        H_dict[(row, row)] = H_dict.get((row, row), 0.0) + Jpar * h1
        
        # Off-diagonal part of H_Sz, Jperp
        for i in range(0,n,2):
            if lst_j[i] != lst_j[(i+1)%n]:
                vet = lst_j.copy()
                vet[i], vet[(i+1)%n] = vet[(i+1)%n], vet[i]
                m = 0
                for k in range(n):
                    m += vet[k] * (2 ** (k))
                column = v_index[m]
                H_dict[(row, column)] = H_dict.get((row, column), 0.0) - Jperp / 2

        del vet
        
        # Diagonal part of H_Sz, Jperp
        vet = lst_j.copy() - 0.5
        h2 = 0
        for i in range(0,n,2):
            h2 += vet[i] * vet[(i+1)%n]
        H_dict[(row, row)] = H_dict.get((row, row), 0.0) + Jperp * h2


        row += 1
        
    for (r, c), val in H_dict.items():
        if abs(val) > 0.0:
            vet_row.append(r)
            vet_col.append(c)
            vet_H.append(val)
    
    return vet_H, vet_row, vet_col


"""
# Arrays of all the allowed Sz values, and corresponding number of 'up' sites
Sz_fix = np.asarray([i for i in range(int(-N/2), int(N/2) +1)], dtype=int)
N_1 = np.asarray([i for i in range(N+1)], dtype=int)
dict_Sz = dict(zip(Sz_fix, N_1))

# Generate the arrays associated to the sector Sz = fixed (ex. 0)
list_Sz_fixed = generate_binary_arrays(N, dict_Sz[0])

# Generate the corresponding arrays of label for the system states referred to 
 # a specific spin sector.
vett_m_Sz = array_of_integers(list_Sz_fixed, N)

# Zip, sort by v_m_Sz0, and unzip
paired = sorted(zip(vett_m_Sz, list_Sz_fixed))      
v_m_Sz_sorted, list_Sz_fixed_sorted = zip(*paired)

# Convert back to arrays if needed
v_m_Sz_sorted = np.asarray(v_m_Sz_sorted, dtype=np.int32)
list_Sz_fixed_sorted = np.asarray(list_Sz_fixed_sorted)


sparse_H, sparse_row, sparse_col = Block_Hamiltonian(v_m_Sz_sorted, list_Sz_fixed_sorted, N, J_par,J_perp)
"""


# Application of a sparse matrix to a vector
def sparse_matvec(H, row, col, v):
    size = max(row) + 1  # number of rows
    w = np.zeros(size)

    for h, r, c in zip(H, row, col):
        w[r] += h * v[c]
        
    return w



# LANCZOS algorithm for the lowest energy value of a specific Sz sector 
# of a hamiltonian sparse matrix
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
# Compare the results with the built-in exact diagonalization algorithm
start_time1 = time.time()
GS_lanczos = lanczos(sparse_H, sparse_row, sparse_col, 15)
print("GS energy from LANCZOS", GS_lanczos)
end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
print("Elapsed time1:", elapsed_time1)


start_time2 = time.time()
from Funz_ladder import Block_Hamiltonian_sparse
Block_H_Sz_sparse = Block_Hamiltonian_sparse(v_m_Sz_sorted, list_Sz_fixed_sorted, N, J_par, J_perp)
eigenvalues_sparse, _ = eigsh(Block_H_Sz_sparse, k=1, which='SA')
print("GS energy for SPARSE MATRIX", np.float64(eigenvalues_sparse[0]))
end_time2 = time.time()
elapsed_time2 = end_time2 - start_time2
print("Elapsed time2:", elapsed_time2)

"""

"""
def lanczos_efficiency(m_min, m_max, d_m):
    arr_m = np.arange(m_min, m_max, d_m)
    arr_eig = np.zeros(len(arr_m), dtype=np.float64)
    arr_time= np.zeros(len(arr_m), dtype=np.float64)
    i = 0
    for val in arr_m:
        
        start_time = time.time()
        GS = lanczos(sparse_H, sparse_row, sparse_col, val)
        arr_eig[i] = GS
        end_time = time.time()
        elapsed_time = end_time - start_time
        arr_time[i] = elapsed_time
        i+=1
        
    return arr_m, arr_eig, arr_time


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False

v_x, v_y, v_t = lanczos_efficiency(1, 7, 1)
fig_m, ax_m = plt.subplots(figsize=(6.2, 4.5))
ax_m.plot(v_x, v_y, label=' rungs ')
ax_m.set_xlabel(' #m steps ', fontsize=15)
ax_m.set_ylabel(' E ', fontsize=15)
ax_m.set_title(r'GS energy for different number of Lanczos steps ')
ax_m.legend(loc='upper right')
ax_m.grid(True)
plt.show()

"""


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False


def lanczos_efficiency(n, m_min, m_max, d_m):
    # Arrays of all the allowed Sz values, and corresponding number of 'up' sites
    sz_fix = np.asarray([i for i in range(int(-n/2), int(n/2) +1)], dtype=int)
    n_1 = np.asarray([i for i in range(n+1)], dtype=int)
    dict_sz = dict(zip(sz_fix, n_1))

    # Generate the arrays associated to the sector Sz = fixed (ex. 0)
    list_sz_fixed = generate_binary_arrays(n, dict_sz[0])

    # Generate the corresponding arrays of label for the system states referred to 
     # a specific spin sector.
    arr_m_sz = array_of_integers(list_sz_fixed, n)

    # Zip, sort by v_m_Sz0, and unzip
    paired = sorted(zip(arr_m_sz, list_sz_fixed))      
    m_sz_sorted, list_sz_fixed_sorted = zip(*paired)

    # Convert back to arrays if needed
    m_sz_sorted = np.asarray(m_sz_sorted, dtype=np.int32)
    list_sz_fixed_sorted = np.asarray(list_sz_fixed_sorted)

    sp_H, sp_row, sp_col = Block_Hamiltonian(m_sz_sorted, list_sz_fixed_sorted, n, J_par,J_perp)
    
    
    arr_m = np.arange(m_min, m_max, d_m)
    arr_eig = np.zeros(len(arr_m), dtype=np.float64)
    arr_time= np.zeros(len(arr_m), dtype=np.float64)
    i = 0
    for val in arr_m:
        
        start_time = time.time()
        GS = lanczos(sp_H, sp_row, sp_col, val)
        arr_eig[i] = GS
        end_time = time.time()
        elapsed_time = end_time - start_time
        arr_time[i] = elapsed_time
        i+=1
        
    return arr_m, arr_eig, arr_time


def multiple_plot_Lanczos(m_min, m_max, d_m ):
    
    fig_m, ax_m = plt.subplots(figsize=(6.2, 4.5))
    
    for i in range(4, 13, 2):
        Num = 2*i  # Total number of sites on the ladder
        v_x, v_y, v_t = lanczos_efficiency(Num, m_min, m_max, d_m)
       

        # Plot of step function    
        ax_m.plot(v_x, v_y, label=f'{i} rungs ')
    ax_m.set_xlabel(r' n° of m steps', fontsize=15)
    ax_m.set_ylabel(r'$ E_0 $', fontsize=15)
    ax_m.set_title(r'GS energy for different number of Lanczos steps ')
    ax_m.legend(loc='upper right')
    ax_m.grid(True)
    plt.show()


justapposed_plots_Lanczos = multiple_plot_Lanczos(1, 36, 5)

