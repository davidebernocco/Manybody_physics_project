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
from Funz_ladder import generate_binary_arrays, array_of_integers
from Funz_ladder import Block_Hamiltonian_sparse, Block_Hamiltonian
from Funz_ladder import lanczos, multiple_plot_Lanczos, multiple_plot_comparison

import matplotlib
matplotlib.rcParams['text.usetex'] = False

Nr = 6
N = 2 * Nr
h = 0
J = 1
theta = math.pi/2
J_par = J * math.cos(theta)
J_perp = J * math.sin(theta)



# -----------------------------------------------------------------------------
# 1) COMPARE the results of self-made LANCZOS algorithm WITH the BUILT-IN exact 
#    diagonalization algorithm for the GS energy.
# -----------------------------------------------------------------------------


# Arrays of all the allowed Sz values, and corresponding number of 'up' sites
Sz_fix_L = np.asarray([i for i in range(int(-N/2), int(N/2) +1)], dtype=int)
N_1_L = np.asarray([i for i in range(N+1)], dtype=int)
dict_Sz_L = dict(zip(Sz_fix_L, N_1_L))

# Generate the arrays associated to the sector Sz = fixed (ex. 0)
list_Sz_fixed_L = generate_binary_arrays(N, dict_Sz_L[0])

# Generate the corresponding arrays of label for the system states referred to 
 # a specific spin sector.
vett_m_Sz_L = array_of_integers(list_Sz_fixed_L, N)

# Zip, sort by v_m_Sz0, and unzip
paired_L = sorted(zip(vett_m_Sz_L, list_Sz_fixed_L))      
v_m_Sz_sorted_L, list_Sz_fixed_sorted_L = zip(*paired_L)

# Convert back to arrays if needed
v_m_Sz_sorted_L = np.asarray(v_m_Sz_sorted_L, dtype=np.int32)
list_Sz_fixed_sorted_L = np.asarray(list_Sz_fixed_sorted_L)

# Define sparse matrix
sparse_H, sparse_row, sparse_col = Block_Hamiltonian(v_m_Sz_sorted_L, list_Sz_fixed_sorted_L, N, J_par,J_perp)


start_time1 = time.time()
GS_lanczos = lanczos(sparse_H, sparse_row, sparse_col, 15)
print("GS energy from LANCZOS", GS_lanczos)
end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
print("Elapsed time1:", elapsed_time1)


start_time2 = time.time()
Block_H_Sz_sparse = Block_Hamiltonian_sparse(v_m_Sz_sorted_L, list_Sz_fixed_sorted_L, N, J_par, J_perp)
eigenvalues_sparse, _ = eigsh(Block_H_Sz_sparse, k=1, which='SA')
print("GS energy for SPARSE MATRIX", np.float64(eigenvalues_sparse[0]))
end_time2 = time.time()
elapsed_time2 = end_time2 - start_time2
print("Elapsed time2:", elapsed_time2)




# -----------------------------------------------------------------------------
# 2) ESTIMATE of Self-made Lanczos GS ENERGY value as function of LANCZOS STEPS
# -----------------------------------------------------------------------------

justapposed_plots_Lanczos = multiple_plot_Lanczos(1, 26, 5, J_par, J_perp)




# -----------------------------------------------------------------------------
#   3) COMPARISON between COMPUTATIONAL TIMES of "myLanczos" and built-in function
# -----------------------------------------------------------------------------

plot_comparison = multiple_plot_comparison(4, 7, J_par, J_perp)

