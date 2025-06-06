"""
Self built functions for the study of the Heisemberg S=1/2 model on a ladder

@author: david
"""

import numpy as np
import math
import itertools
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh


# -----------------------------------------------------------------------------
# FUNCTIONS FOR THE "CLASSIC" LADDER
# -----------------------------------------------------------------------------


# Generate arrays that mimic with 0 and 1 the states in the Sz basis
def generate_binary_arrays(n, k):
    num = math.comb(n, k)
    arrays = np.zeros((num, n), dtype=int)
    j = 0
    for ones_positions in itertools.combinations(range(n), k):
        array = np.zeros(n, dtype=int)
        for pos in ones_positions:
            array[pos] += 1
        arrays[j] = array
        j +=1
    return arrays



# Converts the binary arrays into corresponding integers m
def array_of_integers(lst, n):
    N_Sz = len(lst)
    v_m = np.zeros(N_Sz, dtype=int)
    j = 0
    for lst_j in lst:
        m = 0
        for i in range(n):
            m += lst_j[i] * (2 ** (i))
        v_m[j] += m
        j += 1
    return v_m



# Generate the block of the ladder hamiltonian with defined Sz (h=0 always)
# (it uses sparse matrix formalism to save memory)
def Block_Hamiltonian_sparse(v, lst, n, Jpar, Jperp):
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
        for i in range(0, n, 2):
            if lst_j[i] != lst_j[(i+1)]:
                vet = lst_j.copy()
                vet[i], vet[(i+1)] = vet[(i+1)], vet[i]
                m = sum(vet[k] * (2**k) for k in range(n))
                column = v_index[m]
                H_Sz[row, column] -= Jperp / 2

        # Diagonal J_perp
        vet = lst_j - 0.5
        h2 = sum(vet[i] * vet[(i+1)] for i in range(0, n, 2))
        H_Sz[row, row] += Jperp * h2

    return H_Sz.tocsr()



# For a given number of rungs, it loops on all the non-negative values of Sz  
# giving the lowest energy eigenvalue associated to each Sz-sector of 
# hamiltonian (with h= 0)
def blocks_GS(n, Jpar, Jperp):
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
            eigenvalues_sparse, _ = eigsh(Block_H_Sz_sparse, k=1, which='SA')
            
            E_gs[j] += eigenvalues_sparse[0]
        j += 1
        
    return E_gs



# Starting from the values obtained in Blocks_GS, it builds the eigenvalues
# of the complete hamiltonian (included the interaction with h) at fixed Sz and 
# gives the intervals of h in which the magnetization increases by steps.
# (Through intersections of E0 and E1-h, E1-h and E2-2h,.. up to E_SzMax)
def magnetisation(autov):
    n_inters = int((len(autov)-1)//2)
    h_arr = np.zeros(n_inters, dtype=np.float32)
    m_arr = np.zeros(n_inters, dtype=np.float32)
    x = 0
    for i in range(n_inters, len(autov)-1, 1):
        x = autov[i+1] - autov[i]
        h_arr[i-n_inters] += x
        m_arr[i-n_inters] += i-n_inters
    return h_arr, m_arr



# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    

# -----------------------------------------------------------------------------
# FUNCTIONS FOR THE TRIANGULAR LADDER
# -----------------------------------------------------------------------------


# Generate the block of the TRIANGULAR ladder hamiltonian with defined Sz (h=0)
# (it uses sparse matrix formalism to save memory)
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



# For a given number of rungs, it loops on all the non-negative values of Sz  
# giving the lowest energy eigenvalue associated to each Sz-sector of 
# hamiltonian (with h= 0)
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




