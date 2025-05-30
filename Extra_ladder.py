"""
Old / extra functions for the study of the Heisemberg S=1/2 model on a ladder

@author: david
"""
import numpy as np
import math
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
from itertools import combinations
"""
# Parameters
Nr = 5
N = 2 * Nr
h = 0
J = 1
theta = 0
J_par = J * math.cos(theta)
J_perp = J * math.sin(theta)

"""
"""
# -----------------------------------------------------------------------------
# 1) Build full matrix operator associated to the H (useless here) 
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
# -----------------------------------------------------------------------------
# 2) Use Lanczos algorithm to get an mxm tridiagonal real matrix from H
    # (first version without tolerance check)
    # Just for educational purpose: not efficient as built-in functions!
# -----------------------------------------------------------------------------


def lanczos(H, m, v0=None):
    n = H.shape[0]
    if v0 is None:
        v = np.random.randn(n) + 1j * np.random.randn(n)
    else:
        v = v0
    v = v / np.linalg.norm(v)

    V = np.zeros((n, m), dtype=np.complex128)
    alpha = np.zeros(m, dtype=np.float64)
    beta = np.zeros(m - 1, dtype=np.float64)

    V[:, 0] = v
    w = H @ v
    alpha[0] = np.real(np.vdot(v, w))
    w = w - alpha[0] * v

    for j in range(1, m):
        # Full reorthogonalization
        for i in range(j):
            proj = np.vdot(V[:, i], w)
            w -= proj * V[:, i]

        beta[j - 1] = np.linalg.norm(w)
        if beta[j - 1] < 1e-12:
            V = V[:, :j]
            alpha = alpha[:j]
            beta = beta[:j - 1]
            break

        v = w / beta[j - 1]
        V[:, j] = v
        w = H @ v

        # Reorthogonalize
        for i in range(j + 1):
            proj = np.vdot(V[:, i], w)
            w -= proj * V[:, i]

        alpha[j] = np.real(np.vdot(v, w))

    return alpha, beta, V
"""



"""
# Just for educational purpose: not efficient as built-in functions!

# Use this LANCZOS algorithm to get an mxm tridiagonal real matrix 
# from the block hamiltonian (fixed Sz). 
# Then diagonalize it to get the approximation of the lowest energy eigenvalue
# for that spin sector
# (Tolerance check included!)

def lanczos(H, m, max_iter=500, tol=1e-10):
    
    #Lanczos algorithm for approximating the lowest eigenvalue and eigenvector.
    #- H: Hamiltonian matrix (or sparse matrix)
    #- m: number of Lanczos iterations
    #- max_iter: max number of iterations to prevent infinite loop
    #- tol: tolerance for convergence
    
    n = H.shape[0]
    v = np.ones(n)
    v /= np.linalg.norm(v)  # Normalize the vector

    V = np.zeros((n, m), dtype=np.float64)  # Matrix to store orthonormal basis
    alpha = np.zeros(m, dtype=np.float64)  # Diagonal of the tridiagonal matrix
    beta = np.zeros(m - 1, dtype=np.float64)  # Off-diagonal of the tridiagonal matrix

    V[:, 0] = v  # Initial vector
    w = H @ v  # Apply H to the initial vector
    alpha[0] = np.dot(v, w)  # Compute the first diagonal element
    w -= alpha[0] * v  # Subtract off the diagonal part

    # Main Lanczos iteration loop
    for j in range(1, m):
        # Re-orthogonalize w against the previous vectors in V
        for i in range(j):
            proj = np.dot(V[:, i], w)
            w -= proj * V[:, i]

        beta[j - 1] = np.linalg.norm(w)  # Off-diagonal element
        if beta[j - 1] < tol:
            print(f'Lanczos converged early at iteration {j}')
            V = V[:, :j]  # Truncate V to the converged size
            alpha = alpha[:j]
            beta = beta[:j - 1]
            break

        v = w / beta[j - 1]  # Normalize the new vector
        V[:, j] = v  # Add it to the orthonormal basis
        w = H @ v  # Apply H to the new vector

        # Re-orthogonalize w again
        for i in range(j + 1):
            proj = np.dot(V[:, i], w)
            w -= proj * V[:, i]

        alpha[j] = np.dot(v, w)  # Diagonal element of the tridiagonal matrix

    # Now, solve for the eigenvalues of the tridiagonal matrix
    eigs, _ = eigh_tridiagonal(alpha, beta)

    return alpha, beta, V

# Example usage
from scipy.linalg import eigh_tridiagonal, eigh
alpha, beta, V = lanczos(Block_H_Sz, 100)  # Try with a higher number of iterations
eigs, _ = eigh_tridiagonal(alpha, beta)  # solve for the eigenvalues of the tridiagonal matrix
print('GS energy from Lanczos:', eigs[0])

"""


# ////////////////////////////////////////////////////////////////////////////


"""
# Block hamiltonian for a single chain
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

"""
# LADDER HAMILTONIAN (block of defined Sz)
# Once the sector is chosen, and v_m_sorted is built along with list_Sz_sorted,
# we can build the corresponding hamiltonian block! (Here with dense matrices)

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

# //////////////////////////////

Nr = 6
N = 2 * Nr
h = 0
J = 1
theta = 0
J_par = J * math.cos(theta)
J_perp = J * math.sin(theta)

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

from Funz_ladder import generate_binary_arrays, array_of_integers

Sz_fix = np.asarray([i for i in range(int(-N/2), int(N/2) +1)], dtype=int)
N_1 = np.asarray([i for i in range(N+1)], dtype=int)
dict_Sz = dict(zip(Sz_fix, N_1))

# Generate the arrays associated to the sector Sz = fixed (ex. 0)
list_Sz_fixed = generate_binary_arrays(N, dict_Sz[0])

vett_m_Sz = array_of_integers(list_Sz_fixed, N)

# Zip, sort by v_m_Sz0, and unzip
paired = sorted(zip(vett_m_Sz, list_Sz_fixed))      
v_m_Sz_sorted, list_Sz_fixed_sorted = zip(*paired)

# Convert back to arrays if needed
v_m_Sz_sorted = np.asarray(v_m_Sz_sorted, dtype=np.int32)
list_Sz_fixed_sorted = np.asarray(list_Sz_fixed_sorted)


sparse_H, sparse_row, sparse_col = Block_Hamiltonian(v_m_Sz_sorted, list_Sz_fixed_sorted, N, J_par,J_perp)


from scipy.linalg import eigh_tridiagonal


def sparse_matvec(H, row, col, v):
    size = max(row) + 1  # number of rows
    w = np.zeros(size)

    for h, r, c in zip(H, row, col):
        w[r] += h * v[c]
        
    return w




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


GS_lanczos = lanczos(sparse_H, sparse_row, sparse_col, 64)
print("GS energy from LANCZOS", GS_lanczos)


from Funz_ladder import Block_Hamiltonian_sparse
Block_H_Sz_sparse = Block_Hamiltonian_sparse(v_m_Sz_sorted, list_Sz_fixed_sorted, N, J_par, J_perp)
eigenvalues_sparse, _ = eigsh(Block_H_Sz_sparse, k=1, which='SA')
print("GS energy for SPARSE MATRIX", eigenvalues_sparse[0])

