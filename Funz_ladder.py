"""
Self built functions for the study of the Heisemberg S=1/2 model on a ladder

@author: david
"""
import numpy as np
import math
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
from itertools import combinations

# Parameters
Nr = 6
N = 2 * Nr
h = 0.5
J = 1.0
theta = 0.2
J_par = J * math.cos(theta)
J_perp = J * math.sin(theta)



"""
# Hamiltonian for a single chain
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


# //////////////////////////////


"""
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

