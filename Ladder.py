"""
Heisemberg S=1/2 model on a ladder.
(PBC on the two parallel legs "==")

    0==0==0==0==0      "==" : J_parallel = J*cos(theta)
    |  |  |  |  |      " | " : J_perpendicular = J*sin(theta)
    0==0==0==0==0


The Hilbert space associated to the full system is (C^2)tensor(C^2)...(C^2)

@author: david
"""

import numpy as np
import math
import time


Nr = 5    # Number of ladder rungs
N = 2*Nr  # Total number of sites on the ladder

# Interaction parameters
h = 0
J = 1                 
th = 0
J_par = J*math.cos(th)
J_perp = J*math.sin(th)



# -----------------------------------------------------------------------------
# 1) Build the matrix associated to the Hamiltonian 
# -----------------------------------------------------------------------------

# Define the Pauli spin matrices plus the 2x2 identity matrix
Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)



start_time1 = time.time()

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
#print("Hamiltonian shape:", ham.shape)
#print("Hermitian check (H - H†):", np.linalg.norm(ham - ham.conj().T))

end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
print("Elapsed time:", elapsed_time1)



# -----------------------------------------------------------------------------
# 2) Use the Lanczos algorithm to get an mxm tridiagonal real matrix from H
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




# -----------------------------------------------------------------------------
# 3) Extract the m eigenvalues with a built-in efficient function
# -----------------------------------------------------------------------------


from scipy.linalg import eigh_tridiagonal
alpha, beta, V = lanczos(ham, m=3)
eigs, _ = eigh_tridiagonal(alpha, beta)




