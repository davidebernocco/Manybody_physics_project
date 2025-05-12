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


Nr = 4    # Number of ladder rungs
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

"""
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


# -----------------------------------------------------------------------------
# 3) Extract the m eigenvalues with a built-in efficient function
# -----------------------------------------------------------------------------


from scipy.linalg import eigh_tridiagonal, eigh
#alpha, beta, V = lanczos(ham, m=5)
#eigs, _ = eigh_tridiagonal(alpha, beta)
eigenvalues, eigenvectors = eigh(ham)
#print(eigs[0])
#print(eigenvalues[0])


def sz_total(n):
    Sz_total = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(n):
        Sz_total += Sz_list[i]

    return Sz_total



def compute_sz_sector(eigenv, Sz_tot):
       #Return the total Sz expectation value for each eigenvector.
    sectors = []
    for i in range(eigenv.shape[1]):
        psi = eigenv[:, i]
        sz_val = np.vdot(psi, Sz_tot @ psi).real  # expectation value
        sectors.append(round(sz_val, 5))  # rounding helps reduce numerical noise
    return sectors




# -----------------------------------------------------------------------------
# 4) Obtain the "magnetization vs h" plot, starting from energy eigenvalues
# -----------------------------------------------------------------------------

autovalori = np.asarray([-1,-0.5,3,8])

def magnetisation(autov):
    n_inters = len(autov)-1
    h_arr = np.zeros(n_inters, dtype=np.float32)
    m_arr = np.zeros(n_inters, dtype=np.float32)
    x = 0
    for i in  range(n_inters):
        x = autov[i+1] - autov[i]
        h_arr[i] += x
        m_arr[i] += i
    return h_arr, m_arr



import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False

# Discontinuity points (edges of steps)
x_steps, y_heights = magnetisation(autovalori)


x_plot = np.insert(x_steps, 0, 0)                
y_plot = np.insert(y_heights, 0, 0)  

# Plot of step function
fig_m, ax_m = plt.subplots(figsize=(6.2, 4.5))
ax_m.step(x_plot, y_plot, where='pre')
ax_m.set_xlabel(r'$ h $', fontsize=15)
ax_m.set_ylabel(r'$ m $', fontsize=15)
ax_m.grid(True)
plt.show()



# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    
import itertools
import numpy as np
import math

J=1

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

Nr = 2
N = 2*Nr  

Sz_fix = np.asarray([i for i in range(int(-N/2), int(N/2) +1)], dtype=int)
N_1 = np.asarray([i for i in range(N+1)], dtype=int)
dict_Sz = dict(zip(Sz_fix, N_1))

# If I want to generate the arrays associated to the sector Sz = fixed (ex. 0)
list_Sz_fixed = np.asarray(generate_binary_arrays(N, dict_Sz[0]))


def array_of_integers(lst):
    N_Sz = len(lst)
    v_m = np.zeros(N_Sz, dtype=int)
    j = 0
    for lst_j in lst:
        m = 0
        for i in range(N):
            m += lst_j[i] * (2 ** (i))
        v_m[j] += m
        j += 1
    return v_m

vett_m_Sz = array_of_integers(list_Sz_fixed)



# Zip, sort by v_m_Sz0, and unzip
paired = sorted(zip(vett_m_Sz, list_Sz_fixed))      
v_m_Sz_sorted, list_Sz_fixed_sorted = zip(*paired)

# Convert back to arrays if needed
v_m_Sz_sorted = np.asarray(list(v_m_Sz_sorted), dtype=np.int32)
list_Sz_fixed_sorted = np.asarray(list(list_Sz_fixed_sorted))





# Once the sector is chosen, and v_m_sorted is built along with list_Sz_sorted,
# we can build the corresponding hamiltonian block!

def Block_Hamiltonian(v, lst):
    d = len(lst)
    H_Sz = np.zeros((d,d), dtype=np.float32)
    
    row = 0
    column = 0
    for lst_j in lst:
        
        # Off-diagonal part of H_Sz
        for i in range(N):
            if lst_j[i] != lst_j[(i+1)%N]:
                lst_j[i], lst_j[(i+1)%N] = lst_j[(i+1)%N], lst_j[i]
                m = 0
                for i in range(N):
                    m += lst_j[i] * (2 ** (i))
                column = np.where(v == m)[0][0]
            H_Sz[row, column] -= J/2
        
        
        # Diagonal part of H_Sz
        lst_j = lst_j - 0.5
        h1 = 0
        for i in range(N):
            h1 += lst_j[i] * lst_j[(i+1)%N]
        H_Sz[row, row] += h1
           
        row += 1
    
    return H_Sz


Block_H_Sz = Block_Hamiltonian(v_m_Sz_sorted, list_Sz_fixed_sorted)






