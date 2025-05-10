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

# 🧠 Define the S^z = 0 basis (bitstrings with N/2 1s and N/2 0s)
def get_sz0_basis(N):
    """Return list of basis states with total Sz = 0."""
    half = N // 2
    basis = []
    for bits in combinations(range(N), half):
        state = 0
        for b in bits:
            state |= (1 << b)
        basis.append(state)
    return basis

basis = get_sz0_basis(N)
dim = len(basis)
index_map = {state: i for i, state in enumerate(basis)}  # state -> index in Sz=0 sector

# Spin operators using bit logic
def spin_z(state, site):
    return 0.5 if (state >> site) & 1 else -0.5

def flip_spin(state, i):
    return state ^ (1 << i)

def build_sz0_hamiltonian(basis, index_map, N, h, J_par, J_perp):
    H = lil_matrix((len(basis), len(basis)), dtype=complex)

    for idx, state in enumerate(basis):
        for i in range(N):
            # Magnetic field term
            H[idx, idx] += -h * spin_z(state, i)

            # Loop over XX, YY, ZZ terms
            for coupling, name in [(J_perp, 'rung') if i % 2 == 0 else (J_par, 'leg')]:
                j = (i + 1 if name == 'rung' else i + 2) % N

                si = (state >> i) & 1
                sj = (state >> j) & 1

                # Sz Sz term
                H[idx, idx] += coupling * (0.25 if si == sj else -0.25)

                # S+ S- and S- S+ terms (flip both)
                if si != sj:
                    flipped = state ^ (1 << i) ^ (1 << j)
                    if flipped in index_map:
                        jdx = index_map[flipped]
                        H[idx, jdx] += 0.5 * coupling
    return H.tocsr()

# Build Hamiltonian in Sz=0 sector
H_sz0 = build_sz0_hamiltonian(basis, index_map, N, h, J_par, J_perp)

# Compute ground state energy
energy, vectors = eigsh(H_sz0)
print(f"Ground state energy in Sz=0 sector: {energy[0]:.10f}")
print(f"Hilbert space size: 2^{N} = {2**N}, Sz=0 sector size = {len(basis)}")