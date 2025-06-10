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
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False

from Funz_ladder import Block_Hamiltonian_sparse_TR, blocks_GS_TR
from Funz_ladder import generate_binary_arrays, array_of_integers
from Funz_ladder import magnetisation




Nr = 6    # Number of ladder rungs
N = 2*Nr  # Total number of sites on the ladder

# Interaction parameters
h = 0
J = 1                 
th = math.pi/2
J_par = J*math.cos(th)
J_perp = J*math.sin(th)


"""
# ----------------------------------------------------------------------------
# 1) Provides the eigenvalues of a fixed Sz sector of the TR ladder hamiltonian (h=0)
# -----------------------------------------------------------------------------

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



# -----------------------------------------------------------------------------
# 2) DEGENERACY and GAP of energy eigenvalues vs interacting param theta
# -----------------------------------------------------------------------------


# Given a fixed system size and Sz, computes all the eigenvalues of that specific
# hamiltonian  sector, looping on different values of interacting parameter theta
def iteration(n,th_min,th_max,d_th,sz):
    lista = []
    num_points = int(round((th_max - th_min) / d_th)) + 1
    for t in np.linspace(th_min, th_max, num_points):
        param_Jpar = J*math.cos(t)
        param_Jperp = J*math.sin(t)
        Sz_fix = np.asarray([i for i in range(int(-n/2), int(n/2) +1)], dtype=int)
        N_1 = np.asarray([i for i in range(n+1)], dtype=int)
        dict_Sz = dict(zip(Sz_fix, N_1))
    
        # Generate the arrays associated to the sector Sz = fixed
        list_Sz_fixed = generate_binary_arrays(n, dict_Sz[sz])
    
        vett_m_Sz = array_of_integers(list_Sz_fixed, n)
    
        # Zip, sort by v_m_Sz0, and unzip
        paired = sorted(zip(vett_m_Sz, list_Sz_fixed))      
        v_m_Sz_sorted, list_Sz_fixed_sorted = zip(*paired)
    
        # Convert back to arrays if needed
        v_m_Sz_sorted = np.asarray(v_m_Sz_sorted, dtype=np.int32)
        list_Sz_fixed_sorted = np.asarray(list_Sz_fixed_sorted)
    
    
        # Sparse Hamiltonian
        Block_H_Sz_sparse_TR = Block_Hamiltonian_sparse_TR(v_m_Sz_sorted, list_Sz_fixed_sorted, n, param_Jpar, param_Jperp)
             
        # Convert to dense
        A_dense = Block_H_Sz_sparse_TR.toarray()   
        # Compute all eigenvalues (interesting for the degeneracy)
        #eigenv, _ = eigh(A_dense) (for the full spectrum. Ok for low n!!!)
        eigenv, _ = eigsh(A_dense, k=int(n//2)+1, which='SA')
        
        lista.append(eigenv)
        
    return lista


n_tot,t_m, t_M, d_t, sz_sector = 8, 3*math.pi/8, math.pi/2, math.pi/32, 0
eigen_lst = iteration(n_tot, t_m, t_M, d_t, sz_sector)



# Energy spectra for different coupling values
energy_spectra = eigen_lst

coupling_labels = [fr'$\theta={t:.2f}$' for t in np.arange(t_m, t_M + d_t, d_t)]

# X positions for each spectrum
num_columns = len(energy_spectra)
x_positions = range(-num_columns//2, num_columns//2 + 1)

# Get default color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Plotting parameters
line_length = 1.0

# Determine lowest energy for label alignment
all_energies = [e for spectrum in energy_spectra for e in spectrum]
min_energy = min(all_energies)

plt.figure(figsize=(8, 8))

# Plot each spectrum
for i, (energies, x) in enumerate(zip(energy_spectra, x_positions)):
    color = colors[i % len(colors)]
    for energy in energies:
        plt.hlines(energy, x - line_length/2, x + line_length/2,
                   color=color, linewidth=1)
    
    # Place label below lowest level, centered in column
    plt.text(x, min_energy - 0.5, coupling_labels[i],
             ha='center', va='top', fontsize=10, fontweight='bold', color='black')#=color)

# Format plot
plt.xticks([])
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.ylabel("Energy")
plt.title(fr"Energy Spectra for Varying Coupling Strengths ($N={n_tot:.2f}$, $S_z={sz_sector:.2f}$)")




"""
# -----------------------------------------------------------------------------
# 3) MAGNETIZATION vs h
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False

lowest_eigs_TR = blocks_GS_TR(N, J_par, J_perp)

# Discontinuity points (edges of steps)
x_steps, y_heights = magnetisation(lowest_eigs_TR)


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
# 4) MAGNETIZATION (normalized) vs h: size scaling at fixed interaction param th
# -----------------------------------------------------------------------------


# Plot (normalized) magnetization vs h for several ladder lengths
# at fixed interacting parameter theta
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
        y_heights = np.append(y_heights, i)


        x_plot = np.insert(x_steps, 0, 0) 
        x_plot = np.append(x_plot, 5*x_plot[-1]/4)
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


justapposed_plots_TR = multiple_plot(2, 8, 0, 1, math.pi/2)

"""