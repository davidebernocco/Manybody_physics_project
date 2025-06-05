"""
Heisemberg S=1/2 model on a ladder.
(PBC on the two parallel legs "==")

    1==3==5==7==9      "==" : J_parallel = J*cos(theta)
    |  |  |  |  |      " | " : J_perpendicular = J*sin(theta)
    0==2==4==6==8


The Hilbert space associated to the full system is (C^2)tensor(C^2)...(C^2)

@author: david
"""

import numpy as np
import math
import time
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

from Funz_ladder import generate_binary_arrays, array_of_integers
from Funz_ladder import Block_Hamiltonian_sparse, blocks_GS, magnetisation


Nr = 4    # Number of ladder rungs
N = 2*Nr  # Total number of sites on the ladder

# Interaction parameters
h = 0
J = 1                 
th = math.pi / 2 - 0.1
J_par = J*math.cos(th)
J_perp = J*math.sin(th)



# -----------------------------------------------------------------------------
# 1)  Ladder Hamiltonian as EFFECTIVE HAMILTONIAN of a bosonic particle (triplet)
#      hopping into void neighbours (singlets)
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





def triplet(list_Sz, list_m, n_tr):
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
            
            
        
vet_tr = triplet(list_Sz_fixed_sorted, v_m_Sz_sorted, 1)



keys_list = list(vet_tr.keys())
values_list = list(vet_tr.values())

# Suppose you already have this list (subset of v)
picked_values = keys_list

# Get indices of these values in v
indices = [i for i, val in enumerate(v_m_Sz_sorted) if val in picked_values]

# Select corresponding rows in A
selected_rows = list_Sz_fixed_sorted[indices]



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
        moment_dict = triplet(lst_conf, lst_m, 1)
        lst_pos = [item for sublist in moment_dict.values() for item in sublist]
        my_dict['new triplet position'].append(lst_pos)

    return my_dict


Ham_eff = Hopping_check(selected_rows, vet_tr)
print(Ham_eff)
"""



# -----------------------------------------------------------------------------
# 2) On the degeneracy of levels vs theta
# -----------------------------------------------------------------------------



"""
Sz_fix = np.asarray([i for i in range(int(-N/2), int(N/2) +1)], dtype=int)
N_1 = np.asarray([i for i in range(N+1)], dtype=int)
dict_Sz = dict(zip(Sz_fix, N_1))

# Generate the arrays associated to the sector Sz = fixed
list_Sz_fixed = generate_binary_arrays(N, dict_Sz[0])

vett_m_Sz = array_of_integers(list_Sz_fixed, N)

# Zip, sort by v_m_Sz0, and unzip
paired = sorted(zip(vett_m_Sz, list_Sz_fixed))      
v_m_Sz_sorted, list_Sz_fixed_sorted = zip(*paired)

# Convert back to arrays if needed
v_m_Sz_sorted = np.asarray(v_m_Sz_sorted, dtype=np.int32)
list_Sz_fixed_sorted = np.asarray(list_Sz_fixed_sorted)


# Sparse Hamiltonian
Block_H_Sz_sparse = Block_Hamiltonian_sparse(v_m_Sz_sorted, list_Sz_fixed_sorted, N, J_par, J_perp)
     
# Convert to dense
A_dense = Block_H_Sz_sparse.toarray()   
# Compute all eigenvalues (interesting for the degeneracy)
eigenvalues, _ = eigh(A_dense)
"""

# ----------------------------

"""
import matplotlib.pyplot as plt

# Example energy levels
energies = [1, 2, 3.5, 5, 6.8]  # y-values where lines are placed
line_length = 1.0               # Length of horizontal lines
x_center = 0                    # Horizontal center of the lines

# Create figure
plt.figure(figsize=(4, 6))

# Plot horizontal lines
for energy in energies:
    x_start = x_center - line_length / 2
    x_end = x_center + line_length / 2
    plt.hlines(energy, x_start, x_end, color='black', linewidth=1)

# Remove x-axis and frame
plt.xticks([])
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# Label y-axis
plt.ylabel("Energy")
plt.title("Energy Level Diagram")

# Optional: invert y-axis to show low energy at bottom
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
"""




def iteration(n,th_min,th_max,d_th,sz):
    lista = []
    for t in np.arange(th_min, th_max + d_th, d_th):
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
        Block_H_Sz_sparse = Block_Hamiltonian_sparse(v_m_Sz_sorted, list_Sz_fixed_sorted, n, param_Jpar, param_Jperp)
             
        # Convert to dense
        A_dense = Block_H_Sz_sparse.toarray()   
        # Compute all eigenvalues (interesting for the degeneracy)
        eigenv, _ = eigh(A_dense)
        
        lista.append(eigenv)
        
    return lista


t_m, t_M, d_t = 0, math.pi/2, math.pi/8
risultato = iteration(8, t_m, t_M, d_t, 0)



import matplotlib.pyplot as plt

# Energy spectra for different coupling values
energy_spectra = risultato

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
plt.title("Energy Spectra for Varying Coupling Strengths")



"""
start_time2 = time.time()
eigenvalues_sparse, _ = eigsh(Block_H_Sz_sparse, which='SA')
print("GS energy for SPARSE MATRIX", eigenvalues_sparse[0])
end_time2 = time.time()
elapsed_time2 = end_time2 - start_time2
print("Elapsed time2:", elapsed_time2)

# ////////////////////////////////////////////////////////////////////////////


# -----------------------------------------------------------------------------
# 3) Extract the m eigenvalues with a built-in efficient function
# -----------------------------------------------------------------------------

autovalori = blocks_GS(N, J_par, J_perp)




# ////////////////////////////////////////////////////////////////////////////


# -----------------------------------------------------------------------------
# 4) Obtain the "magnetization vs h" plot, starting from energy eigenvalues
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False

# Discontinuity points (edges of steps)
x_steps, y_heights = magnetisation(autovalori)


x_plot = np.insert(x_steps, 0, 0)  
y_plot = y_heights / Nr             
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
"""



# ///////////////////////////////////////////////////////////////////////////

# -----------------------------------------------------------------------------
# 5) Size scaling at fixed interaction parameters
# -----------------------------------------------------------------------------

"""
def multiple_plot(n_min, n_max, param_h, param_J, param_th ):
   
    # Interaction parameters
    param_J_par = param_J * math.cos(param_th)
    param_J_perp = param_J * math.sin(param_th)
    
    fig_m, ax_m = plt.subplots(figsize=(6.2, 4.5))
    
    for i in range(n_min, n_max+1, 2):
        Num = 2*i  # Total number of sites on the ladder
        autovalori = blocks_GS(Num, param_J_par, param_J_perp)
        
        # Discontinuity points (edges of steps)
        x_steps, y_heights = magnetisation(autovalori)


        x_plot = np.insert(x_steps, 0, 0)  
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


justapposed_plots = multiple_plot(2, 10, 0, 1, 0)
"""


