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

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = False


from Funz_ladder import iteration, multiple_fit, multiple_plot



Nr = 8    # Number of ladder rungs
N = 2*Nr  # Total number of sites on the ladder

# Interaction parameters
h = 0
J = 1                 
th = 0
J_par = J*math.cos(th)
J_perp = J*math.sin(th)




# -----------------------------------------------------------------------------
# 1) DEGENERACY and GAP of energy eigenvalues vs interacting param theta
# -----------------------------------------------------------------------------

n_tot,t_m, t_M, d_t, sz_sector = 12, 7*math.pi/16, math.pi/2, math.pi/64, 0
eigen_lst = iteration(J, n_tot, t_m, t_M, d_t, sz_sector, False)


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
                   color=color, linewidth=2)
    
    # Place label below lowest level, centered in column
    plt.text(x, min_energy - 0.5, coupling_labels[i],
             ha='center', va='top', fontsize=10, fontweight='bold', color='black')#=color)

# Format plot
plt.xticks([])
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.ylabel("Energy")
plt.title(fr"Energy Spectra for Varying Coupling Strengths ($N={n_tot:.2f}$, $S_z={sz_sector:.2f}$)")




# -----------------------------------------------------------------------------
# 1b) Lift of degeneracy of first excited level: COS fit
# -----------------------------------------------------------------------------

# Results for theta =  math.pi/2-math.pi/32. Degeneracy still not lifted completely!
E_nr = {
    4: np.array([-2.09849, -1.99532, -1.90276]),
    6: np.array([-3.59888, -3.54576, -3.44827, -3.40312]),
    8: np.array([-5.09925, -5.06772, -4.99607, -4.92967, -4.90349]),
    10: np.array([-6.59961, -6.57893, -6.52703, -6.46683, -6.42084, -6.40386]),
    12: np.array([-8.09996, -8.08542, -8.04683, -7.9968, -7.94936, -7.91611, -7.90421])
}

k_nr = {}
# Normalize each E_nri array by subtracting its average
# Define the corresponding k values in [0, 2*pi/Nr]
for i in E_nr:
    av_i = (np.max(E_nr[i]) + np.min(E_nr[i])) / 2
    E_nr[i] = E_nr[i] - av_i  # modifies in place
    k_nr[i] = np.linspace(0, math.pi, len(E_nr[i]))

justapposed_fit = multiple_fit()



# -----------------------------------------------------------------------------
# 4) MAGNETIZATION (normalized) vs h: size scaling at fixed interaction param th
# -----------------------------------------------------------------------------


justapposed_plots = multiple_plot(4, 10, 0, 1, math.pi/2)


