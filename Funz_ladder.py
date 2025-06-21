"""
Self-built functions for the study of the Heisemberg S=1/2 model on a ladder

@author: david
"""

import numpy as np
import math
import time
import itertools
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, eigh_tridiagonal

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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



# Given a fixed system size and Sz, computes all the eigenvalues of that specific
# hamiltonian  sector, looping on different values of interacting parameter theta
def iteration(j, n,th_min,th_max,d_th,sz, complete_spectrum):
    lista = []
    num_points = int(round((th_max - th_min) / d_th)) + 1
    for t in np.linspace(th_min, th_max, num_points):
        param_Jpar = j*math.cos(t)
        param_Jperp = j*math.sin(t)
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
        
        if complete_spectrum:
            # Convert to dense (ONLY for LOW n!!!)
            A_dense = Block_H_Sz_sparse.toarray()   
            # Compute all eigenvalues (interesting for the degeneracy)
            eigenv, _ = eigh(A_dense) #(for the full spectrum)
            lista.append(eigenv)
            
        else:
            
            eigenv, _ = eigsh(Block_H_Sz_sparse, k=int(n//2)+1, which='SA')
            lista.append(eigenv)
        
    return lista



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



# Self-explicative..
def fit_cos(x, c1, c2):
    return c1 + 2*c2*np.cos(x)



# Take as input the splitted energy levels and the corresponding k for each 
# N_r = {4,6,8,10,12}, and makes a fit with cosine functions for the dispersion
# law. Then gives the resulting plots.
def multiple_fit(k, E):
    par_dic = {}
    cov_dic = {}
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for idx, i in enumerate(sorted(E)):
        # Fit the curve
        par_dic[i], cov_dic[i] = curve_fit(fit_cos, k[i], E[i])

        # Optional: plot the result
        fig_c, ax_c = plt.subplots(figsize=(6.2, 4.5))
        x_fit = np.linspace(min(k[i]), max(k[i]), 200)
        y_fit = fit_cos(x_fit, *par_dic[i])
        ax_c.scatter(k[i], E[i], color=colors[idx], label=rf'Degenerate levels for $N_r={round(i):.2f}$')
        ax_c.plot(x_fit, y_fit, color=colors[idx], label='Fit')
        
        ax_c.legend()
        ax_c.set_xlabel('k', fontsize=15)
        ax_c.set_ylabel('E(k)', fontsize=15)
        #ax_c.set_title(r'Degenerate-lifted levels fit:  $E(k) = c_1 + 2c_2 Cos(k)$')
        ax_c.grid(True)
        plt.show()
    
    return par_dic, cov_dic



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



# Plot (normalized) magnetization vs h for several ladder lengths
# at fixed interacting parameter theta
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
        y_heights = np.append(y_heights, i)
        

        x_plot = np.insert(x_steps, 0, 0)  
        x_plot = np.append(x_plot, 5*x_plot[-1]/4) # Make last step corresponds to m=max
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



# Plot (normalized) magnetization vs h for several ladder lengths
# at fixed interacting parameter theta
# (The SAME as "multiple_plot", but changes the plot titles..)
def multiple_plot_TR(n_min, n_max, param_h, param_J, param_th ):
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
    ax_m.set_title(fr'Magnetization for ladder TR Hamiltonian ($J={param_J:.2f}$, $\theta={param_th:.2f}$)')
    ax_m.legend(loc='upper left')
    ax_m.grid(True)
    plt.show()




# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    

# -----------------------------------------------------------------------------
# FUNCTIONS FOR THE SELF MADE LANCZOS ALGORITHM
# -----------------------------------------------------------------------------



# LADDER HAMILTONIAN with self made SPARSE MATRIX approach, used along with 
# self made LANCZOS algorithm to provide GS energy.
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



# Application of a sparse matrix to a vector. Used in the Lanczos algorithm
def sparse_matvec(H, row, col, v):
    size = max(row) + 1  # number of rows
    w = np.zeros(size)

    for h, r, c in zip(H, row, col):
        w[r] += h * v[c]
        
    return w



# LANCZOS algorithm for the lowest energy value of a specific Sz sector 
# of a hamiltonian sparse matrix
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



# Provides the self-made lanczos estimate of the GS energy as function of 
# Lanczos steps for a given Nr value
def lanczos_accuracy(n, m_min, m_max, d_m, Jpar, Jperp):
    # Arrays of all the allowed Sz values, and corresponding number of 'up' sites
    sz_fix = np.asarray([i for i in range(int(-n/2), int(n/2) +1)], dtype=int)
    n_1 = np.asarray([i for i in range(n+1)], dtype=int)
    dict_sz = dict(zip(sz_fix, n_1))

    # Generate the arrays associated to the sector Sz = fixed (ex. 0)
    list_sz_fixed = generate_binary_arrays(n, dict_sz[0])

    # Generate the corresponding arrays of label for the system states referred to 
     # a specific spin sector.
    arr_m_sz = array_of_integers(list_sz_fixed, n)

    # Zip, sort by v_m_Sz0, and unzip
    paired = sorted(zip(arr_m_sz, list_sz_fixed))      
    m_sz_sorted, list_sz_fixed_sorted = zip(*paired)

    # Convert back to arrays if needed
    m_sz_sorted = np.asarray(m_sz_sorted, dtype=np.int32)
    list_sz_fixed_sorted = np.asarray(list_sz_fixed_sorted)

    sp_H, sp_row, sp_col = Block_Hamiltonian(m_sz_sorted, list_sz_fixed_sorted, n, Jpar, Jperp)
    
    arr_m = np.arange(m_min, m_max, d_m)
    arr_eig = np.zeros(len(arr_m), dtype=np.float64)
    arr_time= np.zeros(len(arr_m), dtype=np.float64)
    i = 0
    for val in arr_m:
        
        start_time = time.time()
        GS = lanczos(sp_H, sp_row, sp_col, val)
        arr_eig[i] = GS
        end_time = time.time()
        elapsed_time = end_time - start_time
        arr_time[i] = elapsed_time
        i+=1
        
    return arr_m, arr_eig, arr_time



# Plots justapposed results from lanczos_accuracy for different values of Nrungs
def multiple_plot_Lanczos(m_min, m_max, d_m, Jpar, Jperp):
    
    fig_m, ax_m = plt.subplots(figsize=(6.2, 4.5))
    
    for i in range(4, 7, 2):
        Num = 2*i  # Total number of sites on the ladder
        v_x, v_y, v_t = lanczos_accuracy(Num, m_min, m_max, d_m, Jpar, Jperp)
       
        # Plot of step function    
        ax_m.plot(v_x, v_y, label=f'{i} rungs ')
    ax_m.set_xlabel(r' n° of m steps', fontsize=15)
    ax_m.set_ylabel(r'$ E_0 $', fontsize=15)
    ax_m.set_title(r'GS energy for different number of Lanczos steps ')
    ax_m.legend(loc='upper right')
    ax_m.grid(True)
    plt.show()



# For a given number of total sites, it gives the two different times needed to
# numerically estimate the GS energy: for the self-made Lanczos algorithm (where
# we use the optimal number of steps obtained from the previous accuracy study)
# and for the built-in function
def lanczos_comparison(n, m_lanc, Jpar, Jperp):
    # Arrays of all the allowed Sz values, and corresponding number of 'up' sites
    sz_fix = np.asarray([i for i in range(int(-n/2), int(n/2) +1)], dtype=int)
    n_1 = np.asarray([i for i in range(n+1)], dtype=int)
    dict_sz = dict(zip(sz_fix, n_1))

    # Generate the arrays associated to the sector Sz = fixed (ex. 0)
    list_sz_fixed = generate_binary_arrays(n, dict_sz[0])

    # Generate the corresponding arrays of label for the system states referred to 
     # a specific spin sector.
    arr_m_sz = array_of_integers(list_sz_fixed, n)

    # Zip, sort by v_m_Sz0, and unzip
    paired = sorted(zip(arr_m_sz, list_sz_fixed))      
    m_sz_sorted, list_sz_fixed_sorted = zip(*paired)

    # Convert back to arrays if needed
    m_sz_sorted = np.asarray(m_sz_sorted, dtype=np.int32)
    list_sz_fixed_sorted = np.asarray(list_sz_fixed_sorted)

    sp_H, sp_row, sp_col = Block_Hamiltonian(m_sz_sorted, list_sz_fixed_sorted, n, Jpar, Jperp)
    
    
    # Compare the results with the built-in exact diagonalization algorithm
    start_time_l = time.time()
    lanczos(sp_H, sp_row, sp_col, m_lanc)
    end_time_l = time.time()
    elapsed_time_l = end_time_l - start_time_l


    start_time_b = time.time()
    Block_sparse = Block_Hamiltonian_sparse(m_sz_sorted, list_sz_fixed_sorted, n, Jpar, Jperp)
    eigen_sp, _ = eigsh(Block_sparse, k=1, which='SA')
    end_time_b = time.time()
    elapsed_time_b = end_time_b - start_time_b
        
    return elapsed_time_l, elapsed_time_b



# Plot the two results of "lanczos_comparison" for different sizes of the system
def multiple_plot_comparison(n_min, n_max, Jpar, Jperp):
    
    arr_n = np.arange(n_min, n_max, 2)
    x_l = np.zeros(len(arr_n), dtype=np.int32)
    y_l = np.zeros(len(arr_n), dtype=np.float64)
    x_b = np.zeros(len(arr_n), dtype=np.int32)
    y_b = np.zeros(len(arr_n), dtype=np.float64)
    
    i = 0
    for n in arr_n:
        Num = 2*n  # Total number of sites on the ladder
        x_l[i] = n
        x_b[i] = n
        y_l[i], y_b[i] = lanczos_comparison(Num, 15)
        i += 1
       
    fig_m, ax_m = plt.subplots(figsize=(6.2, 4.5))
    ax_m.plot(x_l, y_l, label=r'Lanczos method')
    ax_m.plot(x_b, y_b, label=r'Built-in function')
    ax_m.set_xlabel(r' n° of ladder rungs', fontsize=15)
    ax_m.set_ylabel(r'Elapsed time [s]', fontsize=15)
    ax_m.set_title(r'Computational time to obtain the GS energy')
    ax_m.legend(loc='upper left')
    ax_m.grid(True)
    plt.show()
    
    
    
    
