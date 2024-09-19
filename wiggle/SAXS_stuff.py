import numpy as np
from scipy.interpolate import UnivariateSpline

from Bio.PDB import PDBParser
import biobox as bb

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import least_squares

from tqdm import tqdm


### --- Carbonara --- ###

# average form factor carbonara parameters - found from Chris' optimisation circa 2014
def get_form_factor_carbonara_residue(q_values):

    a1 = -2.3495
    a2 = 6.94704
    a3 = 12.0347
    a4 = 5.8771
    a5 = -0.739775
    b1 = 23.2337
    b2 = 11.7446
    b3 = 7.58147
    b4 = 9.4941
    b5 = 25.4401
    c = 2.29714
    
    gf = 1.05
    formfactor = (
        a1 * np.exp(-gf * b1 * q_values**2) +
        a2 * np.exp(-gf * b2 * q_values**2) +
        a3 * np.exp(-gf * b3 * q_values**2) +
        a4 * np.exp(-gf * b4 * q_values**2) +
        a5 * np.exp(-gf * b5 * q_values**2) +
        c + 0.5
    )
    return formfactor

# Also with average water form factor - maybe pulled from the tables?
def get_form_factor_carbonara_hydration(q_values):

    a1_h = 4.41233
    a2_h = 0.191114
    a3_h = -1.79506
    a4_h = 15.7685
    a5_h = -18.2282
    b1_h = 19.2948
    b2_h = 61.3131
    b3_h = 15.0007
    b4_h = 18.6583
    b5_h = 19.3619
    c_h = 0.233361

    gf = 1.05
    formfactor = (
        a1_h * np.exp(-gf * b1_h * q_values**2) +
        a2_h * np.exp(-gf * b2_h * q_values**2) +
        a3_h * np.exp(-gf * b3_h * q_values**2) +
        a4_h * np.exp(-gf * b4_h * q_values**2) +
        a5_h * np.exp(-gf * b5_h * q_values**2) +
        c_h + 0.5
    )
    return formfactor


# Debye formula for CA position with averaged form factors
def calculate_carbonara_scattering_profile_noWater(q_values, pdb_file):

    # get the form factor residue average
    carb_ff = get_form_factor_carbonara_residue(q_values)

    M = bb.Molecule(pdb_file)
    cond_ca = M.data['name'] == 'CA'
    ca_coords = M.coordinates[0][cond_ca]

    I_q = np.zeros_like(q_values)
    
    # calculation of SAXS profile
    for i in tqdm( range(len(ca_coords)) ):
        r_ij = np.linalg.norm(ca_coords[i] - ca_coords, axis=1)
        I_q += carb_ff**2 * np.sum( np.sinc(q_values[:, None] * r_ij / np.pi), axis=1 )

    return I_q


### --- Explicit SAXS --- ###

# Known atomic form factors
def get_atomic_form_factor_dict(q):
    # Extended atomic form factor parameters for common elements using HF and RHF methods
    params = {
        'H': [0.489918, 0.262003, 0.196767, 0.049879, 20.6593, 7.74039, 49.5519, 2.20159, 0.001305],
        'C': [2.31000, 1.02000, 1.58860, 0.865000, 20.8439, 10.2075, 0.568700, 51.6512, 0.215600],
        'N': [12.2126, 3.13220, 2.01250, 1.16630, 0.005700, 9.89330, 28.9975, 0.582600, 0.255300],
        'O': [3.04850, 2.28680, 1.54630, 0.867000, 13.2771, 5.70110, 0.323900, 32.9089, 0.250800],
        'P': [6.43450, 4.17910, 1.78000, 1.49080, 1.90670, 27.1570, 0.526000, 68.1645, 0.286977],
        'S': [6.90530, 5.20340, 1.43790, 1.58630, 1.46790, 22.2151, 0.253600, 56.1720, 0.215600],
        'Fe': [11.7695, 7.35730, 3.52220, 2.30450, 0.005700, 1.49230, 85.3905, 0.657000, 0.215600],
        'Mg': [3.49880, 3.83780, 1.32840, 0.849700, 2.16760, 0.381400, 10.4045, 22.3119, 0.485300],
        'Ca': [8.00000, 7.95790, 6.49000, 1.92000, 0.080000, 0.808500, 8.24560, 0.425000, 0.215600]
        # Add more elements as needed
    }
    
    form_factor_q_dict = {}

    for element in params.keys():

        a1, a2, a3, a4, b1, b2, b3, b4, c = params[element]
        f_q = a1 * np.exp(-b1 * (q / (4 * np.pi))**2) + \
            a2 * np.exp(-b2 * (q / (4 * np.pi))**2) + \
            a3 * np.exp(-b3 * (q / (4 * np.pi))**2) + \
            a4 * np.exp(-b4 * (q / (4 * np.pi))**2) + c
        
        form_factor_q_dict[element] = f_q

    return form_factor_q_dict


# Explicit calculate debye formula over every atom - including hydrogen
def calculate_scattering_profile(q_values,pdb_file):

    # get the form factor lookup table
    form_factor_q_dict = get_atomic_form_factor_dict(q_values)

    # read in pdb
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)

    # get dem atoms
    atoms = list(structure.get_atoms())

    # Extract coordinates and elements
    coordinates = np.array([atom.coord for atom in atoms])
    elements = [atom.element for atom in atoms]

    # Initialize SAXS profile
    I_q = np.zeros_like(q_values)

    # pre-compute form factors for all elements (helps with performance)
    form_factors = np.array([form_factor_q_dict[el] for el in elements])

    # calculation of SAXS profile
    for i in tqdm( range(len(atoms)) ):
        r_ij = np.linalg.norm(coordinates[i] - coordinates, axis=1)
        I_q += np.sum(form_factors[i][:, None] * form_factors.T * np.sinc(q_values[:, None] * r_ij / np.pi), axis=1)


    return I_q


### --- Explicit Solvent modelling --- ###


### --- pyFoXS --- ###


### --- Visualisation --- ###

def plot_fit_data(q_exp, I_exp, model_data, xlims=None, fig_title='SAXS Profile'):
    """
    Plot and compare multiple SAXS profiles with experimental data.

    Parameters:
    q_exp (array): Experimental q-values
    I_exp (array): Experimental intensities
    model_data (list of dict): List of model data, each dict should have keys 'q', 'I', and 'label'
    xlims (tuple): Limits for the x-axis (optional)
    fig_title (str): Title of the figure (optional)
    """

    # plotting lims
    q_lb = min(min(q_exp), *[min(model['q']) for model in model_data])
    q_ub = max(max(q_exp), *[max(model['q']) for model in model_data])


    plt.figure(figsize=(10, 8), dpi=300)

    # Plot the experimental data
    plt.subplot(2, 1, 1)
    plt.scatter(q_exp, I_exp, label='Experimental data', alpha=0.2, color='red')
    plt.ylabel('I(q) (log scale)')
    plt.yscale('log')
    plt.title(fig_title)
    plt.legend()
    
    if xlims is not None:
        plt.xlim(xlims)
    else:
        plt.xlim([q_lb, q_ub])

    # Plot the residuals
    plt.subplot(2, 1, 2)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
    plt.xlabel('q (Å⁻¹)')
    plt.ylabel('Residuals (log scale)')
    
    if xlims is not None:
        plt.xlim(xlims)
    else:
        plt.xlim([q_lb, q_ub])

    for model in model_data:
        q_model = model['q']
        I_model = model['I']
        label = model['label']

        # Determine the common q range
        q_min = max(min(q_exp), *[min(model['q']) for model in model_data])
        q_max = min(max(q_exp), *[max(model['q']) for model in model_data])
        q_common_indices = (q_exp >= q_min) & (q_exp <= q_max)
        q_common = q_exp[q_common_indices]
        I_exp_common = I_exp[q_common_indices]
        
        # Interpolate the model to match common q values
        spline_model = UnivariateSpline(q_model, I_model, s=0)
        interpolated_I_model = spline_model(q_common)
        
        # Calculate the scale factor (least squares fit)
        scale_factor = np.sum(I_exp_common * interpolated_I_model) / np.sum(interpolated_I_model**2)

        # Apply the scale factor
        scaled_I_model = scale_factor * interpolated_I_model

        # Calculate residuals
        residuals = np.log(I_exp_common) - np.log(scaled_I_model)

        # Plot the scaled model
        plt.subplot(2, 1, 1)
        plt.plot(q_common, scaled_I_model, label=f'{label} (scaled)')
        
        # Plot the residuals
        plt.subplot(2, 1, 2)
        plt.plot(q_common, residuals, label=f'Residuals: {label}')
    
    plt.subplot(2, 1, 1)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.legend()

    plt.tight_layout()
    plt.show()

     


def calculate_chi_squared(I_exp, I_mod, sigma_exp=None):
    
    # If no experimental uncertainties provided, assume a value of 1
    if sigma_exp is None:
        sigma_exp = np.ones_like(I_exp)
    
    # Compute the chi-squared value
    chi_squared = np.sum(((I_exp - I_mod) ** 2) / (sigma_exp ** 2)) * 1/(len(I_exp))
    
    return round(chi_squared, 2)

def fancy_SAXS_profile_1(q_exp, I_exp, I_exp_err, q_mod1, I_mod1, label1, maxq=0.22, savename=None, data_label='data'):

    fontsize = 16

    # set q range to match experimental data
    minq = q_exp.min()
    # maxq = q_exp.max()

    cond = (q_exp >= minq) & (q_exp <= maxq)
    q_exp = q_exp[cond]
    I_exp = I_exp[cond]
    I_exp_err = I_exp_err[cond]


    # interpolate model to match experimental q values
    f1 = UnivariateSpline(q_mod1, I_mod1, s=0)
    q_mod1 = q_exp
    I_mod1 = f1(q_exp)

    scale_factor1 = np.sum(I_exp * I_mod1) / np.sum(I_mod1 ** 2)
    I_mod1 = I_mod1 * scale_factor1

    chisq1 = calculate_chi_squared(I_exp, I_mod1, sigma_exp=I_exp_err)


    fig = plt.figure(figsize=(6, 5), dpi=300)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])  # 2:1 ratio with some space at the bottom

    n_grain = 2

    # Plot the experimental data
    ax1 = plt.subplot(gs[0])
    ax1.scatter(q_exp[::n_grain], I_exp[::n_grain], label=data_label, alpha=0.4, color='grey', s=3)
    ax1.plot(q_mod1, I_mod1, label=label1+' χ2: '+str(chisq1), color='red', linewidth=1)

    ax1.set_ylabel('I(q)', fontdict={'size': fontsize})
    ax1.set_yscale('log')
    ax1.legend()
    # ax1.grid(True)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # ax1.set_xlim([min(q_mod), max(q_mod)])


    # Plot the residuals
    ax2 = plt.subplot(gs[1])
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.7)
    ax2.set_xlabel('q (Å⁻¹)', fontdict={'size': fontsize})
    ax2.set_ylabel('Residuals', fontdict={'size': fontsize})

    # # Calculate residuals
    residuals1 = np.log(I_exp) - np.log(I_mod1)
    ax2.plot(q_exp, residuals1, color='red', linewidth=1)


    # ax2.set_xlim([min(q_mod), max(q_mod)])
    ax2.set_ylim([-0.35, 0.35])

    # Adjust tick size
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=10)

    # Align y-labels
    fig.align_ylabels([ax1, ax2])

    # Adjust layout
    plt.tight_layout()

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax1.grid(True, color='lightgrey', linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.grid(True, color='lightgrey', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add some padding to the left for y-labels
    # plt.subplots_adjust(left=0.25)
    if savename:
        plt.savefig(savename, dpi=300)


    plt.show()


def fancy_SAXS_profile_2(q_exp, I_exp, I_exp_err, q_mod1, I_mod1, label1, q_mod2, I_mod2, label2, maxq = 0.22, savename=None, data_label='data'):
    
    fontsize = 16
    # set q range to match experimental data
    minq = q_exp.min()
    
    # maxq = q_exp.max()

    cond = (q_exp >= minq) & (q_exp <= maxq)
    q_exp = q_exp[cond]
    I_exp = I_exp[cond]
    I_exp_err = I_exp_err[cond]


    # interpolate model to match experimental q values
    f1 = UnivariateSpline(q_mod1, I_mod1, s=0)
    q_mod1 = q_exp
    I_mod1 = f1(q_exp)

    scale_factor1 = np.sum(I_exp * I_mod1) / np.sum(I_mod1 ** 2)
    I_mod1 = I_mod1 * scale_factor1

    chisq1 = calculate_chi_squared(I_exp, I_mod1, sigma_exp=I_exp_err)


    f2 = UnivariateSpline(q_mod2, I_mod2, s=0)
    I_mod2 = f2(q_exp)
    q_mod2 = q_exp

    scale_factor2 = np.sum(I_exp * I_mod2) / np.sum(I_mod2 ** 2)
    I_mod2 = I_mod2 * scale_factor2

    chisq2 = calculate_chi_squared(I_exp, I_mod2, sigma_exp=I_exp_err)


    fig = plt.figure(figsize=(6, 5), dpi=300)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])  # 2:1 ratio with some space at the bottom

    n_grain = 2

    # Plot the experimental data
    ax1 = plt.subplot(gs[0])
    ax1.scatter(q_exp[::n_grain], I_exp[::n_grain], label=data_label, alpha=0.4, color='grey', s=3)
    ax1.plot(q_mod1, I_mod1, label=label1+' χ2: '+str(chisq1), color='red', linewidth=1)
    ax1.plot(q_mod2, I_mod2, label=label2+' χ2:  '+str(chisq2), color='blue', linewidth=1)

    ax1.set_ylabel('I(q)', fontdict={'size': fontsize})
    ax1.set_yscale('log')
    ax1.legend()
    # ax1.grid(True)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # ax1.set_xlim([min(q_mod), max(q_mod)])


    # Plot the residuals
    ax2 = plt.subplot(gs[1])
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.7)
    ax2.set_xlabel('q (Å⁻¹)', fontdict={'size': fontsize})
    ax2.set_ylabel('Residuals', fontdict={'size': fontsize})

    # # Calculate residuals
    residuals1 = np.log(I_exp) - np.log(I_mod1)
    ax2.plot(q_exp, residuals1, color='red', linewidth=1)

    residuals2= np.log(I_exp) - np.log(I_mod2)
    ax2.plot(q_exp, residuals2, color='blue', linewidth=1)

    # ax2.set_xlim([min(q_mod), max(q_mod)])
    ax2.set_ylim([-0.35, 0.35])

    # Adjust tick size
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=10)

    # Align y-labels
    fig.align_ylabels([ax1, ax2])

    # Adjust layout
    plt.tight_layout()

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax1.grid(True, color='lightgrey', linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.grid(True, color='lightgrey', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add some padding to the left for y-labels
    # plt.subplots_adjust(left=0.25)
    if savename:
        plt.savefig(savename, dpi=300)


    plt.show()
