import numpy as np
from scipy.interpolate import UnivariateSpline

from Bio.PDB import PDBParser
import biobox as bb

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

from statsmodels.nonparametric.smoothers_lowess import lowess
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



def compare_plotting(q_exp, I_exp, I_exp_err, mod_dict, maxq=0.22, ylims=None, savename=None, exp_label='experimental'):

    '''
    Compare experimental data with model data

    args:
        q_exp: numpy array of q values
        I_exp: numpy array of I values
        I_exp_err: numpy array of I error values
        mod_dict: dictionary of model data
        maxq: maximum q value to plot
        ylims: y limits to plot

    Example:
        saxs_files = glob('MySAXS_files/*.w/e')
        saxs_files = sorted(saxs_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        q_lst, I_lst, I_err_lst = [], [], []

        for f in saxs_files:
            q, I, I_err = get_saxs_from_file(f)

            q_lst.append(q)
            I_lst.append(I)
            I_err_lst.append(I_err)

        I_av = np.mean(I_lst, axis=0)
        I_err_av = np.mean(I_err_lst, axis=0)

        mod_dict = {'WAXSiS': [qWAX, IWAX]}
        fn = 0
        compare_plotting(q[fn:], I_av[fn:], I_err_av[fn:], mod_dict, maxq = 0.25, savename=None)
    '''

    # Constants
    COLORS = {
        'red': '#b2182b',
        'blue': '#2166ac',
        'exp': 'black',
        'errorbar': 'lightgrey'
    }
    MODEL_COLORS = [COLORS['red'], COLORS['blue'], 'orange', 'green', 'purple', 'cyan', 'magenta']
    FONTSIZE = {
        'xlabel': 20,
        'ylabel': 20,
        'title': 16,
        'legend': 20,
        'tick': 16
    }

    def preprocess_data(q, I, I_err, minq, maxq):
        cond = (q >= minq) & (q <= maxq) & (I > 0)
        return q[cond], I[cond], I_err[cond]

    def calculate_model_data(q_exp, mod_dict):
        
        model_data = []
        for name, (q_mod, I_mod) in mod_dict.items():
            f = UnivariateSpline(q_mod, I_mod, s=0)
            I_mod_interp = f(q_exp)
            scale_factor = np.sum(I_exp * I_mod_interp) / np.sum(I_mod_interp ** 2)
            # scale_factor = 1

            I_mod_scaled = I_mod_interp * scale_factor

            logged_residual = np.log(I_exp) - np.log(I_mod_scaled)

            delta_residual = (I_exp - I_mod_scaled) / I_exp_err
            delta_residual_smoothed = lowess(delta_residual, q_exp, frac=0.1)

            chi_squared = np.sum(((I_exp - I_mod_scaled) / I_exp_err) ** 2) / len(I_exp)

            print(name, chi_squared)
            model_data.append({
                'name': name,
                'I_mod_scaled': I_mod_scaled,
                'logged_residual': logged_residual,
                'delta_residual': delta_residual,
                'delta_residual_smoothed': delta_residual_smoothed,
                'chi_squared': chi_squared
            })
        return model_data

    def setup_plot():
        plt.rcParams.update({'font.family': 'Arial'})
        fig = plt.figure(figsize=(10, 7), dpi=300)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax0)
        return fig, ax0, ax1

    def plot_experimental_data(ax, q, I, I_err, exp_label='experimental'):
        ax.errorbar(q, I, yerr=I_err, fmt='o', markersize=3, label=exp_label,
                    color=COLORS['exp'], ecolor=COLORS['errorbar'], capsize=0.3, alpha=0.3, zorder=1)

    def plot_model_data(ax0, ax1, q, model_data):
        for i, data in enumerate(model_data):
            color = MODEL_COLORS[i % len(MODEL_COLORS)]
            ax0.plot(q, data['I_mod_scaled'], color=color, label=f"{data['name']+' χ2: '+str(np.round(data['chi_squared'],2))}", zorder=2, alpha=0.8, linewidth=2)
            ax1.scatter(q, data['delta_residual'], marker='o', s=4, alpha=0.3, color=color)
            ax1.plot(data['delta_residual_smoothed'][:, 0], data['delta_residual_smoothed'][:, 1], color=color, alpha=0.8, linewidth=2)

    def format_axes(ax0, ax1):
        ax0.set_ylabel('I(q)', fontsize=FONTSIZE['ylabel'])
        ax0.set_yscale('log')
        ax0.legend(fontsize=FONTSIZE['legend'], frameon=False, loc='lower left', bbox_to_anchor=(0.05, 0.05))
        ax0.grid(True, linestyle=':', alpha=0.5)
        ax0.tick_params(axis='both', which='major', labelsize=FONTSIZE['tick'])
        ax0.tick_params(axis='both', which='minor', labelsize=FONTSIZE['tick']-2)

        

        ax1.set_xlabel('q (Å$^{-1}$)', fontsize=FONTSIZE['xlabel'])
        ax1.set_ylabel('Δ / σ', fontsize=FONTSIZE['ylabel'])
        ax1.grid(True, linestyle=':', alpha=0.5)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax1.tick_params(axis='both', which='major', labelsize=FONTSIZE['tick'])
        ax1.tick_params(axis='both', which='minor', labelsize=FONTSIZE['tick']-2)

        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['bottom'].set_visible(False)
        ax0.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.2, label='Zero Line')
    
        for ax in [ax0, ax1]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # Main execution
    q_exp, I_exp, I_exp_err = preprocess_data(q_exp, I_exp, I_exp_err, q_exp.min(), maxq)
    model_data = calculate_model_data(q_exp, mod_dict)
    
    fig, ax0, ax1 = setup_plot()
    plot_experimental_data(ax0, q_exp, I_exp, I_exp_err, exp_label)
    plot_model_data(ax0, ax1, q_exp, model_data)
    format_axes(ax0, ax1)

    fig.align_ylabels([ax0, ax1])
    
    if ylims:
        ax0.set_ylim(ylims)
    else:
   
        log_I = np.log10(I_exp)
        q1, q3 = np.percentile(log_I, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter the outliers
        non_outlier_mask = (log_I >= lower_bound) & (log_I <= upper_bound)
        if np.any(non_outlier_mask):
            y_min = 10**(np.min(log_I[non_outlier_mask]) - 0.2)
            y_max = 10**(np.max(log_I[non_outlier_mask]) + 0.2)
        else:
            y_min = np.min(I_exp) * 0.5
            y_max = np.max(I_exp) * 2
        
        ax0.set_ylim(y_min, y_max)

    fig.align_ylabels([ax0, ax1])
    plt.tight_layout()

    
    if savename:
        plt.savefig(savename, format="svg", bbox_inches="tight")
    plt.show()

    return model_data
