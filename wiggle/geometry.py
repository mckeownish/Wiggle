import numpy as np 
import plotly.graph_objects as go

import pandas as pd
from biopandas.pdb import PandasPdb


def fit_rms(ref_c, c):
    
    # move geometric center to the origin
    
    ref_trans = np.average(ref_c, axis=0)
    ref_c = ref_c - ref_trans
    
    c_trans = np.average(c, axis=0)
    c = c - c_trans

    # covariance matrix
    cov = np.dot(c.T, ref_c)

    # Singular Value Decomposition
    (r1, s, r2) = np.linalg.svd(cov)

    # compute sign (remove mirroring)
    if np.linalg.det(cov) < 0:
        r2[2,:] *= -1.0
    U = np.dot(r1, r2)
    return (c_trans, U, ref_trans)

def get_pair_rmsd(c1, c2):
    # rmsd = 0.0
    c_trans, U, ref_trans = fit_rms(c1, c2)
    new_c2 = np.dot(c2 - c_trans, U) + ref_trans
    rmsd = np.sqrt( np.average( np.sum( ( c1 - new_c2 )**2, axis=1 ) ) )

    return rmsd

def return_aligned_xyz(c1, c2):

    c_trans, U, ref_trans = fit_rms(c1, c2)
    new_c2 = np.dot(c2 - c_trans, U) + ref_trans
    
    return new_c2
    
    
def find_optimal_alignment(c1, c2, rmsd_threshold=1.0, min_length=10):
    n = min(len(c1), len(c2))
    best_length = 0
    best_rmsd = float('inf')
    best_start = 0
    best_transformation = None
    
    for start in range(n - min_length + 1):
        for length in range(min_length, n - start + 1):
            subset_c1 = c1[start:start+length]
            subset_c2 = c2[start:start+length]
            
            c_trans, U, ref_trans = fit_rms(subset_c1, subset_c2)
            aligned_subset = np.dot(subset_c2 - c_trans, U) + ref_trans
            rmsd = np.sqrt(np.average(np.sum((subset_c1 - aligned_subset)**2, axis=1)))
            
            if rmsd <= rmsd_threshold and length > best_length:
                best_length = length
                best_rmsd = rmsd
                best_start = start
                best_transformation = (c_trans, U, ref_trans)
    
    if best_length > 0:
        # Apply the best transformation to the entire structure
        c_trans, U, ref_trans = best_transformation
        aligned_full_c2 = np.dot(c2 - c_trans, U) + ref_trans
        return best_start, best_length, best_rmsd, aligned_full_c2
    else:
        return None, None, None, None


def plot_aligned_2(coords_1, coords_2, outline=False):
    
    fig = go.Figure()

    aligned_coords = find_optimal_alignment(coords_1, coords_2)

    
    x_lst1, y_lst1, z_lst1 = coords_1[:,0], coords_1[:,1], coords_1[:,2]
    fig.add_trace(go.Scatter3d(x=x_lst1, y=y_lst1, z=z_lst1,
                                   mode='lines', opacity=1,
                                   line=dict( width=7, color='black')))

    x_lst2, y_lst2, z_lst2 = aligned_coords[:,0], aligned_coords[:,1], aligned_coords[:,2]
    fig.add_trace(go.Scatter3d(x=x_lst2, y=y_lst2, z=z_lst2,
                                   mode='lines', opacity=1,
                                   line=dict( width=7, color='red')))
    
    fig['layout']['showlegend'] = False
    fig.update_layout( scene=dict(
        xaxis_title='',
        yaxis_title='',
        zaxis_title='',
        aspectratio = dict( x=1, y=1, z=1 ),
        aspectmode = 'manual',
        xaxis = dict(
            gridcolor="white",
            showbackground=False,
            zerolinecolor="white",
            nticks=0,
            showticklabels=False),
        yaxis = dict(
            gridcolor="white",
            showbackground=False,
            zerolinecolor="white",
            nticks=0,
            showticklabels=False),
        zaxis = dict(
            gridcolor="white",
            showbackground=False,
            zerolinecolor="white",
            nticks=0,
            showticklabels=False),),
        )

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_traces(showlegend=False)
    fig.update_layout(width=800, height=800)
    return fig



# Atomic masses dictionary (simplified, add more if needed)
atomic_masses = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'S': 32.06
}


def load_pdb(file_path):
    """Load a PDB file using BioPandas."""
    pdb = PandasPdb().read_pdb(file_path)
    return pdb.df['ATOM']


def get_ca_coords(atom_df):

    return atom_df[ atom_df['atom_name'] == 'CA' ][['x_coord', 'y_coord', 'z_coord']].to_numpy()

def get_ca_from_pdb(pdb_file):
    atom_df = load_pdb(pdb_file)
    return get_ca_coords(atom_df) 


def get_atomic_mass(element):
    return atomic_masses.get(element, 0.0)


def calculate_center_of_mass(df):
    masses = np.array([get_atomic_mass(atom) for atom in df['element_symbol']])
    coords = df[['x_coord', 'y_coord', 'z_coord']].to_numpy()
    
    total_mass = np.sum(masses)
    if total_mass == 0:
        return np.zeros(3)  # Avoid division by zero
    
    center_of_mass = np.sum(coords.T * masses, axis=1) / total_mass
    return center_of_mass


def calculate_distance(coord1, coord2):
    return np.linalg.norm(coord1 - coord2)

def calculate_normal_vector(coord1, coord2):
    vector = coord2 - coord1
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector  # Avoid division by zero
    return vector / norm


def calculate_ca_geometric_normals(ca_coords):
    
    # Calculate vectors between consecutive CA atoms
    ca_vectors = np.diff(ca_coords, axis=0)
    
    # Calculate normals as the average of adjacent vectors
    normals = np.diff(ca_vectors, axis=0)
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]  # Normalize
    
    # Handle the first and last normals separately
    first_normal = ca_vectors[0] / np.linalg.norm(ca_vectors[0])
    last_normal = ca_vectors[-1] / np.linalg.norm(ca_vectors[-1])
    
    normals = np.vstack([first_normal, normals, last_normal])
    
    # negative give outward normals
    return -normals


def cart2spherical(coords):
    """
    Convert cartesian coordinates to spherical coordinates.
    """
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)  # Polar angle in the XY-plane
    phi = np.arccos(z / r)     # Angle from the Z-axis

    return r, theta, phi

def calculate_angle_differences(ca_coords, com_coords, geometric_normal_vectors):
    """
    Calculate the angular differences between the COM vectors and the geometric normal vectors.
    """


    # Calculate the vectors from CA to COM
    com_vectors = com_coords - ca_coords
    
    # Normalize the vectors
    com_vectors_norm = com_vectors / np.linalg.norm(com_vectors, axis=1)[:, None]
    normal_vectors_norm = geometric_normal_vectors / np.linalg.norm(geometric_normal_vectors, axis=1)[:, None]

    # Convert to spherical coordinates
    _, theta_com, phi_com = cart2spherical(com_vectors_norm)
    _, theta_normal, phi_normal = cart2spherical(normal_vectors_norm)

    # Calculate angular differences
    delta_theta = np.rad2deg(theta_com - theta_normal)
    delta_phi = np.rad2deg(phi_com - phi_normal)

    # Sum of angle differences (example with a sliding window of size 4)
    window_size = 4
    sum_theta = np.array([np.sum(delta_theta[i:i+window_size]) for i in range(len(delta_theta) - window_size + 1)])
    sum_phi = np.array([np.sum(delta_phi[i:i+window_size]) for i in range(len(delta_phi) - window_size + 1)])

    return delta_theta, delta_phi #, sum_theta, sum_phi



def calculate_geometric_positions(ca_coords, normal_vectors, lengths_to_com):
    """
    Calculate geometric positions based on normal vectors and lengths to the COM.
    """
    geometric_positions = ca_coords + normal_vectors * lengths_to_com[:, np.newaxis]
    return geometric_positions



def process_pdb_file(pdb_file):
    ppdb = PandasPdb().read_pdb(pdb_file)
    df = ppdb.df['ATOM']

    missing_cas = {}

    # remove alternative resiude locations
    df = df[ (df['alt_loc'] == 'A') | (df['alt_loc'] == '') ].reset_index(drop=True)
    
    residues = df.groupby(['chain_id', 'residue_number', 'residue_name'])
    results = []

    ca_coords = df[df['atom_name'] == 'CA'][['x_coord', 'y_coord', 'z_coord']].to_numpy()
    ca_geometric_normals = calculate_ca_geometric_normals(ca_coords)

    if len(ca_coords) != len(ca_geometric_normals):
        print(f"Length mismatch between CA coordinates and geometric normals in file {pdb_file}")
        return pd.DataFrame()  # Return an empty DataFrame if there is a mismatch

    for (chain_id, residue_number, residue_name), df_group in residues:
        ca_atom = df_group[df_group['atom_name'] == 'CA']

        if ca_atom.empty:
            print(f"No CA atom found for residue {residue_name}{residue_number} in file {pdb_file}")
            continue
            missing_cas[pdb_file] = residue_number

        ca_coord = ca_atom[['x_coord', 'y_coord', 'z_coord']].to_numpy().flatten()

        # Filter out backbone atoms (N, CA, C, O)
        side_chain_atoms = df_group[~df_group['atom_name'].isin(['N', 'CA', 'C', 'O'])]

        if side_chain_atoms.empty and residue_name not in ['ALA', 'GLY']:
            print(f"No side chain atoms found for residue {residue_name}{residue_number} in file {pdb_file}")

            raise ValueError(f"No side chain atoms found for residue {residue_name}{residue_number} in file {pdb_file}")
            # com = np.array([np.nan, np.nan, np.nan])
            # ca_com_normal_vector = np.array([np.nan, np.nan, np.nan])
            # distance_to_com = np.nan

        # if not side_chain_atoms.empty:

        if residue_name in ['ALA', 'GLY']:
            com = ca_coord
            ca_com_normal_vector = np.array([np.nan, np.nan, np.nan])
            
            distance_to_com = 1
            use_in_dist = False

        else:

            com = calculate_center_of_mass(side_chain_atoms)
            ca_com_normal_vector = calculate_normal_vector(ca_coord, com)
            distance_to_com = calculate_distance(ca_coord, com)
            use_in_dist = True
            
        # else:
        #     com = np.array([np.nan, np.nan, np.nan])
        #     ca_com_normal_vector = np.array([np.nan, np.nan, np.nan])
        #     distance_to_com = np.nan

        # Store results
        results.append({
            'chain_id': chain_id,
            'residue_name': residue_name,
            'residue_number': residue_number,
            'ca': ca_coord,
            'center_of_mass': com,
            'ca_com_normal_vector': ca_com_normal_vector,
            'distance_to_com': distance_to_com,
            'use_in_dist': use_in_dist
        })

    results_df = pd.DataFrame(results)
    
    ca_com_norms = np.array(results_df['ca_com_normal_vector'].values.tolist())

    ca_arr = np.array( results_df['ca'].values.tolist() )
    com_arr = np.array( results_df['center_of_mass'].values.tolist() )

    ca_arr = ca_arr[~np.isnan(ca_com_norms).any(axis=1)]
    ca_geo_norms = ca_geometric_normals[~np.isnan(ca_com_norms).any(axis=1)]
    com_arr = com_arr[~np.isnan(ca_com_norms).any(axis=1)]


    delta_theta, delta_phi = calculate_angle_differences(ca_arr, com_arr, ca_geo_norms)
    # results_df['delta_theta'] = delta_theta
    # results_df['delta_phi'] = delta_phi

    results_df['ca_geometric_normals'] = ca_geometric_normals.tolist()
    results_df['ca_geometric_normals'] = results_df['ca_geometric_normals'].apply(np.array)

    return results_df, delta_theta, delta_phi
