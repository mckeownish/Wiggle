import os
import subprocess

import numpy as np
from scipy.spatial.distance import pdist, squareform

import pandas as pd
from biopandas.pdb import PandasPdb

from typing import Tuple, Dict, Optional, List, NamedTuple, Union
from pathlib import Path
import tempfile
import warnings

import pickle

# form factors for hydrated geometric waxsis approx 
with open('/Users/josh/Documents/PhD/NoLooseEnds_Lab/FoXS-carbonara-comparison/Improving_Carbonara/models/implicit_form_factor_dict_q50_pro_corr_10-65.pkl', 'rb') as f:
    implicit_ff = pickle.load(f)


class ProteinAnalysis:
    """Class to handle protein structure analysis"""
    
    def __init__(self):
        # My residue COM distances from CA 
        self.residue_com_distances = {
            'ARG': 4.2662, 'ASN': 2.5349, 'ASP': 2.5558,
            'CYS': 2.3839, 'GLN': 3.1861, 'GLU': 3.2541,
            'HIS': 3.1861, 'ILE': 2.3115, 'LEU': 2.6183,
            'LYS': 3.6349, 'MET': 3.1912, 'PHE': 3.4033,
            'PRO': 1.8773, 'SEC': 1.5419, 'SER': 1.9661,
            'THR': 1.9533, 'TRP': 3.8916, 'TYR': 3.8807,
            'VAL': 1.9555, 'GLY': np.nan, 'ALA': np.nan
        }


    def extract_ca_coordinates(self, pdb_file: str) -> pd.DataFrame:

        # Read PDB file using biopandas
        ppdb = PandasPdb().read_pdb(pdb_file)
        atom_df = ppdb.df['ATOM']
        
        # Filter for CA atoms and clean alternative locations
        ca_df = atom_df[
            (atom_df['atom_name'] == 'CA') & 
            ((atom_df['alt_loc'] == '') | (atom_df['alt_loc'] == 'A'))
        ].copy()
        
        # Reset index after filtering
        ca_df.reset_index(drop=True, inplace=True)
        
        return ca_df


    def calculate_geometric_vectors_pro_corr(self, ca_coords, residue_names):
    
        # Calculate vectors between consecutive CA atoms
        ca_vectors = np.diff(ca_coords, axis=0)
        chain_direction = ca_vectors / np.linalg.norm(ca_vectors, axis=1)[:, None]
        
        # Calculate normals as the average of adjacent vectors
        normals = np.diff(ca_vectors, axis=0)
        normals = normals / np.linalg.norm(normals, axis=1)[:, None]  # Normalize

        a_pro = 47
        pro_mask = (residue_names[1:-1] == 'PRO')

        if np.any(pro_mask):

            rotation_axes = np.cross(normals, chain_direction[:-1])
            rotation_axes = rotation_axes / np.linalg.norm(rotation_axes, axis=1)[:, None]

            angle = np.radians(a_pro)
            cos_theta = np.cos(angle) 
            sin_theta = np.sin(angle)

            pro_rotated = (normals[pro_mask] * cos_theta + 
                        np.cross(rotation_axes[pro_mask], normals[pro_mask]) * sin_theta +
                        rotation_axes[pro_mask] * (np.sum(rotation_axes[pro_mask] * normals[pro_mask], axis=1) * (1 - cos_theta))[:, None])


            normals[pro_mask] = pro_rotated
        
        # Handle the first and last normals separately - hmm this approach is okay, although do I need to negate the last inferred normal (double neg)? 
        # [first] -> CA -> CA -> CA -> [end], since we -ve all normals
        #                           ^^ *(-) corrected
        first_normal = ca_vectors[0] / np.linalg.norm(ca_vectors[0])
        last_normal = -ca_vectors[-1] / np.linalg.norm(ca_vectors[-1])
        
        normals = np.vstack([first_normal, normals, last_normal])
        
        # negative give outward normals
        return -normals


    def calculate_geometric_vectors(self, ca_coords: np.ndarray) -> np.ndarray:

        # Calculate vectors between consecutive CA atoms
        ca_vectors = np.diff(ca_coords, axis=0)
        
        # Calculate and normalize normals
        normals = np.diff(ca_vectors, axis=0)
        normals = normals / np.linalg.norm(normals, axis=1)[:, None]
        
        # Handle first and last positions
        first_normal = ca_vectors[0] / np.linalg.norm(ca_vectors[0])
        last_normal = -ca_vectors[-1] / np.linalg.norm(ca_vectors[-1])
        
        # Combine all normals
        normals = np.vstack([first_normal, normals, last_normal])
        
        # Negate for outward direction
        return -normals


    def place_side_chains(self, ca_coords: np.ndarray, geometric_vectors: np.ndarray, residue_names: np.ndarray) -> np.ndarray:

        # Get distances for each residue
        distances = np.array([self.residue_com_distances.get(res, np.nan) for res in residue_names])
        
        # Calculate side chain positions
        side_chain_positions = ca_coords + (distances[:, np.newaxis] * geometric_vectors)
        
        return side_chain_positions
    

    def process_structure(self, pdb_file: str) -> dict:

        # Extract CA coordinates
        ca_df = self.extract_ca_coordinates(pdb_file)
        
        # Get CA coordinates as numpy array
        ca_coords = ca_df[['x_coord', 'y_coord', 'z_coord']].values
        self.ca_coords = ca_coords

        residue_names = ca_df['residue_name'].values
        self.residue_names = residue_names

        chain_ids = ca_df['chain_id'].values
        
        # Calculate geometric vectors
        geometric_vectors = self.calculate_geometric_vectors_pro_corr(ca_coords, residue_names)
        
        # Place side chains
        side_chain_positions = self.place_side_chains(ca_coords, geometric_vectors, residue_names)
        

        exl_cond = (ca_df['residue_name'] != 'ALA') & (ca_df['residue_name'] != 'GLY')

        bb_exl_centre_types = ['BB']*sum(exl_cond)
        bb_exl_ca_coords = ca_coords[exl_cond]
        bb_exl_chains = chain_ids[exl_cond]

        com_exl_centre_types = ca_df['residue_name'].values[exl_cond]
        com_exl_coords = side_chain_positions[exl_cond]
        com_exl_chains = chain_ids[exl_cond]


        ALA_GLY_cond = (ca_df['residue_name'] == 'ALA') | (ca_df['residue_name'] == 'GLY')

        bb_ALA_GLY_centre_types = ca_df['residue_name'].values[ALA_GLY_cond]
        bb_ALA_GLY_ca_coords = ca_coords[ALA_GLY_cond]
        bb_ALA_GLY_chains = chain_ids[ALA_GLY_cond]


        centre_types = np.concatenate([bb_exl_centre_types, com_exl_centre_types, bb_ALA_GLY_centre_types])
        geo_centre_coordinates = np.concatenate([bb_exl_ca_coords, com_exl_coords, bb_ALA_GLY_ca_coords])
        chains = np.concatenate([bb_exl_chains, com_exl_chains, bb_ALA_GLY_chains])

        # Calculate *all* scattering centres pairwise distances
        dist_matrix = squareform(pdist(geo_centre_coordinates))


        return {
            'centre_types':centre_types,
            'chain_ids':chains,
            'geo_centre_coordinates':geo_centre_coordinates,
            'r_ij':dist_matrix,
        }


    def calculate_saxs_implicit( self, preprocessed_data, form_factors_dict=implicit_ff ):
        """
        Estimate WAXSiS output from only protein alpha-carbons and geometrically placed sidechains

        Method uses hydrated form factors for each scattering centre type
        
        """

        centre_types = preprocessed_data['centre_types']
        r_ij = preprocessed_data['r_ij']

        # Get form factors for each centre type
        form_factors = np.array([form_factors_dict[ct] for ct in centre_types])  # shape: NxQ
        ff_products = form_factors @ form_factors.T
        
        # adaptive q range - will accept any q range and not throw an error? 
        N = len(form_factors_dict['BB'])
        q = np.linspace(0, (N-1)/100, N)

        # Expand q and r_ij for broadcasting
        q_expanded = q[:, np.newaxis, np.newaxis]  # shape: Qx1x1
        r_ij_expanded = r_ij[np.newaxis, :, :]  # shape: 1xNxN
        
        # Calculate sinc terms
        # np.sinc(x) = sin(πx)/(πx), which is what we want with x = qr/π
        sinc_terms = np.sinc(q_expanded * r_ij_expanded / np.pi)  # shape: QxNxN
        
        # Calculate intensity by summing over all pairs
        I_q = np.sum(ff_products * sinc_terms, axis=(1,2))
        
        return I_q



