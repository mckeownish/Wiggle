"""
Carbonara's Data Tools (CDT) Module
-----------------------------------

This module provides functionalities for processing specific data inputs/outputs related
to Carbonara, such as extracting coordinates, secondary structures, sequences, and
generating PDB files for protein structure analysis. It encompasses tools for inferring
geometric CB positions and preparing data for structure AA reconstruction and analysis.

Available Functions:
--------------------
extract_coords:
    Extracts coordinate data from a file (XYZ.dat)

extract_coords_from_PDB:
    Extracts CA coordinate data from a file pdb file

extract_sequence_FP:
    Extract sequence from a fingerprint file (FP.dat)
    
Carbonara_2_PDB:
    Generates a PDB file from Carbonara output.

infer_CB_positions:
    Infers C-beta atom positions from alpha carbon coordinates.

DSSP_structure_extractor:
    Predicts secondary structure from a PDB file using DSSP.

generate_random_structures:
    Generates random protein structures by altering linker sections.

remove_isolated_elements:
    Removes isolated elements in secondary structure predictions.

get_secondary:
    Retrieves the secondary structure from a fingerprint file.

read_coords:
    Reads and cleans coordinate data from a file.

section_finder:
    Identifies protein sub-unit sections from the full secondary structure.

find_sheet_indices:
    Locates sheet sub-unit section indices.

find_linker_indices:
    Identifies linker sub-unit section indices.

generate_random_structures:
    Creates random structures by altering one linker section at a time.

sheet_group_mask:
    Groups adjacent sheets in a secondary structure file.

linker_group_mask:
    Groups adjacent linkers in a secondary structure file.

get_sheet_coords:
    Retrieves coordinates of CA atoms in each sheet structure.

get_section_groupings:
    Groups sections in a secondary structure.

list_nohidden:
    Lists files in a directory, excluding hidden ones.

sheet_pipe:
    Processes and extracts sheet coordinates from data files.

sheet_pairwise_bond_number:
    Computes the number of pairs of CA atoms within a threshold.

random_bond_finder:
    Finds bond patterns in randomly generated structures.

set_up_varying_sections:
    Sets up sections of a protein that vary in structure.

find_non_varying_linkers:
    Identifies linkers that do not vary significantly in structure.

setup_working_directory:
    Sets up a directory for data processing.

setup_molecule_directory:
    Sets up a directory for a specific molecule's data.

pdb_2_biobox:
    Converts a PDB file into a BioBox object.

extract_CA_coordinates:
    Extracts alpha carbon coordinates from a BioBox object.

extract_sequence_fromBB:
    Extracts the sequence of amino acids from a BioBox object.

write_fingerprint_file:
    Writes fingerprint data to a file.

write_coordinates_file:
    Writes coordinate data to a file.

write_mixture_file:
    Writes mixture data to a file.

write_varysections_file:
    Writes varying section data to a file.

write_saxs:
    Writes SAXS data to a file.

read_dssp_file:
    Reads DSSP file and simplifies the secondary structure.

simplify_secondary:
    Simplifies secondary structure annotations.

write_sh_file:
    Writes a shell script for running structure predictions.

write_sh_qvary_file:
    Writes a shell script with varying q values for structure fitting.

SAXS_selection_plotter:
    Plots SAXS data with selected q-range.

get_minmax_q:
    Retrieves minimum and maximum q values from SAXS data.

sort_by_creation:
    Sorts files by their creation time.

smooth_me:
    Smoothens coordinates for visualization.

smooth_me_varying:
    Smoothens varying sections of coordinates for visualization.

line_plotly:
    Creates a 3D line plot using Plotly.

log2df:
    Converts log file data into a pandas DataFrame.

df2plot:
    Creates a plot from a DataFrame.

SAXS_fit_plotter:
    Plots SAXS fitting data.

fit_rms:
    Calculates root mean square fit for coordinates.

find_rmsd:
    Finds the RMSD between two sets of coordinates.

align_coords:
    Aligns a tensor of coordinates.

coord_tensor_pairwise_rmsd:
    Computes pairwise RMSD for a tensor of coordinates.

cluster:
    Performs clustering on RMSD data.

visualise_clusters:
    Visualizes clusters of structures.

overlay_coords:
    Overlays multiple sets of coordinates for comparison.
"""

import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.spatial.distance import cdist

import os
import subprocess
import shutil
from tqdm import tqdm
from glob import glob

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import biobox as bb

from plotly.subplots import make_subplots
import plotly.graph_objects as go

#import hdbscan


def extract_coords(coords_file):

    """
    Extracts and cleans coordinate data from a given file.

    This function reads a file containing coordinates, typically from a .dat file.
    It removes any rows containing NaN values to ensure clean data for further processing.

    Parameters:
    coords_file (str): The path to the file containing coordinate data.

    Returns:
    numpy.ndarray: An array of cleaned coordinates.
    """
    
    coords = np.genfromtxt(coords_file)
    coords = coords[~np.isnan(coords).any(axis=1)]
    
    return coords
    

def extract_coords_from_PDB(pdb_file):

    """
    Extract the XYZ coords from a PDB file.

    This function reads a protein pdb file with biobox and extracts the XYZ coordinates of CA atoms

    Parameters:
    
    coords_file (str): Path to the file with alpha carbon coordinates.
    fingerprint_file (str): Path to the Carbonara specific format file containing
                            secondary structure and sequence information.
                            
    output_file (str): Path for the output PDB file.

    Returns:
    None
    """
    
    M = pdb_2_biobox(pdb_file)
    coords = extract_CA_coordinates(M)
    
    return coords

def extract_sequence_FP(fingerprint_file):

    """
    Extract a sequence single string file from FingerPrint file (a Chris special format).


    Parameters:

    fingerprint_file (str): Path to the Carbonara specific format file containing
                            secondary structure and sequence information.
                            
    Returns:
    seq (str): Single string with HS- format sequence of protein.
    """
    
    seq = open(fingerprint_file, 'r').readlines()[2][:-1]
    
    return seq


def Carbonara_2_PDB(coords_file, fingerprint_file, output_file):
    
    """
    Generates a PDB file from Carbonara output.

    Processes the coordinates and fingerprint data from Carbonara's output,
    maps amino acids to their three-letter abbreviations, and writes the
    result to a PDB file using the biobox library.

    Parameters:
    
    coords_file (str): Path to the file with alpha carbon coordinates.
    fingerprint_file (str): Path to the Carbonara specific format file containing
                            secondary structure and sequence information.
                            
    output_file (str): Path for the output PDB file.

    Returns:
    None
    """
    # read in coordinates and fingerprint 
    coords = extract_coords(coords_file)
    size = coords.shape[0]
    seq = extract_sequence_FP(fingerprint_file)
    
    # map the AA shorthand to 3 letter abr.     
    aa_map = {
            'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU',
            'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
            'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER',
            'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
        }
    
    # map sequence to 3 character abr.
    seq_3 = []
    for a in list(seq):
        seq_3.append(aa_map[a])
    
    # create dataframe for biobox molecule type
    df = pd.DataFrame({'atom':['ATOM']*size, 'index':np.arange(size), 'name':['CA']*size, 
                       'resname':seq_3, 'chain':['A']*size, 'resid':np.arange(size),
                       'occupancy':[1]*size, 'beta':[50]*size, 'atomtype':['C']*size, 
                       'radius':[1.7]*size, 'charge':[0]*size})
    
    # take full advantage of Matteo's lovely biobox library - manually 'create' a molecule 
    molecule = bb.Molecule()
    molecule.data = df
    molecule.coordinates = np.expand_dims(coords, axis=0)
    
    # write out!
    molecule.write_pdb(output_file)
    

def infer_CB_positions(CA_xyz):
    
    """
    Geometrically infers the positions of C-beta atoms from alpha carbon (CA) coordinates.

    Given a set of alpha carbon coordinates, this function calculates the normals
    to the CA chain and uses these to estimate the positions of C-beta atoms.

    Parameters:
    CA_xyz (numpy.ndarray): A Nx3 numpy array of alpha carbon coordinates.

    Returns:
    CB_xyz (numpy.ndarray): An (N-2)x3 array of inferred beta carbon coordinates {excludes chain ends}
    """
    
    CA_vecs = np.diff(CA_xyz, axis=0)
    normals = np.diff(CA_vecs, axis=0)
    normals = normals/np.linalg.norm(normals, axis=1)[:,None]

    av_bond_len = 3.8

    normals = normals*av_bond_len

    CB_xyz = CA_xyz[1:-1] - normals # minus as to face outwards
    
    return CB_xyz


    
def interlace_CA_CB_write(CA_xyz, CB_xyz, protein, output_pdb):
    
    # positions of CB - wont add CB to the residues
    gly_idx = np.where(protein.data['resname'].values== 'GLY')[0]
  
    # atom names
    atom_name_lst = []

    # interlaced CA + CB coordinates
    coordinates = CA_xyz[0]

    # number of CA
    size = CA_xyz.shape[0]

    # creating index for resid
    idx_counter = 0
    idx_lst = []

    # new interlaced CA CB sequence
    new_seq = []

    # extract CA sequence
    ca_seq = protein.data['resname'].values 
    
    for i in range(size):

        # ...CA bit...
        idx_lst.append(idx_counter)
        atom_name_lst.append('CA')
        new_seq.append(ca_seq[i])

        if i > 0:
                coordinates = np.vstack([coordinates, CA_xyz[i]])

        # ...CB bit...
        if i not in gly_idx:

            if (i > 0) & (i < size-1):
                idx_lst.append(idx_counter)
                atom_name_lst.append('CB')
                coordinates = np.vstack([coordinates, CB_xyz[i-1]])
                new_seq.append(ca_seq[i])

        idx_counter += 1

        
    tot_size = int(CA_xyz.shape[0]+CB_xyz.shape[0]-gly_idx.shape[0])

    if coordinates.shape[0] == tot_size:

        # create dataframe for biobox molecule type
        df = pd.DataFrame({'atom':['ATOM']*tot_size, 'index':np.arange(tot_size),
                           'name':atom_name_lst, 'resname':new_seq,
                           'chain':['A']*tot_size, 'resid':idx_lst,
                           'occupancy':[1]*tot_size, 'beta':[50]*tot_size,
                           'atomtype':['C']*tot_size, 'radius':[1.7]*tot_size, 
                           'charge':[0]*tot_size})

    else:
        raise ValueError('Total number of CA + CB - no. GLY res does not equal the coordinate size!')
        
        
    molecule = bb.Molecule()
    molecule.data = df
    molecule.coordinates = np.expand_dims(coordinates, axis=0)
    
    molecule.write_pdb(output_pdb)
    
    
    
def CA_PDB_2_CB_PDB(CA_pdb, output_pdb):
    
    # Load protein into BioBox object
    protein = bb.Molecule(CA_pdb)
     
    # Get CA coordinates
    CA_xyz = protein.coordinates[0]
    
    # infer the CB positions
    CB_xyz = infer_CB_positions(CA_xyz)
    
    interlace_CA_CB_write(CA_xyz, CB_xyz, protein, output_pdb)
    

def remove_isolated_elements(secondary, struct_sym):
    
    """
    This function scans the secondary structure array and replaces isolated instances
    of helix (H) or sheet (S) with linker (-).
        
        Example (struct_sym='H')
        
        ---H---SSSSS--- =:> -------SSSSS---
        
        <Vibe check: how could a single helix element exist when definition comes from 3+ CAs>
        
        Example (struct_sym='S')
        
        HHHHH---S-----S =:> HHHHH----------
        
    Parameters:
    secondary (numpy.ndarray): Array of secondary structure symbols (HS- format).
    struct_sym (str): The structural symbol to be targeted for removal.

    Returns:
    secondary (numpy.ndarray): The modified secondary structure array with isolated H or S replaced with -.
    """
    
    x = np.where(secondary==struct_sym)[0]

    consec_x = np.split(x, np.where(np.diff(x) != 1)[0]+1)

    sheet2linker_idx = []

    for arr in consec_x:
        if arr.shape[0]==1:
            sheet2linker_idx.append(arr.item())
            
    secondary[sheet2linker_idx] = '-'
    
    return secondary


    
def DSSP_structure_extractor(pdb_file, DSSP_path='/home/josh/anaconda3/bin/mkdssp'):
    
    """
    Predicts secondary structure from an (all atomistic) PDB file using DSSP.

    This function uses DSSP (Define Secondary Structure of Proteins) to predict the secondary
    structure of protein residues in a PDB file. It simplifies the DSSP output to a basic
    set of secondary structure labels.

    Parameters:
    pdb_file (str): Path to the PDB file for secondary structure prediction.
    DSSP_path (str): Path to dssp or mkdssp local install (defaulted to my local machine)
    
    Returns:
    numpy.ndarray: An array of residue secondary structure labels.
    """
    
    p = PDBParser()
    structure = p.get_structure("PDB_file", pdb_file)
    model = structure[0]

    # **************** LOOK HERE! THE MKDSSP LOC NEEDS TO BE CHANGED FOR PLATFORM ********************
    dssp = DSSP(model, pdb_file, dssp=DSSP_path)

    simplify_dict = {'H' : 'H', 'P' : 'H', 'B': 'S', 'E': 'S', 'G': 'H', 'I': 'H', 'T': '-', 'S': '-', '-': '-', ' ': '-'}
    secondary_struct = []

    for key in list(dssp.keys()):
        secondary_struct.append( simplify_dict[ dssp[key][2] ] )
        
    secondary_struct = np.asarray(secondary_struct)

    # cleaning up DSSP's mess - remove those pesky isolated helices and secondary
    secondary_struct = remove_isolated_elements(secondary_struct, 'H')
    secondary_struct = remove_isolated_elements(secondary_struct, 'S')

    return secondary_struct
        
    
def get_secondary(fingerprint_file):

    """
    Retrieves the secondary structure from a fingerprint file (HS- format expected).

    Parameters:
    fingerprint_file (str): Path to the fingerprint file (HS- format).

    Returns:
    numpy.ndarray: An array representing the secondary structure.
    """
    
    return np.asarray(list(np.loadtxt(fingerprint_file, str)[2]))


def read_coords(coords_file):
    
    """
    Reads coordinate data from a file and cleans it.

    This function reads XYZ coordinates from a specified file (usually .dat),
    removes any NaN values, and returns a clean NumPy array of the coordinates.

    Parameters:
    coords_file (str): Path to the file containing coordinates.

    Returns:
    numpy.ndarray: An array of cleaned coordinates.
    """
    
    coords = np.genfromtxt(coords_file)
    coords = coords[~np.isnan(coords).any(axis=1)]
    
    return coords


def section_finder(ss):
    
    """
    Identifies distinct sections in the protein's secondary structure.
    
    ADD EXAMPLE TO CLARIFY: ['H', 'H', 'H', 'S', 'S', 'S', '-', '-'] =:> ['H', 'S', '-']
    (I think)
    
    Parameters:
    ss (numpy.ndarray): Array representing the protein's secondary structure.

    Returns:
    sections (numpy.ndarray): An array of distinct section identifiers.
    """
    
    sections = []
    structure_change = np.diff(np.unique(ss, return_inverse=True)[1])

    for i, c in enumerate( structure_change ):

        if c!=0:
            sections.append(ss[i])

        if i==structure_change.shape[0]-1:
            sections.append(ss[i])
            
    sections = np.array(sections)
    
    return sections #, linker_indices #, structure_change


def find_sheet_indices(sections):
    
    """
    Locates indices of sheet residues within the protein's secondary structure.

    Parameters:
    sections (numpy.ndarray): Array of section identifiers in the secondary structure.

    Returns:
    numpy.ndarray: Array of indices corresponding to sheet sections.
    """

    sheet_indices = np.where(sections=='S')[0]
    return sheet_indices


def find_linker_indices(sections):
    
    '''Find linker sub-unit section indices'''
    
    linker_indices = np.where(sections=='-')[0]
    return linker_indices


def generate_random_structures(coords_file, fingerprint_file):
    
    '''Generate random structures changing one linker section at a time
    
    Parameters
    coords_file:       /path/ to CA coordinates.dat file
    fingerprint_file:  /path/ to fingerprint.dat file
    
    Return
    Generated structures are written to ~/rand_structures/.. section_*LINKERINDEX*.dat as xyz
    Linker Indices
    '''
    
    """
    Generates random protein structures by altering linker sections.

    This function creates random protein structures by changing one linker section
    at a time based on the provided coordinate and fingerprint files.

        Generated structures are written to ~/rand_structures/..
        > section_*LINKERINDEX*.dat as xyz
        
    Parameters:
    coords_file (str): Path to CA coordinates .dat file.
    fingerprint_file (str): Path to fingerprint .dat file.

    Returns:
    linker_indices (numpy.ndarray): An array of linker indices used in generating structures.
    """
    
    linker_indices = find_linker_indices( section_finder( get_secondary(fingerprint_file) ) ) 
    
    current = os.getcwd()
    random = 'rand_structures'
    random_working = os.path.join(current, random)

    if os.path.exists(random_working) and os.path.isdir(random_working):
        shutil.rmtree(random_working)

    os.mkdir(random_working)

    for l in tqdm(linker_indices):
        
        outputname = random_working+'/section_'+str(l)
        
        # this must not run detached - full random structure output is needed when called later!
        result = subprocess.run(['./generate_structure', fingerprint_file, coords_file, outputname, str(l)], capture_output=True, text=True)
    
    return linker_indices


def sheet_group_mask(ss):
     
    '''Groups adjacent sheets in secondary structure file and returns a grouping mask ( 0 : not a sheet;  1+: sheet )
    
    Parameters
    ss (numpy array):            Secondary structure labels (array of strings)
    
    Returns
    sheet_groups (numpy array):  Mask of grouped sheet sections
    '''
    
    sheet_mask = (ss == 'S')*1
    sheet_groups = np.zeros(ss.shape[0])
    group = 1
    
    if sheet_mask[0] == 1:
        label = True
    else:
        label = False

    for i, c in enumerate(np.diff(sheet_mask)):
        
        if c == 1:
            label = True

        elif c==-1:
            label=False
            group += 1

        else:
            pass 

        if label == True:
            if ss[i+1] == 'S':
                sheet_groups[i+1] = group
                
    return sheet_groups


def linker_group_mask(ss):
    
    '''Groups adjacent linkers in secondary structure file and returns a grouping mask ( 0 : not a linker;  1+: linker )
    
    Parameters
    ss (numpy array):             Secondary structure labels (array of strings)
    
    Returns
    linker_groups (numpy array):  Mask of grouped linker sections
    '''
    
    linker_mask = (ss == '-')*1
    linker_groups = np.zeros(ss.shape[0])
    group = 1
    
    # checking first index for linker 
    if linker_mask[0] == 1:
        label = True
        linker_groups[0] = group
    else:
        label = False

    for i, c in enumerate(np.diff(linker_mask)):
    
        if c == 1:
            label = True

        elif c==-1:
            label=False
            group += 1

        else:
            pass 

        if label == True:
            
            linker_groups[i+1] = group
                
    return linker_groups #, linker_mask


def get_sheet_coords(coords, sheet_groups):

    '''Finds CA coordinates of 
    
    Parameters
    coords (numpy array):        xyz coordinates of all protein CA atoms
    sheet_groups (numpy array):  Mask of grouped sheet sections
    
    Returns
    sheet_coords (numpy array):  xyz coordinates of CA atoms in each sheet structure [ [...sheet 1 coords...] [...sheet 2 coords...] ... ]
    '''
    
    sheet_coords = []

    for g in np.unique(sheet_groups):
        if g>0:
            sheet_coords.append(coords[sheet_groups==g])
    
    # print('**********************')
    # print(sheet_coords)
    # sheet_coords = np.asarray(sheet_coords)
    
    return sheet_coords


def get_section_groupings(ss, structure_change):
    
    group = 0
    structural_groups = np.zeros(ss.shape)
    structural_groups[0] = group

    for i, c in enumerate(structure_change):

        if c != 0:
            group += 1

        structural_groups[i+1] = group
    return structural_groups


def list_nohidden(path):
    lst = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            lst.append(f)
    return lst


def sheet_pipe(coords_file, fingerprint_file):
   
    coords = read_coords(coords_file)
    ss = get_secondary(fingerprint_file)
    sheet_groups = sheet_group_mask( np.asarray(ss) )
    sheet_coords = get_sheet_coords(coords, sheet_groups)

    return sheet_coords


def sheet_pairwise_bond_number(sheet_coords, thr=5.5):
    
    '''Finds the number of pairs of CA atoms within some threshold between all sheet sections
    
    Parameters
    sheet_coords (numpy array): xyz coordinates of CA atoms in each sheet structure [ [...sheet 1 coords...] [...sheet 2 coords...] ... ]
    thr (float) {optional}:     Cutoff distance for inter-sheet bonding (default = 5.5 Å)
    
    Returns
    pairwise_bond_num (numpy array): Lower triangular array containing the number of individual CA bonds within threshold between each sheet pair
    
    '''
    
    number_bonds = 0

    pairwise_bond_num = np.zeros([len(sheet_coords), len(sheet_coords)])

    for i in range(1,len(sheet_coords)):

        for j in range(0,i):

            arr1, arr2 = sheet_coords[j], sheet_coords[i]
            dist_matrix = cdist(arr1, arr2)
            indices = np.where(dist_matrix < thr)

            pairwise_bond_num[i,j] = indices[0].shape[0]

            number_bonds += indices[0].shape[0]
    return pairwise_bond_num 


def random_bond_finder(rand_file_dir, fingerprint_file, linker_indices):
   
    # grouping all random structure for each linker together
    
    struture_lst = list_nohidden(rand_file_dir)
   
    linker_file_dict = {}
    for l in linker_indices:
        tmp = []
       
        for file in np.sort(struture_lst):
            if str(l) == file.split('_')[1]:
                tmp.append(file)

        linker_file_dict[l] = tmp
       
    # Pairwise sheet bonds for each random str for each linker
    linker_bond_dict = {}

    for l in linker_indices:
       
        tmp = []
       
        for file in linker_file_dict[l]:
            coords_file = rand_file_dir+file
            sheet_coords = sheet_pipe(coords_file, fingerprint_file)
            tmp.append( sheet_pairwise_bond_number(sheet_coords) )
   
        linker_bond_dict[l] = tmp
    
    return linker_bond_dict 



def set_up_varying_sections(initial_coords_file, fingerprint_file):
    
    # Reference initial structure
    sheet_coords = sheet_pipe(initial_coords_file,
                              fingerprint_file)
    ref_bonds = sheet_pairwise_bond_number(sheet_coords, thr=5.5)


    # Generate the random structure changing each linker section
    linker_indices = generate_random_structures(initial_coords_file,
                                                        fingerprint_file)
    
    # Calculate the number of inter-sheet bonds for each rand struct
    linker_bond_arr_dict = random_bond_finder('rand_structures/', 
                                              fingerprint_file,
                                              linker_indices)
    
    # Find number of bond breaks relative to initial structure
    bond_breaks_dict = {}

    for l in linker_indices:

        bond_break_lst = []
        for bond_arr in linker_bond_arr_dict[l]:


            bond_break_lst.append( (ref_bonds > bond_arr).sum() )

        bond_breaks_dict[l] = sum(bond_break_lst)/(len(linker_bond_arr_dict[l])+1)
    
    return linker_indices, bond_breaks_dict


def find_non_varying_linkers(linker_indices, bond_breaks_dict, av_break_thr=0.001):

    # Linker indices that cause no bond breaks
    # print(bond_breaks_dict)
    conds = np.asarray(list(bond_breaks_dict.values())) < av_break_thr
      
    allowed_linker = linker_indices[conds]

    # if 0 in linker_indices:
    #     linker_indices = np.delete(linker_indices, np.where(linker_indices==0)[0].item())

    # if 0 in allowed_linker:
    #     allowed_linker = np.delete(allowed_linker, np.where(allowed_linker==0)[0].item())

    return allowed_linker


# ------ Carbonara Setup Funcs ---------


def setup_working_directory(run_name='Fitting'):
    
    current = os.getcwd()
    working = 'newFitData/'+run_name+'/'
    working_path = os.path.join(current, working)
    
    if os.path.exists(working_path):
        shutil.rmtree(working_path)
        print('Removing existing working directory')
        
    os.makedirs(working_path)
    os.mkdir(working_path+'/fitdata')

    print('Complete')
    return working_path


def setup_molecule_directory(molecule_name='Test_Molecule', ignore_overwrite=True):
    
    current = os.getcwd()
    working = 'newFitData/'+molecule_name+'/'
    working_path = os.path.join(current, working)
    skip = False

    if ignore_overwrite:
        if os.path.exists(working_path):
            shutil.rmtree(working_path)
            print('Removing existing working directory')

        os.makedirs(working_path)

    else:
        if os.path.exists(working_path):
            skip = True
        else:
            os.makedirs(working_path)

    # os.mkdir(working_path+'/fitdata')

    
    return working_path


def pdb_2_biobox(pdb_file):
    M = bb.Molecule()
    M.import_pdb(pdb_file)
    return M


def extract_CA_coordinates(M):
    ca_idx = (M.data['name']=='CA').values
    ca_coords = M.coordinates[0][ca_idx]
    
    if ca_coords.shape[0] != M.data['resid'].nunique():
        raise Exception("You better check your PDB... The number of CA atoms does not equal the number of ResIDs in your PDB file!") 
    else:
        return ca_coords

    
def extract_sequence_fromBB(M):

    aa_names = {
                'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU',
                'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
                'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
                'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER',
                'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
                }

    names_aa = {y: x for x, y in aa_names.items()}
    
    ca_idx = (M.data['name']=='CA').values
    resnames = M.data['resname'][ca_idx].map(names_aa).values
    
    if resnames.shape[0] != M.data['resid'].nunique():
        raise Exception("You better check your PDB... The number of CA atoms does not equal the number of ResIDs in your PDB file!") 
    else:
        return resnames

                                                  
def write_fingerprint_file(number_chains, sequence, secondary_structure, working_path):
    
    assert isinstance(number_chains, int), 'Yikes... The number of chains is not int type!'
    
    if number_chains > 1:
        print('Are sure you have more than one chain - if not this will cause segmentation errors later! You have been warned...')
    
    seq_run = ''.join(list(sequence))
    ss_run = ''.join(list(secondary_structure))
    
    if len(seq_run) != len(ss_run):
        raise Exception("Uh Oh... The length of sequence and secondary structure is not equal!") 
    
    f = open(working_path+"/fingerPrint1.dat", "w")
    f.write(str(number_chains))
    f.write('\n \n')
    f.write(seq_run)
    f.write('\n \n')
    f.write(ss_run)
    f.close()
    
    
def write_coordinates_file(coords, working_path):
    
    assert type(coords).__module__ == np.__name__, 'Thats never good... the CA coordinates are not a numpy array'
    np.savetxt(working_path+'/coordinates1.dat', coords, delimiter=' ', fmt='%s',newline='\n', header='', footer='')
    
    
def write_mixture_file(working_path):
    # if default:
    f = open(working_path+"/mixtureFile.dat", "w")
    f.write(str(1))
        
#     else:
#          copy input file


def write_varysections_file(varying_sections, working_path):
    # auto: run beta sheet breaking code; write output sections to file
    f = open(working_path+"/varyingSectionSecondary1.dat", "w")
    for i, s in enumerate(varying_sections):
        f.write(str(s))
        
        if i < len(varying_sections)-1:
            f.write('\n')
    f.close()

    
def write_saxs(SAXS_file, working_path):
    
    saxs_arr = np.genfromtxt(SAXS_file)
    
    if saxs_arr.shape[1] == 3:
        saxs_arr = saxs_arr[:,:2]
    
    np.savetxt(working_path+'/Saxs.dat', saxs_arr, delimiter=' ', fmt='%s',newline='\n', header='', footer='')


def read_dssp_file(dssp_filename):
    
    simplify_dict = {'H': 'H', 'B': 'S', 'E': 'S', 'G': 'H', 'I': 'H', 'T': '-', 'S': '-', '-': '-', ' ': '-'}
    
    lines=[]
    with open(dssp_filename) as input_data:
        # Skips text before the beginning of the interesting block:
        for line in input_data:
            if line.strip() == '#  RESIDUE AA STRUCTURE BP1 BP2  ACC     N-H-->O    O-->H-N    N-H-->O    O-->H-N    TCO  KAPPA ALPHA  PHI   PSI    X-CA   Y-CA   Z-CA': 
                break
        # Reads text until the end of the block:
        for line in input_data:  # This keeps reading the file
            lines.append(simplify_dict[line[16]])
    return ''.join(lines)
    
    
def simplify_secondary(dssp_struct):
    
    simplify_dict = {'H': 'H', 'B': 'S', 'E': 'S', 'G': 'H', 'I': 'H', 'T': '-', 'S': '-', '-': '-', ' ': '-'}
    
    secondary_structure = []
    
    for s in dssp_struct:
        
        if s not in list(simplify_dict.keys()):
            print('>>> ', s, ' <<<')
            raise Exception('Secondary structure not recognised!')
            
        secondary_structure.append(simplify_dict[s])
        
    return secondary_structure


def write_sh_file(working_path, fit_n_times, min_q, max_q, max_fit_steps):
    
    curr = os.getcwd()
    run_file = curr + '/RunMe.sh'

    with open(run_file, 'w+') as fout:
        fout.write('#!/bin/bash')
        
        saxs_file = working_path+'/Saxs.dat'
        FP_file = working_path+"/fingerPrint1.dat"
        coords_file = working_path+'/coordinates1.dat'
        varying_file = working_path+"/varyingSectionSecondary1.dat"
        mixture_file = working_path+"/mixtureFile.dat"
        
        # Auto assign min / max q from SAXS profile
        # saxs_arr = np.genfromtxt(saxs_file)
        # min_q = np.round(saxs_arr[:,0].min(),2)
        # max_q = np.round(saxs_arr[:,0].max(),2)
        
        fout.write('\nfor i in {1..'+str(fit_n_times)+'}')

        fout.write('\n\ndo')
        fout.write('\n\n   echo " Run number : $i "')
        fout.write('\n\n   ./predictStructure ' + saxs_file + ' ' + working_path+'/' + ' ' + coords_file + ' ' + 'none' + ' ' + varying_file + ' ' + '1' + ' ' + 'none' + \
                   ' ' + 'none' + ' ' + str(min_q) + ' ' + str(max_q) + ' ' + str(max_fit_steps) + ' ' + working_path+'/fitdata/fitmolecule$i' + ' ' + working_path+'/fitdata/scatter$i.dat' + ' ' + mixture_file + ' ' +'1')
                   
        fout.write('\n\ndone')
        
    print('Successfully written bash script to: ', run_file) 


def write_sh_qvary_file(working_path, mol_name, fit_name, fit_n_times, min_q, max_q, max_fit_steps):
    
    curr = os.getcwd()
    script_name = '/RunMe_'+ mol_name + '_' + fit_name + '.sh'
    run_file = curr + script_name

    with open(run_file, 'w+') as fout:
        fout.write('#!/bin/bash')
        
        saxs_file = working_path+'Saxs.dat'
        FP_file = working_path+"fingerPrint1.dat"
        coords_file = working_path+'coordinates1.dat'
        varying_file = working_path+"varyingSectionSecondary1.dat"
        mixture_file = working_path+"mixtureFile.dat"
        
        # Auto assign min / max q from SAXS profile
        # saxs_arr = np.genfromtxt(saxs_file)
        # min_q = np.round(saxs_arr[:,0].min(),2)
        # max_q = np.round(saxs_arr[:,0].max(),2)
        
        fout.write('\nfor i in {1..'+str(fit_n_times)+'}')

        fout.write('\n\ndo')
        fout.write('\n\n   echo " Run number : $i "')
        fout.write('\n\n   ./predictStructureQvary ' + saxs_file + ' ' + working_path + ' ' + coords_file + ' ' + 'none' + ' ' + varying_file + ' ' + '1' \
                   + ' ' + 'none' + ' ' + 'none' + ' ' + str(min_q) + ' ' + str(max_q) + ' ' + str(max_fit_steps) + ' ' + working_path+fit_name+'/mol$i' \
                   + ' ' + working_path+fit_name+'/scatter$i.dat' + ' ' + mixture_file + ' ' + working_path + ' ' + working_path+fit_name+'/fitLog$i.dat')
                   
        fout.write('\n\ndone')
        
    # return script_name[1:]
    # print('Successfully written bash script to: ', run_file) 




def SAXS_selection_plotter(SAXS_file, min_q, max_q):

    SAXS = np.genfromtxt(SAXS_file)

    q = SAXS[:,0]
    I = SAXS[:,1]

    q_selected = q[(q>=min_q)&(q<=max_q)]
    q_grey = q[(q<min_q) | (q>=max_q)]

    I_selected = I[(q>=min_q)&(q<=max_q)]
    I_grey = I[(q<min_q) | (q>=max_q)]

    fig = go.Figure( data=[go.Scatter(x=q_grey, y=I_grey, mode='markers', line=dict(color="grey"), opacity=0.7, name='Excluded')])
    fig.add_trace( go.Scatter(x=q_selected, y=I_selected, mode='markers', line=dict(color="crimson"), name='Selected') )
    fig.update_layout(
                    # title='Selected q range',
                     yaxis_type = "log", template='plotly_white',
                    width=800, height=700, font_size=28)
    fig.update_xaxes(title='q')
    fig.update_yaxes(title='I')
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))


    return fig


def get_minmax_q(SAXS_file):
    
    SAXS = np.genfromtxt(SAXS_file)

    q = SAXS[:,0]

    q_exp_min = float(np.round(q.min(),2))
    q_exp_max = float(np.round(q.max(),2))

    q_spread = q_exp_max - q_exp_min

    q_Q1 = float(np.round(0.00*q_spread,2))
    q_Q3 = float(np.round(0.45*q_spread,2))

    return q_exp_min, q_exp_max, q_Q1, q_Q3


def sort_by_creation(file_lst):

    files = list(filter(os.path.isfile, file_lst))
    files.sort(key=lambda x: os.path.getmtime(x))
    return files



# ------ Protein Visulation ------


def smooth_me(coords, ss, oversample=5):
    
    color_dic = {'-':'yellow', 'H':'red', 'S':'blue'}
    
    structure_change = np.diff(np.unique(ss, return_inverse=True)[1])
    sections = section_finder(ss)
    structural_groups = get_section_groupings(ss, structure_change)
    
    color_lst = []
    
    x_lst = [] 
    y_lst = [] 
    z_lst = [] 

    for i, sec in enumerate(sections):

        tmp = coords[np.where(structural_groups==i)]

        if tmp.shape[0]>3:

            if sec == 'H':
                tck, u = interpolate.splprep([tmp[:,0], tmp[:,1], tmp[:,2]], s=.25)

            else: 
                tck, u = interpolate.splprep([tmp[:,0], tmp[:,1], tmp[:,2]], s=10)
            res_size = oversample*tmp.shape[0]
            u_fine = np.linspace(0,1,res_size)
            new_points = interpolate.splev(u_fine, tck)

            x_lst = x_lst + list(new_points[0])
            y_lst = y_lst + list(new_points[1])
            z_lst = z_lst + list(new_points[2])

            color_lst = color_lst + [color_dic[sec]] * res_size

        else:

            x_lst = x_lst + list(tmp[:,0])
            y_lst = y_lst + list(tmp[:,1])
            z_lst = z_lst + list(tmp[:,2])

            color_lst = color_lst + [color_dic[sec]] * tmp.shape[0]
    
    return x_lst, y_lst, z_lst, color_lst


def smooth_me_varying(coords, ss, vary_sections, oversample=5):

    structure_change = np.diff(np.unique(ss, return_inverse=True)[1])
    sections = section_finder(ss)
    structural_groups = get_section_groupings(ss, structure_change)

    color_lst = []

    x_lst = [] 
    y_lst = [] 
    z_lst = [] 

    for i, sec in enumerate(sections):

        tmp = coords[np.where(structural_groups==i)]

        if tmp.shape[0]>3:

            if sec == 'H':
                tck, u = interpolate.splprep([tmp[:,0], tmp[:,1], tmp[:,2]], s=1)

            else: 
                tck, u = interpolate.splprep([tmp[:,0], tmp[:,1], tmp[:,2]], s=100)
            res_size = oversample*tmp.shape[0]
            u_fine = np.linspace(0,1,res_size)
            new_points = interpolate.splev(u_fine, tck)

            x_lst = x_lst + list(new_points[0])
            y_lst = y_lst + list(new_points[1])
            z_lst = z_lst + list(new_points[2])

            if i in vary_sections:
                color_lst = color_lst + ['red'] * res_size

            elif sec == 'S':
                color_lst = color_lst + ['blue'] * res_size
            
            else:
                color_lst = color_lst + ['grey'] * res_size
        else:

            x_lst = x_lst + list(tmp[:,0])
            y_lst = y_lst + list(tmp[:,1])
            z_lst = z_lst + list(tmp[:,2])

            if i in vary_sections:
                color_lst = color_lst + ['red'] * tmp.shape[0]
            else:
                color_lst = color_lst + ['grey'] * tmp.shape[0]

    return x_lst, y_lst, z_lst, color_lst


def line_plotly(x_lst, y_lst, z_lst, color_lst, outline=False):
    
    fig = go.Figure()

    if outline:
        fig.add_trace(go.Scatter3d(x=x_lst, y=y_lst, z=z_lst,
                                mode='lines',
                                line=dict( width=21, color='black')))

    fig.add_trace(go.Scatter3d(x=x_lst, y=y_lst, z=z_lst,
                               mode='lines',
                               line=dict( width=12, color=color_lst)))

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
    return fig


def log2df(log_file):
    arr = np.loadtxt(log_file, str, skiprows=4)

    columns = ['index', 'step_num', 'fit_quality', 'writhe_penalty', 'overlap_penalty', 'contact_penalty', 'time', 'max_fitting_k', 'file_name']
    df_log = pd.DataFrame(arr, columns=columns)

    for c in columns[:-1]:
        df_log[c] = df_log[c].astype(float)

    df_log['time'] = df_log['time'].astype(float) * 1/60 * 10**(-6)
    df_log['total_penalty'] = df_log['fit_quality'] + df_log['writhe_penalty'] + df_log['overlap_penalty'] + df_log['contact_penalty']
    
    return df_log


def df2plot(df_log, highlight=False):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_log['time'].values,
                             y=df_log['total_penalty'].values,
                             name='Total Penalty',
                             mode='lines + markers',
                             line=dict(color="lightblue", width=3) ))
    
    if highlight:
        fig.add_trace(go.Scatter(x=[df_log['time'].values[highlight]],
                                 y=[df_log['total_penalty'].values[highlight]],
                                 name='selected',
                                 mode='markers',
                                 marker=dict( symbol="hexagon", color="crimson", size=10
                                            ) ))

    fig.update_layout(template='simple_white') 
    # height=600, width=1400
    fig.update_layout(xaxis_title="Time (Minutes)", yaxis_title="Total Penalty")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_traces(showlegend=False)
    # fig.show()
    return fig


def SAXS_fit_plotter(SAXS_file, fit_file, full_q=True):
    
    fig = make_subplots(rows=2, cols=1,row_heights=[0.7,0.3],vertical_spacing=0,shared_xaxes=True)

    SAXS = np.genfromtxt(SAXS_file)

    fitting = np.genfromtxt(fit_file, skip_footer=1)
    fit_q = fitting[:,0]
    fit_I = fitting[:,2]

    q = SAXS[:,0]
    I = np.log(SAXS[:,1])
    
    min_q = fit_q.min()
    max_q = fit_q.max()

    cond = (q >= min_q) & (q <= max_q) 
    q_range = q[cond]
    I_range = I[cond]

    tck = interpolate.splrep(fit_q, fit_I)
    spli_I = interpolate.splev(q_range,tck)

    residuals = spli_I - I_range

    if full_q:
        fig.add_trace( go.Scatter(x=q, y=I, mode='markers', line=dict(color="grey"), opacity=0.7, name='Data'),row=1,col=1 )
    
    else:
        fig.add_trace( go.Scatter(x=q_range, y=I_range, mode='markers', line=dict(color="grey"), opacity=0.7, name='Data'),row=1,col=1 )

    fig.add_trace( go.Scatter(x=fit_q, y=fit_I, mode='markers', 
                        marker=dict( color='crimson', size=8),
                         name='Fit'),row=1,col=1 )
                         
    fig.add_trace( go.Scatter(x=q_range, y=spli_I, mode='lines', line=dict(color="crimson", width=3), name='Fit'),row=1,col=1 )

    fig.add_trace(go.Scatter(x=q_range,y=residuals,mode='lines',name='Residual',showlegend=False,line=dict(color='red')),row=2,col=1)
    fig.add_trace(go.Scatter(x=q_range,y=np.zeros_like(q_range),mode='lines',showlegend=False,line=dict(color='black',dash='dash',width=1)),row=2,col=1)

    fig.update_layout(
        # title='Experiment vs Fit', 
                      # yaxis_type = "log", 
                    template='simple_white',
                    # width=1200, height=800, 
                    font_size=18)

    fig.update_yaxes(title_text="Intensity I(q)", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    fig.update_xaxes(title_text="q", row=2, col=1)

    # max_res = max( np.abs(residuals).max(), .5)
    max_res = np.abs(residuals).max()*1.3
    fig.update_yaxes(range=[-max_res,max_res],row=2,col=1)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_traces(showlegend=False)
    return fig


def fit_rms(ref_c,c):

    # move geometric center to the origin
    ref_trans = np.average(ref_c, axis=0)
    ref_c = ref_c - ref_trans
    c_trans = np.average(c, axis=0)
    c = c - c_trans

    # covariance matrix
    C = np.dot(c.T, ref_c)

    # Singular Value Decomposition
    (r1, s, r2) = np.linalg.svd(C)

    # compute sign (remove mirroring)

    if np.linalg.det(C) < 0:
        r2[2,:] *= -1.0
    U = np.dot(r1, r2)

    return (c_trans, U, ref_trans)


def find_rmsd(c1, c2, align_return=False):

    rmsd = 0.0
    c_trans, U, ref_trans = fit_rms(c1, c2)

    new_c2 = np.dot(c2 - c_trans, U) + ref_trans

    rmsd = np.sqrt( np.average( np.sum( ( c1 - new_c2 )**2, axis=1 ) ) )

    if align_return:
        return rmsd, new_c2
    
    else:
        return rmsd
    
def align_coords(tensor):
    
    new_coord_tensor = np.zeros_like(tensor)
    new_coord_tensor[:,:,0] = tensor[:,:,0]
    size = tensor.shape[-1]
    
    for i in range(1,size):
        rmsd, align_coords = find_rmsd(tensor[:,:,0], tensor[:,:,i], align_return=True)
        new_coord_tensor[:,:,i] = align_coords
        
        
    return new_coord_tensor


def coord_tensor_pairwise_rmsd(coord_tensor):
      
    size = coord_tensor.shape[-1]
    rmsd_arr = np.zeros((size, size))
    # size = 10
    for i in range(size):
        for j in range(i):
            rmsd = find_rmsd(coord_tensor[:,:,i], coord_tensor[:,:,j])
            rmsd_arr[i,j] = rmsd_arr[j,i] = rmsd
            
    return rmsd_arr


#def cluster(rmsd_arr, min_cluster_size=3):
#    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
#    clusterer.fit(rmsd_arr)
#
#    return clusterer.labels_, clusterer.probabilities_


def visualise_clusters(coord_tensor, labels, best_fit_files):


    test = np.dstack([  coord_tensor[:,:,labels==0], coord_tensor[:,:,labels==1],
                        coord_tensor[:,:,labels==2], coord_tensor[:,:,labels==3],
                        coord_tensor[:,:,labels==-1]])
    
    rmsd_arr_t = coord_tensor_pairwise_rmsd(test)

    cluster_cumsum = np.cumsum( np.asarray( [sum(labels==0), sum(labels==1), sum(labels==2),

                                         sum(labels==3), sum(labels==-1)] ) )

    cluster_cumsum = np.insert(cluster_cumsum, 0, 0)

    bf_names = []
    for bf in best_fit_files:
        bf_names.append(bf.split('/')[-2]+'/'+bf.split('/')[-1][:-4])

    bf_names = np.asarray(bf_names)
    bf_names_sort = np.concatenate([bf_names[labels==0], bf_names[labels==1], bf_names[labels==2], bf_names[labels==3], bf_names[labels==-1]])
        
    hovertext = list()
    for yi, yy in enumerate(bf_names_sort):
        hovertext.append(list())
        for xi, xx in enumerate(bf_names_sort):
            hovertext[-1].append('Structure X: {}<br />Structure Y: {}<br /> RMSD: {}'.format(xx, yy, round(rmsd_arr_t[xi, yi],1)))
            
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=rmsd_arr_t, colorscale='RdBu_r', text=hovertext, hoverinfo='text'))

    for v in cluster_cumsum:

        fig.add_hline(y=v-0.5, line_width=2, line_color="black")
        fig.add_vline(x=v-0.5, line_width=2, line_color="black")

    fig.update_layout(height=700, width=650)
    return fig, bf_names_sort



def overlay_coords(tensor):
    
    fig = go.Figure()

    for i in range(tensor.shape[-1]):
#         print(i)
        fig.add_trace(go.Scatter3d(x=tensor[:,0,i], y=tensor[:,1,i], z=tensor[:,2,i], opacity=0.4,
                               mode='lines',
                               line=dict( width=12,
#                                          color='red',
#                                          color=np.arange(tensor[:,0,i].shape[0]), 
                                         colorscale='greys')))
    
    
    av_coords = tensor.mean(axis=2)
    
    fig.add_trace(go.Scatter3d(x=av_coords[:,0], y=av_coords[:,1], z=av_coords[:,2], opacity=0.7,
                               mode='lines',
                               line=dict( width=12,
                                         color='black',
#                                          color=np.arange(tensor[:,0,i].shape[0]), 
                                         colorscale='greys')))
        
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

    fig.update_layout(width=800, height=800)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_traces(showlegend=False)
    return fig
