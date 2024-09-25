import pandas as pd
import numpy as np

def import_STRIDE_traj(file_path):

    ss = np.genfromtxt(file_path, dtype='str')
    df = pd.DataFrame(ss, columns=['resnum', 'chain', '__', 'frame', 'SecondaryStruct'])

    mapping_ss = {'C':1, 'T':1, 'B':2, 'E':2, 'G':3, 'H':3}
    mapping_ss_hsl = {'C':'-', 'T':'-', 'B':'S', 'E':'S', 'G':'H', 'H':'H'}

    df['SecondaryStruct_num'] = df['SecondaryStruct'].map(mapping_ss)
    df['SecondaryStruct_reduced'] = df['SecondaryStruct'].map(mapping_ss_hsl)

    # Get the number of unique frames and residues
    n_frames = df['frame'].nunique()
    n_residues = df['resnum'].nunique()

    # Reshape the columns
    ss_num = df['SecondaryStruct_num'].values.reshape(n_frames, n_residues).T
    ss_str = df['SecondaryStruct'].values.reshape(n_frames, n_residues).T
    ss_str_reduced = df['SecondaryStruct_reduced'].values.reshape(n_frames, n_residues).T

    return ss_num, ss_str, ss_str_reduced



def import_STRIDE_single(file_path):

    ss_str = np.genfromtxt(file_path, dtype='str')
    ss_str = ss_str[:, -1]

    mapping_ss = {'C':1, 'T':1, 'B':2, 'E':2, 'G':3, 'H':3}
    mapping_ss_hsl = {'C':'-', 'T':'-', 'B':'S', 'E':'S', 'G':'H', 'H':'H'}

    ss_num = [mapping_ss[s] for s in ss_str]
    ss_num = np.array(ss_num)

    ss_reduced = [mapping_ss_hsl[s] for s in ss_str]
    ss_reduced = np.array(ss_reduced)

    return ss_num, ss_str, ss_reduced
