import numpy as np
import pandas as pd
from matplotlib import cm
import nglview as nv


def make_color_matrix(residue_values, color_map='Reds'):

    # Create the color matrix using the 'viridis' colormap
    cmap = cm.get_cmap(color_map)
    color_matrix = np.empty(residue_values.shape[0], dtype=object)

    for i, cont in enumerate(residue_values):
        color = cmap(cont)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), int(color[1]*255), int(color[2]*255))
        color_matrix[i] = hex_color

    return color_matrix


def make_color_df(color_matrix):

    # color matrix needs to be same length as number of residues! 
    residues_indices = np.arange(1, len(color_matrix)+1)

    residue_df = pd.DataFrame({'residue_id':residues_indices, 'color':color_matrix})
    residue_df['residue_id_str'] = residue_df['residue_id'].astype(str)

    return residue_df


def assign_color_scheme(residue_df):

    color_residue_map = residue_df[["color", "residue_id_str"]].to_numpy().tolist()
    color_scheme = nv.color._ColorScheme(color_residue_map, label="#interactions")

    return color_scheme


def get_color_scheme(residue_values, color_map='Reds'):

    color_matrix = make_color_matrix(residue_values, color_map)
    residue_df = make_color_df(color_matrix)
    color_scheme = assign_color_scheme(residue_df)

    return color_scheme

