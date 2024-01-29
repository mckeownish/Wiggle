import numpy as np
from tqdm import tqdm
import biobox as bb

import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl

from wiggle.writhe import find_Sigma_array

# from writhe_c import find_Sigma_array


#def unit_vec(u1, u2):
#
#    cross = np.cross(u1, u2)
#    return np.divide( cross, np.linalg.norm(cross) )
#
#
#def points_2_vectors(p1, p2, p3, p4):
#
#    r_12 = p2 - p1
#    r_13 = p3 - p1
#    r_14 = p4 - p1
#
#    r_23 = p3 - p2
#    r_24 = p4 - p2
#    r_34 = p4 - p3
#
#    return r_12, r_13, r_14, r_23, r_24, r_34
#
#
#def Gauss_Int_4_segment(p1, p2, p3, p4):
#
#    r_12, r_13, r_14, r_23, r_24, r_34 = points_2_vectors(p1, p2, p3, p4)
#
#    n1 = unit_vec(r_13, r_14)
#    n2 = unit_vec(r_14, r_24)
#    n3 = unit_vec(r_24, r_23)
#    n4 = unit_vec(r_23, r_13)
#
#    Sigma_star = np.arcsin(np.dot(n1,n2)) + np.arcsin(np.dot(n2,n3)) \
#               + np.arcsin(np.dot(n3,n4)) + np.arcsin(np.dot(n4,n1))
#
#    return 1/(4*np.pi) * Sigma_star * np.sign( np.dot( np.cross(r_34, r_12), r_13 ) )
#
#
#def find_Sigma_array(curve_points):
#
#    # number of segments is number of XYZ points -1
#    # eg. *-*-*-*, #(*)=4, #(-)=3
#    segment_num = int( curve_points.shape[0] - 1)
#
#    # Initialising the Sigma array, segment pair-wise writhe values
#    Sigma_array = np.zeros([segment_num, segment_num])
#
#    # i = end point, j < i
#    # starting at 2, since i,i = i,i+1 = 0
#    # (self and adjacent segments have zero writhe)
#    for i in range(2, segment_num):
#
#        p1 = curve_points[i,:]
#        p2 = curve_points[i+1,:]
#
#
#        for j in np.arange(0, i-1):
#
#            p3 = curve_points[j,:]
#            p4 = curve_points[j+1,:]
#
#            Sigma_array[i,j] = Gauss_Int_4_segment(p1, p2, p3, p4)
#
#    # x2 to account for double contribution (symmetric argument)
#    return 2*Sigma_array
#
#    # trim the top 2 rows and right 2 columns due to segments: i,i = i,i+1 = 0
#    # resulting square matrix: (N-3)x(N-3), N = number of points in curve
##    return 2*Sigma_array[2:,:-2]


def calculate_segment_to_all_from_points(curve_points):

    Sigma_array = find_Sigma_array(curve_points)
    segment_number = Sigma_array.shape[0]

    # segment to all writhe
    StA = np.zeros(segment_number)

    for i in range(segment_number):

        # sum column n + row n
        StA[i] = sum(Sigma_array[:,i]+Sigma_array[i,:])
    
    return StA
    
def calculate_segment_to_all_from_sigma(Sigma_array):
    
    segment_number = Sigma_array.shape[0]
 
    # segment to all writhe
    StA = np.zeros(segment_number)

    for i in range(segment_number):

        # sum column n + row n
        StA[i] = sum(Sigma_array[:,i]+Sigma_array[i,:])
    
    return StA


def writhe_heatmap_plotly(writhe_arr):

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=writhe_arr,
                             colorscale='RdBu_r',
                             # color_continuous_midpoint=0, color_continuous_scale='RdBu_r',
                             colorbar = dict(title='Writhe')))
    fig.update_layout(height=600, template='simple_white')
   
    return fig


def writhe_heatmap_plot_non_linear_color(Sigma_array, title='Colour Scale: Distribution'):

    Sigma_array[np.triu_indices(Sigma_array.shape[0], -1)] = np.nan
    x = np.ravel(Sigma_array)
    x = x[~np.isnan(x)]

    sorted_x = np.sort(x)
    
    # index of 1%, 2%, ... 99% point in sorted x values
    deci_indices = np.floor(np.arange(1,100)*sorted_x.shape[0]/100)
    deci_vals = sorted_x[deci_indices.astype(int)]
    
    # Whatever happens, jappens - no monkey business moving middles values around
    # bounds = deci_vals

    # Shift entire s-curve so middle value is zero - not sure if this is okay to do, but I'm god here so what I say goes!
    bounds = deci_vals - deci_vals[49]

    # Just a cheeky check that we havent fucked the top / bottom end - as we append the first + last val of x as the bin edges
    assert sorted_x[-1] - sorted_x[-2] > deci_vals[49], 'Shift to renormalise color scale larger than diff between last 2 vals'
    assert sorted_x[1] - sorted_x[0] > deci_vals[49], 'Shift to renormalise color scale larger than diff between first 2 vals'
    
    bounds = np.append(sorted_x[0], bounds)
    bounds = np.append(bounds, sorted_x[-1])
    
    mpl.rcParams['figure.dpi'] = 300
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), layout='constrained')
    
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    pcm = ax.imshow(Sigma_array, norm=norm, cmap='RdBu_r')
    fig.colorbar(pcm, ax=ax, extend='both', orientation='vertical')
    ax.set_title(title)

    return fig


def writhe_diff(writhe_arr_X, writhe_arr_Y):
   
    wr_diff = writhe_arr_X - writhe_arr_Y
    gw = wr_diff[:,0]
    gw = np.append(gw, 0)
   
    return gw
   
   
def writhe_diff_plotter(writhe_arr_X, writhe_arr_Y):
   
    gw = writhe_diff(writhe_arr_X, writhe_arr_Y)
   
    fig = make_subplots(vertical_spacing=0.1, horizontal_spacing=.01,
                    rows=1, cols=2,
                    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                    subplot_titles=("Structure X", "Structure Y"))


    fig.add_trace(go.Scatter3d(x=writhe_arr_X[:,0],y=writhe_arr_X[:,1], z=writhe_arr_X[:,2], opacity=.9,
                               mode='markers+lines', name='X',
                               line=dict(width=12, color=gw, colorscale='RdBu_r'),
                               marker=dict(size=5, color=np.arange(writhe_arr_X.shape[0]), colorscale='gray', opacity=0.6)),
                  row=1, col=1)

    fig.add_trace(go.Scatter3d(x=writhe_arr_Y[:,0],y=writhe_arr_Y[:,1], z=writhe_arr_Y[:,2], opacity=.9,
                               mode='markers+lines', name='Y',
                               line=dict(width=12, color=gw, colorscale='RdBu_r'),
                               marker=dict(size=5, color=np.arange(writhe_arr_Y.shape[0]), colorscale='gray', opacity=0.6)),
                  row=1, col=2)

    fig.update_layout(
        title_text='Change in writhe',
        height=800,
        width=1600,
        template='simple_white' )

    return fig



def writhe_evolution(traj_file, n, begin_frame, end_frame, step=10):

    u = bb.Molecule(traj_file)
    print("We've read your trajectory...")
    
    # if (begin_frame!=None) & (end_frame!=None):
    
    frame_range = np.arange(u.coordinates.shape[0])[begin_frame:end_frame][::step]


    alpha_conds = u.data['name']=='CA'

    frame_coordinates = u.coordinates[0][alpha_conds][::n]
    Sigma_array = find_Sigma_array(frame_coordinates)
    sp_size = ep_size = Sigma_array.shape[0]

    all_coords = np.zeros([frame_range.shape[0], frame_coordinates.shape[0], 3])

    matrix_nres = np.full([frame_range.shape[0], sp_size, ep_size], np.nan)

    for i, frame in enumerate(tqdm(frame_range)):

        frame_coordinates = u.coordinates[frame][alpha_conds][::n]
        all_coords[i, :, :] = frame_coordinates
        Sigma_array = find_Sigma_array(frame_coordinates)    

        for sp in np.arange(sp_size):
            for ep in np.arange(sp, ep_size):
                matrix_nres[i, ep,sp] = np.sum(Sigma_array[sp:ep,sp:ep])
                
    
    matrix_nres = matrix_nres[:,3:,:-3]
    a, b = np.triu_indices(matrix_nres.shape[1],k=1)
    matrix_nres[:,a,b] = np.nan
    
    return matrix_nres
    
    
def writhe_fingerprint(segments):
    
    Sigma_array = find_Sigma_array(segments)
    
    sp_size = ep_size = Sigma_array.shape[0]
    
    matrix_nres = np.full([sp_size, ep_size], np.nan)
    
    for sp in np.arange(sp_size):
        for ep in np.arange(sp, ep_size):
            matrix_nres[ep,sp] = np.sum(Sigma_array[sp:ep,sp:ep])
            
    matrix_nres = matrix_nres[3:,:-3]
    a, b = np.triu_indices(matrix_nres.shape[1],k=1)
    matrix_nres[a,b] = np.nan   
    
    return matrix_nres


def sigma_evolution(traj_file, n, begin_frame, end_frame, step=10):
    
    
    u = bb.Molecule(traj_file)
    print("We've read your trajectory...")

    alpha_conds = u.data['name']=='CA'
    
    
    frame_range = np.arange(u.coordinates.shape[0])[begin_frame:end_frame][::step]
    
    
    frame_coordinates = u.coordinates[0][alpha_conds][::n]
    Sigma_array = find_Sigma_array(frame_coordinates)    
    sp_size = ep_size = Sigma_array.shape[0]

    all_coords = np.zeros([frame_range.shape[0], frame_coordinates.shape[0], 3])

    matrix_nres = np.full([frame_range.shape[0], sp_size, ep_size], np.nan)

    for i, frame in enumerate(tqdm(frame_range)):

        frame_coordinates = u.coordinates[frame][alpha_conds][::n]
        all_coords[i, :, :] = frame_coordinates
        matrix_nres[i, :, :] = find_Sigma_array(frame_coordinates) 
        
        
    return matrix_nres
