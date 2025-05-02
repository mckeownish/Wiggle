import MDAnalysis as mda
from MDAnalysis.analysis import align
import mdtraj as md

from sklearn.decomposition import PCA
import numpy as np


def align_first_frames_merge(gro_file_list, xtc_files_list, merge_xtc_name):

    # ensure the number of gro and xtc files is the same
    assert len(gro_file_list) == len(xtc_files_list), "Different number of gro and xtc files"

    # ensure merge_xtc_name ends with .xtc
    assert merge_xtc_name.endswith('.xtc'), "The output file name must end with .xtc"

    # ensure merge_xtc_name is not in the list of xtc_files
    assert merge_xtc_name not in xtc_files_list, "The output file name must not be in the list of input xtc files"


    # Load the first trajectory (this will be used as the reference for alignment)
    u_ref = mda.Universe(gro_file_list[0], xtc_files_list[0])

    # Select atoms or group for alignment (e.g., protein backbone)
    selection = u_ref.select_atoms("protein and backbone")

    # Get the first frame from the first trajectory to use as the reference
    reference_frame = u_ref.trajectory[0]  # Load the first frame

    # Prepare the merged trajectory writer
    with mda.Writer(merge_xtc_name, n_atoms=u_ref.atoms.n_atoms) as writer:
        # Iterate over the trajectories to align them to the first frame of the first trajectory
        for gro_file, traj_file in zip(gro_file_list[1:], xtc_files_list[1:]):  # Add more trajectories as needed

            u = mda.Universe(gro_file, traj_file)
            
            # Align the first frame of this trajectory to the first frame of the reference trajectory
            aligner = align.AlignTraj(u, u_ref, select="protein and backbone", ref_frame=0, in_memory=True)
            aligner.run()

            # Write the aligned frames to the merged trajectory
            for ts in u.trajectory:
                writer.write(u.atoms)

def find_PCA_motions(merged_xtc_file, first_frame_file, save_prefix):

    '''saves the characteristic CA chain motions of the first 3 principal components of the protein in PDB format
       Open them in VMD or PyMol (Using a bead representation) to visualize the motions!'''

    #load and prep traj
    traj = md.load(merged_xtc_file, top=first_frame_file)
    ca_atoms = traj.topology.select('name CA')
    ca_traj = traj.xyz[:, ca_atoms, :]
    ca_traj_reshaped = ca_traj.reshape(ca_traj.shape[0], ca_traj.shape[1] * 3)

    #PCA
    pca = PCA(n_components=3)
    pca_traj = pca.fit_transform(ca_traj_reshaped)

    pc_vectors = pca.components_.reshape(3, ca_traj.shape[1], 3)

    # Finding the contribution of each PC to the overall protein dynamics!
    mean_structure = np.mean(ca_traj, axis=0)

    #interpolate between the mean and displaced structures
    
    for i, vec in enumerate(pc_vectors):

        save_pc_motion(vec, f'{save_prefix}_PC{i}', ca_atoms, ca_traj, traj, n_frames=20, scale_factor=10)


#save PC motion in PDB format 
def save_pc_motion(pc_vector, filename, ca_atoms, ca_traj, traj, n_frames=20, scale_factor=10):
    mean_structure = np.mean(ca_traj, axis=0)

    #interpolate between the mean and displaced structures
    frames = []
    ca_topology = traj.top.subset(ca_atoms)

    #forward motion frames
    for i in range(n_frames):
        displacement = scale_factor * ((i / (n_frames - 1)) - 0.5) * pc_vector
        interpolated_structure = mean_structure + displacement
        frames.append(interpolated_structure)

    #backward motion frames 
    for i in range(n_frames - 1, -1, -1):  # Reverse iteration
        frames.append(frames[i])  # Append the reverse frame

    motion_traj = md.Trajectory(np.array(frames), ca_topology)

    motion_traj.save(f'{filename}_motion.pdb')