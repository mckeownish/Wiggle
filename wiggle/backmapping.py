from wiggle.CarbonaraDataTools import Carbonara_2_PDB
from wiggle.CA_2_AA import CA2AA
import os


def backmap_ca_chain(coords_file, fingerprint_file, write_directory, name):

	# write the CA chain into pdb format - note this won't work if non-standard residues are present!
	ca_pdb_output_name = os.path.join(write_directory, name+'_CA.pdb')
	Carbonara_2_PDB(coords_file, fingerprint_file, ca_pdb_output_name)
	print('Alpha Coordinates pdb written to: ', ca_pdb_output_name)

	aa_pdb_output_name = os.path.join(write_directory, name+'_AA.pdb')
	CA2AA(ca_pdb_output_name, aa_pdb_output_name, iterations=3, stout=False)
	print('All Atomistic pdb written to: ', aa_pdb_output_name)

