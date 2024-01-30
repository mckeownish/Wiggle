# Wiggle

This repo is a collection of packages/functions used in my e-lab _No Loose Ends_, where I explore a subfield of mathematics known as knot theory, specifically metrics of self entanglement such as the writhe. This metric has lots of potential as a geometrical representation of curves used in solving real world problems such as proteins folding.

## Sub Modules

### writhe
An efficient cython implementation of the writhe (self engtanglement) metric between segments on a curve.
Additionally implements an efficient pairwise 2D writhe for all curve segment pairs.

### writhe_evo
Set of helper function for working with the writhe metric, includes visualisation, trajectory analysis and simple comparisons methods.

### CarbonaraDataTools
This module provides functionalities for processing specific data inputs/outputs related
to Carbonara (SAXS protein), such as extracting coordinates, secondary structures, sequences, and
generating PDB files for protein structure analysis. It encompasses tools for inferring
geometric CB positions and preparing data for structure AA reconstruction and analysis.

### CA_2_AA
Reconstruct an approximation of the full atomistic structure (side chains + full backbone) from the alpha carbon backbone. Outputs in the pdb format. Reconstruction performed with MODELLER and involves energy minimisation of backbone conformation - don't be surprised if the CA positions change (relax) a little! 
