
This repo is a collection of packages/functions used in my _No Loose Ends Lab_, where I explore a subfield of mathematics known as knot theory, specifically metrics of self entanglement such as the writhe. This metric has lots potential as a geometrical representation for solving real world problems such as proteins folding.

*Sub Modules in wiggle*

Writhe
An efficient cython implementation of the writhe (self engtanglement) metric between segments on a curve.
Additionally implements an efficient pairwise 2D writhe for all curve segment pairs.

Carbonara Data Tools
This module provides functionalities for processing specific data inputs/outputs related
to Carbonara (SAXS protein), such as extracting coordinates, secondary structures, sequences, and
generating PDB files for protein structure analysis. It encompasses tools for inferring
geometric CB positions and preparing data for structure AA reconstruction and analysis.