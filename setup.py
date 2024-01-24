from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

# Define your Cython extensions
extensions = [
    Extension("wiggle.writhe", ["cython_src/writhe_c.pyx"]),
    # Add other extensions if any
    ]
    

# Use cythonize on the extensions
setup(
    name='wiggle',
    version='0.1',
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
    install_requires=[
        'numpy',
        'scipy',
        'plotly',
        'pandas',
        'biobox',
        'biopython',
        'tqdm'
        
    ],
    author='Josh McKeown',  # Add author information
    author_email = 'josh.j.mckeown@durham.ac.uk',
    description = 'Package used in my No Loose Ends Lab, where I explore a subfield of mathematics known as knot theory, specifically a metric of self entanglement referred to as the writhe. Im looking into its potential as a geometrical representation to help solve real world problems such as proteins folding!'
)
