# writhe_cython.pyx

cimport numpy as cnp
import numpy as np
cimport cython
from cython.parallel import prange


from libc.math cimport asin, sqrt

# --------------------------
# Helper Functions
# --------------------------

# efficient 3D dot product
cdef double dot_product(cnp.ndarray[cnp.float64_t, ndim=1] u1, cnp.ndarray[cnp.float64_t, ndim=1] u2):

    """
    Computes the dot product of two 3-dimensional vectors.

    Parameters:
    u1 (numpy.ndarray): First vector.
    u2 (numpy.ndarray): Second vector.

    Returns:
    double: The dot product of the two vectors.
    """

    cdef double dot = 0.0
    cdef int i

    for i in range(3):
        dot += u1[i] * u2[i]

    return dot


# efficient 3D cross prod 
cdef cnp.ndarray[cnp.float64_t, ndim=1] cross_product(cnp.ndarray[cnp.float64_t, ndim=1] u1, cnp.ndarray[cnp.float64_t, ndim=1] u2):

    """
    Computes the cross product of two 3-dimensional vectors.

    Parameters:
    u1 (numpy.ndarray): First vector.
    u2 (numpy.ndarray): Second vector.

    Returns:
    numpy.ndarray: The cross product of the two vectors.
    """

    cdef cnp.ndarray[cnp.float64_t, ndim=1] cross = np.empty(3, dtype=np.float64)

    cross[0] = u1[1]*u2[2] - u1[2]*u2[1]
    cross[1] = u1[2]*u2[0] - u1[0]*u2[2]
    cross[2] = u1[0]*u2[1] - u1[1]*u2[0]
    
    return cross


# efficient unit vectors
cdef cnp.ndarray[cnp.float64_t, ndim=1] unit_vec(cnp.ndarray[cnp.float64_t, ndim=1] u1, cnp.ndarray[cnp.float64_t, ndim=1] u2):

    """
    Computes the unit vector for the cross product of two 3-dimensional vectors.

    Parameters:
    u1 (numpy.ndarray): First vector.
    u2 (numpy.ndarray): Second vector.

    Returns:
    numpy.ndarray: Unit vector of the cross product.
    """

    cdef cnp.ndarray[cnp.float64_t, ndim=1] cross = cross_product(u1, u2)
    cdef double norm
    cdef int i

    norm = 0.0
    for i in range(3):
        norm += cross[i]*cross[i]
    norm = sqrt(norm)

    for i in range(3):
        cross[i] /= norm

    return cross


# --------------------------
# Main Functions
# --------------------------

# Writhe for segment pair
cdef double Gauss_Int_4_segment(cnp.ndarray[cnp.float64_t, ndim=1] p1, 
                                cnp.ndarray[cnp.float64_t, ndim=1] p2,
                                cnp.ndarray[cnp.float64_t, ndim=1] p3,
                                cnp.ndarray[cnp.float64_t, ndim=1] p4):

    """
    Computes the geometrical entanglement (writhe) between any 2 given segments r1 and r2 
    { r1 = [p1->p2], r2 = [p3->p4] }

    More info on discrete writhe numerical implementation: https://en.wikipedia.org/wiki/Writhe

    Parameters:
    p1, p2, p3, p4 (numpy.ndarray): XYZ points defining two segments of a discrete curve.

    Returns:
    double: The entanglement (writhe) between two segments.
    """

    cdef cnp.ndarray[cnp.float64_t, ndim=1] r_12 = p2 - p1
    cdef cnp.ndarray[cnp.float64_t, ndim=1] r_13 = p3 - p1
    cdef cnp.ndarray[cnp.float64_t, ndim=1] r_14 = p4 - p1
    cdef cnp.ndarray[cnp.float64_t, ndim=1] r_23 = p3 - p2
    cdef cnp.ndarray[cnp.float64_t, ndim=1] r_24 = p4 - p2
    cdef cnp.ndarray[cnp.float64_t, ndim=1] r_34 = p4 - p3

    cdef cnp.ndarray[cnp.float64_t, ndim=1] n1 = unit_vec(r_13, r_14)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] n2 = unit_vec(r_14, r_24)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] n3 = unit_vec(r_24, r_23)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] n4 = unit_vec(r_23, r_13)

    cdef double Sigma_star = asin(dot_product(n1,n2)) + asin(dot_product(n2,n3)) \
               + asin(dot_product(n3,n4)) + asin(dot_product(n4,n1))
    
    return 1/(4*3.141592653589793) * Sigma_star * (1 if dot_product(cross_product(r_34, r_12), r_13) > 0 else -1)


# Matrix of Writhes for all segment pairs in curve (symmetric along diagonal)
cpdef cnp.ndarray[cnp.float64_t, ndim=2] find_Sigma_array(cnp.ndarray[cnp.float64_t, ndim=2] segments):

    """
    Computes the pairwise entanglement (writhe ~ Sigma values) for each pair of segments in a discrete curve input.

    Parameters:
    segments (numpy.ndarray): Array of coordinates a discrete curve.

    Returns:
    numpy.ndarray: A 2D array representing the entanglement writhe for each pair of segments in a discrete curve.
    """

    cdef int size = segments.shape[0]-1
    cdef cnp.ndarray[cnp.float64_t, ndim=2] Sigma_array = np.zeros([size, size], dtype=np.float64)
    cdef int i, j
    cdef cnp.ndarray[cnp.float64_t, ndim=1] p1, p2, p3, p4
    
    for i in range(1,segments.shape[0]-1):

        p1 = segments[i,:]
        p2 = segments[i+1,:]
        
        for j in range(0, i-1):

            p3 = segments[j,:]
            p4 = segments[j+1,:]

            Sigma_array[i,j] = Gauss_Int_4_segment(p1, p2, p3, p4)
            
    return 2*Sigma_array


# find the writhe fingerprint - a varying sum metric 

# cpdef cnp.ndarray[cnp.float64_t, ndim=2] writhe_fingerprint(cnp.ndarray[cnp.float64_t, ndim=2] segments):
    
#     cdef cnp.ndarray[cnp.float64_t, ndim=2] Sigma_array = find_Sigma_array(segments)
    
#     cdef int sp_size = Sigma_array.shape[0]
#     cdef int ep_size = Sigma_array.shape[1]
    
#     cdef cnp.ndarray[cnp.float64_t, ndim=2] matrix_nres = np.full([sp_size, ep_size], np.nan)
    
#     cdef int sp, ep

#     for sp in prange(sp_size, nogil=True):
#         for ep in prange(sp, ep_size, nogil=True):
#             matrix_nres[ep,sp] = np.sum(Sigma_array[sp:ep,sp:ep])
            
#     matrix_nres = matrix_nres[3:,:-3]
#     cdef int a, b
#     a, b = np.triu_indices(matrix_nres.shape[1],k=1)
#     matrix_nres[a,b] = np.nan   
    
#     return matrix_nres