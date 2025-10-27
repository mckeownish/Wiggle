# writhe_cython.pyx

cimport numpy as cnp
import numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport asin, sqrt


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
cpdef cnp.ndarray[cnp.float64_t, ndim=2] find_Sigma_array(cnp.ndarray[cnp.float64_t, ndim=2] curve_points):

    """
    Computes the pairwise entanglement (writhe ~ Sigma values) for each pair of segments in a discrete curve input.

    Parameters:
    curve_points (numpy.ndarray): Array of coordinates describing a discrete curve.

    Returns:
    numpy.ndarray: A 2D array representing the entanglement writhe for each pair of segments in a discrete curve.
    """

    # number of segments is number of XYZ points -1
#   # eg. *-*-*-*, #(*)=4, #(-)=3

    cdef int segment_num = curve_points.shape[0]-1
    cdef cnp.ndarray[cnp.float64_t, ndim=2] Sigma_array = np.zeros([segment_num, segment_num], dtype=np.float64)
    cdef int i, j
    cdef cnp.ndarray[cnp.float64_t, ndim=1] p1, p2, p3, p4
    
    for i in range(1,curve_points.shape[0]-1):

        p1 = curve_points[i,:]
        p2 = curve_points[i+1,:]
        
        for j in range(0, i-1):

            p3 = curve_points[j,:]
            p4 = curve_points[j+1,:]

            Sigma_array[i,j] = Gauss_Int_4_segment(p1, p2, p3, p4)
            
    return 2*Sigma_array



# --- PARALLEL_VERSION ---


# GIL-free helper functions for parallel computation
cdef inline double dot_product_nogil(double* u1, double* u2) nogil:
    return u1[0]*u2[0] + u1[1]*u2[1] + u1[2]*u2[2]

cdef inline void cross_product_nogil(double* u1, double* u2, double* result) nogil:
    result[0] = u1[1]*u2[2] - u1[2]*u2[1]
    result[1] = u1[2]*u2[0] - u1[0]*u2[2]
    result[2] = u1[0]*u2[1] - u1[1]*u2[0]

cdef inline void unit_vec_from_cross_nogil(double* u1, double* u2, double* result) nogil:
    cdef double norm
    cdef int i
    
    cross_product_nogil(u1, u2, result)
    norm = sqrt(result[0]*result[0] + result[1]*result[1] + result[2]*result[2])
    
    if norm > 0:
        for i in range(3):
            result[i] /= norm
    else:
        for i in range(3):
            result[i] = 0.0

cdef inline double triple_product_nogil(double* a, double* b, double* c) nogil:
    cdef double cross[3]
    cross_product_nogil(a, b, cross)
    return dot_product_nogil(cross, c)

cdef double compute_gauss_int_nogil(float* p1, float* p2, float* p3, float* p4) nogil:
    cdef double r_12[3], r_13[3], r_14[3], r_23[3], r_24[3], r_34[3]
    cdef double n1[3], n2[3], n3[3], n4[3]
    cdef double Sigma_star, sign
    cdef int k
    
    for k in range(3):
        r_12[k] = <double>p2[k] - <double>p1[k]
        r_13[k] = <double>p3[k] - <double>p1[k]
        r_14[k] = <double>p4[k] - <double>p1[k]
        r_23[k] = <double>p3[k] - <double>p2[k]
        r_24[k] = <double>p4[k] - <double>p2[k]
        r_34[k] = <double>p4[k] - <double>p3[k]
    
    unit_vec_from_cross_nogil(r_13, r_14, n1)
    unit_vec_from_cross_nogil(r_14, r_24, n2)
    unit_vec_from_cross_nogil(r_24, r_23, n3)
    unit_vec_from_cross_nogil(r_23, r_13, n4)
    
    Sigma_star = (asin(dot_product_nogil(n1, n2)) + 
                  asin(dot_product_nogil(n2, n3)) +
                  asin(dot_product_nogil(n3, n4)) + 
                  asin(dot_product_nogil(n4, n1)))
    
    sign = 1.0 if triple_product_nogil(r_34, r_12, r_13) > 0 else -1.0
    
    return 0.07957747154594767 * Sigma_star * sign

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.float32_t, ndim=3] find_Sigma_array_batch_parallel(
    cnp.ndarray[cnp.float32_t, ndim=3] trajectory,
    int num_threads=1
):
    cdef int n_frames = trajectory.shape[0]
    cdef int n_atoms = trajectory.shape[1]
    cdef int segment_num = n_atoms - 1
    
    cdef cnp.ndarray[cnp.float32_t, ndim=3] results = np.zeros(
        (n_frames, segment_num, segment_num), dtype=np.float32
    )
    
    cdef int frame_idx, i, j
    cdef float[:, :, ::1] traj_view = trajectory
    cdef float[:, :, ::1] results_view = results
    
    with nogil:
        for frame_idx in prange(n_frames, num_threads=num_threads, schedule='dynamic'):
            for i in range(1, n_atoms - 1):
                for j in range(0, i-1):
                    results_view[frame_idx, i, j] = <float>compute_gauss_int_nogil(
                        &traj_view[frame_idx, i, 0],
                        &traj_view[frame_idx, i+1, 0],
                        &traj_view[frame_idx, j, 0],
                        &traj_view[frame_idx, j+1, 0]
                    )
            
            for i in range(segment_num):
                for j in range(segment_num):
                    results_view[frame_idx, i, j] *= 2.0
    
    return results

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