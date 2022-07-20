# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: c_string_type=unicode, c_string_encoding=utf8
# cython: language_level=3
# cython: infer_types=True
# # cython: initializedcheck=False
# # cython: profile=True
# # cython: linetrace=True
# # cython: binding=True
# # distutils: define_macros=CYTHON_TRACE=1
# # distutils: define_macros=CYTHON_TRACE_NOGIL=1

from libc cimport limits
from libc.stdlib cimport free, malloc
from libcpp.vector cimport vector


cpdef float affineGapDistanceInputOrder(const int[::1]& int_memview_1, const int[::1]& int_memview_2, const int length1, const int length2):
    # suppose len(int_memview_1) <= len(int_memview_2)
    cdef const int * int_ptr_1 = &int_memview_1[0]
    cdef const int * int_ptr_2 = &int_memview_2[0]

    # Initialize C Arrays
    cdef int memory_size = sizeof(float) * (length1+1)
    cdef float *D = <float*> malloc(memory_size)
    cdef float *V_current = <float*> malloc(memory_size)
    cdef float *V_previous = <float*> malloc(memory_size)

    cdef int i, j
    cdef float distance

    # Set up Recurrence relations
    #
    # Base conditions
    # V(0,0) = 0
    # V(0,j) = gapWeight + spaceWeight * i
    # D(0,j) = Infinity
    V_current[0] = 0
    for j in range(1, length1 + 1) :
        V_current[j] = 0.5 + 0.5 * j
        D[j] = limits.INT_MAX

    for i in range(1, length2 +1) :
        V_previous, V_current = V_current, V_previous

        # Base conditions
        # V(i,0) = 0.5 + 0.5 * i
        # I(i,0) = Infinity
        V_current[0] = 0.5 + 0.5 * i
        I = limits.INT_MAX

        for j in range(1, length1+1) :

            # I(i,j) is the edit distance if the jth character of string 1
            # was inserted into string 2.
            I = min(I, V_current[j-1] + 0.5) + 0.5

            # D(i,j) is the edit distance if the ith character of string 2
            # was deleted from string 1
            #
            # D(i,j) = min((i-1,j), V(i-1,j) + 0.5) + 0.5
            D[j] = min(D[j], V_previous[j] + 0.5) + 0.5

            # M(i,j) is the edit distance if the ith and jth characters
            # match or mismatch
            #
            # M(i,j) = V(i-1,j-1) + (matchWeight | misMatchWeight)
            if int_ptr_2[i-1] == int_ptr_1[j-1] :
                M = V_previous[j-1]
            else:
                M = V_previous[j-1] + 1

            # V(i,j) is the minimum edit distance
            #
            # V(i,j) = min(E(i,j), F(i,j), G(i,j))
            V_current[j] = min(I, D[j], M)

    distance = V_current[length1]

    free(D)
    free(V_current)
    free(V_previous)

    return distance


cpdef float affineGapDistance(int[::1] string_a, int[::1] string_b):
    """
    Calculate the affine gap distance between two strings

    Default weights are from Alvaro Monge and Charles Elkan, 1996,
    "The field matching problem: Algorithms and applications"
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.23.9685
    """

    cdef int length1 = string_a.shape[0]
    cdef int length2 = string_b.shape[0]


    if length1 < length2 :
        return affineGapDistanceInputOrder(string_b, string_a,
                              length2,
                              length1)
    else:
        return affineGapDistanceInputOrder(string_a, string_b,
                              length1,
                              length2)


cpdef float normalizedAffineGapDistance(int[::1] string_a, int[::1] string_b) except? 999 :

    cdef int length1 = string_a.shape[0]
    cdef int length2 = string_b.shape[0]

    cdef float normalizer = length1 + length2

    if normalizer == 0:
        raise ZeroDivisionError('normalizedAffineGapDistance cannot take two empty strings')

    cdef float distance = affineGapDistance(string_a, string_b)

    return distance/normalizer

cpdef vector[float] affinaGapDistanceArray(vector[int[::1]] strings):
    len_list_strings = strings.size()
    size = len_list_strings * (len_list_strings - 1) / 2
    cdef vector[float] out = vector[float](size)
    k = 0
    for i in range(len_list_strings):
        for j in range(i+1, len_list_strings):
            out[k] = (affineGapDistance(strings[i], strings[j]))
            k += 1
    return out
