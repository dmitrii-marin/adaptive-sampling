cdef extern from "closest_point.h":
  void closest_point(const double* target, int h, int w, double weight, double* output);

import numpy as np
cimport numpy as np
cimport cython


def closest_boundary_point(np.ndarray[double, ndim=3] target, double weight):
  cdef int h = target.shape[0]
  cdef int w = target.shape[1]
  cdef int c = target.shape[2]
  assert c == 2, "Last dimension must be 2, got %d" % c
  cdef np.ndarray[double, ndim=3, mode="c"] arr = np.ascontiguousarray(target)
  cdef np.ndarray[double, ndim=3, mode="c"] output = np.zeros_like(arr)
  closest_point(<const double*>arr.data, h, w, weight, <double*>output.data)
  return output
