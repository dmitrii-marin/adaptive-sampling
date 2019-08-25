# Copyright 2019 Dmitrii Marin (https://github.com/dmitrii-marin) All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
