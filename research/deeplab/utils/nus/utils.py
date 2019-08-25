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

import numpy as np
import scipy
from .build.closest_point_cython import closest_boundary_point as regularize_locations

def get_edges(arr, ignore_label=None, target_labels=None):
    edges = np.zeros_like(arr) != 0

    edges[:,1:]  = np.logical_or(edges[:,1:],  arr[:,1:] != arr[:,:-1])
    edges[:,:-1] = np.logical_or(edges[:,:-1], arr[:,1:] != arr[:,:-1])

    edges[1:,:]  = np.logical_or(edges[1:,:],  arr[1:,:] != arr[:-1,:])
    edges[:-1,:] = np.logical_or(edges[:-1,:], arr[1:,:] != arr[:-1,:])

    if ignore_label is not None:
        edges[arr == ignore_label] = False

    if target_labels is not None:
        mask = np.isin(np.array(arr), target_labels)
        edges[np.logical_not(mask)] = False

    return edges


def get_near_boundary_sampling_locations(edges, size, alpha=0.5):
    dist, closest = scipy.ndimage.morphology.distance_transform_edt(
        np.logical_not(edges),
        return_indices=True,
    )
    closest = closest.astype(np.float) / (np.array(edges.shape[:2]) - 1)[:,None,None]
    closest = np.transpose(closest, [1, 2, 0])
    i, j = np.mgrid[:closest.shape[0]-1:1j*size[0], :closest.shape[0]-1:1j*size[1]].astype(int)
    targets = closest[i, j]
    return regularize_locations(targets, alpha).astype(np.float32)
