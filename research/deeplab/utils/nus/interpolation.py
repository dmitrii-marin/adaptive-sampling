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


def nus_interpolate(data, sampling, shape):
    assert len(data.shape) == 3, \
       "nus_interpolate: only support 3D data arrays, got shape %s" % list(data.shape)
    channels = data.shape[-1]
    points = sampling.reshape((-1, 2)) * (np.array(shape).T - 1)
    values = data.reshape((-1, channels))
    xi = np.stack(
        np.mgrid[:shape[0], :shape[1]],
        axis=-1,
    ).reshape((-1, 2))
    output_shape = tuple(shape.astype(int)) + (channels,)
    return scipy.interpolate.griddata(
        points,
        values,
        xi,
    ).astype(data.dtype).reshape(output_shape)
