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


def normalize_locations(locations):
    locations = np.copy(locations)
    locations[..., 0, :, 0] = 0
    locations[..., -1, :, 0] = 1
    locations[..., :, 0, 1] = 0
    locations[..., :, -1, 1] = 1
    return np.clip(locations, 0, 1)


def rel_to_abs(size, locations):
    size = np.array(size, np.float32)
    return np.round(locations * (size - 1)).astype(np.int32)


def sample(image, locations):
    return image[
        locations[..., 0],
        locations[..., 1],
        :
    ]


def sample_rel_with_proj(image, locations):
    return sample(
        image,
        rel_to_abs(
            image.shape[:2],
            normalize_locations(locations),
        ),
    )
