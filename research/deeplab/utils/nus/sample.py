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
