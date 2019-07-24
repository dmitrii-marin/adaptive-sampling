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
