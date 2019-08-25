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

import tensorflow as tf
import numpy as np
from deeplab import common, model
from deeplab.utils import nus
import threading

flags = tf.app.flags
FLAGS = flags.FLAGS

SAMPLING = "sampling"
TARGET_SAMPLING = "target_sampling"

flags.DEFINE_enum('nus_type', None, ['net', 'uniform'], "Non-uniform sampling type")

flags.DEFINE_integer('nus_net_input_size', 32, '')

flags.DEFINE_integer('nus_net_stride', 4, '')

flags.DEFINE_integer('nus_sampling_size', 128, '')

flags.DEFINE_float('nus_depth_multiplier', 1.0, '')

flags.DEFINE_boolean('nus_train', False, '')

flags.DEFINE_enum('nus_preprocess', None, ['net', 'uniform'], '')

flags.DEFINE_list('nus_target_classes', None, '')

flags.DEFINE_float('nus_alpha', 0.5, '')

flags.DEFINE_string('nus_checkpoint', None, '')


def _nus_locations(images, write_summary=True):
    input_size = [FLAGS.nus_net_input_size] * 2
    if input_size != images.get_shape()[1:3]:
        images = tf.image.resize_bilinear(
            images,
            input_size
        )

    model_options = common.ModelOptions(
        outputs_to_num_classes= { SAMPLING: 2 },
        crop_size=input_size,
        # atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.nus_net_stride)
    model_options = model_options._replace(
        depth_multiplier=FLAGS.nus_depth_multiplier)

    with tf.variable_scope("Auxiliary-Network"):
        if FLAGS.nus_train:
            shifts = model._get_logits(
                images,
                model_options=model_options,
                weight_decay=FLAGS.weight_decay,
                is_training=True,
                fine_tune_batch_norm=FLAGS.fine_tune_batch_norm,
            )[SAMPLING]
        else:
            shifts = model._get_logits(
                images,
                model_options=model_options,
                is_training=False,
                fine_tune_batch_norm=False,
            )[SAMPLING]

        tensor_size = FLAGS.nus_net_input_size // FLAGS.nus_net_stride

        uniform_sampling = np.stack(
            np.mgrid[:1:1j*tensor_size, :1:1j*1j*tensor_size],
            axis=-1,
        )

        mean_locations = tf.contrib.framework.model_variable(
            "mean_locations",
            dtype=tf.float32,
            shape=[1, tensor_size, tensor_size, 2],
            initializer=tf.initializers.constant(uniform_sampling),
            trainable=FLAGS.nus_train,
        )
        sampling = shifts + mean_locations
        if write_summary:
            tf.summary.histogram("shifts", shifts)
            tf.summary.histogram("mean_locations", mean_locations)
    return sampling


def get_nus_init():
    assert FLAGS.nus_checkpoint, "Requires a checkpoint"

    exclude_list = ['global_step']
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(
        exclude=exclude_list)
    init_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
        FLAGS.nus_checkpoint,
        variables_to_restore,
        ignore_missing_vars=False)

    def _init_impl(unused_scaffold, sess):
        tf.logging.info("Initializing NUS network from checkpoint")
        sess.run(init_op, init_feed_dict)

    return _init_impl


class Predictor:
    def __init__(self, id):
        self.id = id
        graph = tf.Graph()
        with graph.as_default():
          with tf.device('/cpu:0'):
            self.image = tf.placeholder(
                dtype=tf.float32,
                shape=[1, None, None, 3],
                name="image",
            )
            self.prediction = _nus_locations(self.image, write_summary=False)
            config = tf.ConfigProto(
                device_count = {'GPU': 0},
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1,
            )
            self.sess = tf.Session(config=config)
            get_nus_init()(None, self.sess)

    def __call__(self, image):
        sampling = self.sess.run(
            self.prediction,
            feed_dict={self.image: image}
        )
        return nus.normalize_locations(sampling)

    def __del__(self):
        del self.sess


_predictors = {}
_lock = threading.Lock()
def get_nus_predictor():
    id = threading.current_thread().ident
    if id in _predictors:
        return _predictors[id]
    with _lock:
        if id not in _predictors:
            _predictors[id] = Predictor(id)
            tf.logging.info("Predictor for %s is created, total #%d"
                % (id, len(_predictors)))
    return _predictors[id]


def _nus_uniform_locations():
    tensor_size = FLAGS.nus_sampling_size
    return np.stack(np.mgrid[:1:1j*tensor_size, :1:1j*tensor_size], axis=-1)[None, ...]


def _resize_locations(sampling_location):
    tensor_size = [FLAGS.nus_sampling_size] * 2
    return tf.image.resize_bilinear(
        sampling_location, tensor_size, align_corners=True)


def viz(locations):
    def _np_impl(locations):
        viz = np.zeros((FLAGS.nus_sampling_size, FLAGS.nus_sampling_size), np.float32)
        locations = nus.rel_to_abs(
            viz.shape,
            nus.normalize_locations(locations),
        )
        viz[locations[..., 0], locations[..., 1]] = 1
        return viz

    n = locations.get_shape()[0]
    viz = tf.stack([tf.py_func(
        _np_impl,
        [loc[0]],
        tf.float32,
        name="vizualization"
    ) for loc in tf.split(locations, n)])[..., None]
    viz.set_shape([n, FLAGS.nus_net_input_size, FLAGS.nus_net_input_size, 1])
    return viz


def _nus_sample(samples, sampling_location):
    sampling_location = tf.convert_to_tensor(
        sampling_location, dtype=tf.float32)
    n = samples[common.IMAGE].get_shape()[0]
    if len(sampling_location.get_shape()) == 4:
        split_loc = tf.split(sampling_location, n)
    else:
        split_loc = [sampling_location[None,...]] * n

    def sample(image, locations):
        sampled = tf.py_func(
          nus.sample_rel_with_proj,
          [image, locations],
          image.dtype,
          stateful=False,
          name="nus.sample",
        )
        shape = image.get_shape()
        loc_shape = locations.get_shape()
        new_shape = (loc_shape[0], loc_shape[1], shape[-1])
        sampled.set_shape(new_shape)
        return sampled

    if common.IMAGE in samples:
        with tf.name_scope(None, "NUS-Image-Sample"):
            images = tf.split(samples[common.IMAGE], n)
            samples[common.IMAGE] = tf.stack(
              [sample(image[0], loc[0]) for image, loc in zip(images, split_loc)],
              axis=0,
            )

    if common.LABEL in samples:
        with tf.name_scope(None, "NUS-Label-Sample"):
            labels = tf.split(samples[common.LABEL], n)
            samples[common.LABEL] = tf.stack(
              [sample(label[0], loc[0]) for label, loc in zip(labels, split_loc)],
              axis=0,
    	    )

    return samples


def _get_near_boundary_sampling_locations(label, ignore_label):
    if FLAGS.nus_target_classes is not None:
        target_classes = [int(label) for label in FLAGS.nus_target_classes]
    else:
        target_classes = None
    tensor_size = FLAGS.nus_net_input_size // FLAGS.nus_net_stride
    size = [tensor_size] * 2
    def _impl(label_map):
        edges = nus.get_edges(label_map, ignore_label, target_classes)
        return nus.get_near_boundary_sampling_locations(edges, size, FLAGS.nus_alpha)
    n = label.get_shape()[0]
    with tf.name_scope(None, "near_boundary_sampling_locations"):
        locations = tf.stack([
            tf.py_func(
                _impl,
                [l[0,:,:,0]],
                tf.float32,
                stateful=False,
                name="near_edge_sampling_locations"
            ) for l in tf.split(label, n)
        ])
        locations.set_shape([n] + size + [2])
    return locations
