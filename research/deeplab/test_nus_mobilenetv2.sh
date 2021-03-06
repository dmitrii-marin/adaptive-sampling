#!/bin/bash
# Modified by Dmitrii Marin, https://github.com/dmitrii-marin
#
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
#
# This script is used to run local test on PASCAL VOC 2012 using MobileNet-v2.
# Users could also modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test_mobilenetv2.sh
#
#

source "nus_common.sh"

PASCAL_FOLDER="cityscapes"
EXP_FOLDER="exp/nus${AUX_INPUT_SIZE}_128_mobilenetv2_lr${LEARNING_RATE}_mt${DEPTH_MULTIPLIER}"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"
#CKPT_NAME="deeplabv3_mnv2i_pascal_train_aug"
CKPT_NAME="deeplabv3_mnv2_cityscapes_train"
TF_INIT_CKPT="${CKPT_NAME}_2018_02_05.tar.gz"
#TF_INIT_CKPT="${CKPT_NAME}_2018_01_29.tar.gz"
cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

PASCAL_DATASET="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/tfrecord"


# Train 10 iterations.
NUM_ITERATIONS=30000
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --dataset="cityscapes" \
  --train_split="train" \
  --base_learning_rate=${LEARNING_RATE} \
  --model_variant="mobilenet_v2" \
  --nus_type="net" \
  --nus_net_input_size=${AUX_INPUT_SIZE} \
  --nus_net_stride=${STRIDE} \
  --nus_sampling_size=${SAMPLING_SIZE} \
  --nus_depth_multiplier=${DEPTH_MULTIPLIER} \
  --nus_train=True \
  --max_resize_value=${AUX_INPUT_SIZE} \
  --min_resize_value=${AUX_INPUT_SIZE} \
  --nus_target_classes="11,12,13,14,15,16,17,18" \
  --min_scale_factor=1 \
  --max_scale_factor=1 \
  --train_batch_size=32 \
  --train_crop_size="1024,1024" \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${PASCAL_DATASET}" \
  --save_summaries_images=True

