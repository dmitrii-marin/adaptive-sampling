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

# Set up the working directories.
PASCAL_FOLDER="cityscapes"
EXP_FOLDER="exp/train_on_train_set_mobilenetv2_w_nus_at_${SAMPLING_SIZE}"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/export"

if [ -z ${UNIFORM} ]; then
  NUS_PREPROCES_TYPE="net"
else
  NUS_PREPROCES_TYPE="uniform"
  TRAIN_LOGDIR="${TRAIN_LOGDIR}_${NUS_PREPROCES_TYPE}"
  EVAL_LOGDIR="${EVAL_LOGDIR}_${NUS_PREPROCES_TYPE}"
  VIS_LOGDIR="${VIS_LOGDIR}_${NUS_PREPROCES_TYPE}"
  EXPORT_DIR="${EXPORT_DIR}_${NUS_PREPROCES_TYPE}"
fi

mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

if [ -z ${MODE} ]; then MODE=TRAIN; fi

if [ ${MODE} == TRAIN ]; then
  # Copy locally the trained checkpoint as the initial checkpoint.
  # https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_128.tgz
  TF_INIT_ROOT="https://storage.googleapis.com/mobilenet_v2/checkpoints"
  #CKPT_NAME="deeplabv3_mnv2i_pascal_train_aug"
  CKPT_NAME="mobilenet_v2_1.0_${SAMPLING_SIZE}"
  TF_INIT_CKPT="${CKPT_NAME}.tgz"
  #TF_INIT_CKPT="${CKPT_NAME}_2018_01_29.tar.gz"
  cd "${INIT_FOLDER}"
  wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
  tar -xf "${TF_INIT_CKPT}"
  cd "${CURRENT_DIR}"
fi

PASCAL_DATASET="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/tfrecord"

NUS_CHECKPOINT="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/exp/nus64_128_mobilenetv2_lr0.007_mt0.25/train/model.ckpt-30000"



if [ ${MODE} == "TRAIN" ]; then
  # Train 10 iterations.
  NUM_ITERATIONS=90000
  python "${WORK_DIR}"/train.py \
    --logtostderr \
    --dataset="cityscapes" \
    --train_split="train" \
    --model_variant="mobilenet_v2" \
    --base_learning_rate=0.007 \
    --nus_preprocess="${NUS_PREPROCES_TYPE}" \
    --nus_net_input_size=${AUX_INPUT_SIZE} \
    --nus_net_stride=${STRIDE} \
    --nus_sampling_size=${SAMPLING_SIZE} \
    --nus_depth_multiplier=${DEPTH_MULTIPLIER} \
    --nus_checkpoint="${NUS_CHECKPOINT}" \
    --min_scale_factor=1 \
    --max_scale_factor=1 \
    --train_batch_size=16 \
    --train_crop_size="1024,1024" \
    --training_number_of_steps="${NUM_ITERATIONS}" \
    --fine_tune_batch_norm=true \
    --train_logdir="${TRAIN_LOGDIR}" \
    --dataset_dir="${PASCAL_DATASET}" \
    --tf_initial_checkpoint="${INIT_FOLDER}/${CKPT_NAME}.ckpt" \
    --save_summaries_images=True \

 elif [ ${MODE} == "EVAL" ]; then

   python "${WORK_DIR}"/eval.py \
     --logtostderr \
     --dataset="cityscapes" \
     --eval_split="val" \
     --model_variant="mobilenet_v2" \
     --eval_crop_size="1024,1024" \
     --output_stride=8 \
     --checkpoint_dir="${TRAIN_LOGDIR}" \
     --eval_logdir="${EVAL_LOGDIR}" \
     --dataset_dir="${PASCAL_DATASET}" \
     --max_number_of_evaluations=-1 \
     --nus_preprocess="${NUS_PREPROCES_TYPE}" \
     --nus_net_input_size=${AUX_INPUT_SIZE} \
     --nus_net_stride=${STRIDE} \
     --nus_sampling_size=${SAMPLING_SIZE} \
     --nus_depth_multiplier=${DEPTH_MULTIPLIER} \
     --nus_checkpoint="${NUS_CHECKPOINT}" \

 fi
