#!/bin/bash

ENV_NAME="dockbiotic"

CURRENT_DIR=$(dirname "$(readlink -f "$0")")
MODELS_DIR=$CURRENT_DIR/../../saved_models
export DEEPCHEM_DATA_DIR="$CURRENT_DIR"/../../deepchem_data_dir

COMMAND="""
python $CURRENT_DIR/script.py \
    --random_seed 42 \
    --dataset dockstring  \
    --num_epochs 20  \
    --model_dir $MODELS_DIR/pretrained_dockstring
"""
conda run --no-capture-output --name $ENV_NAME $COMMAND