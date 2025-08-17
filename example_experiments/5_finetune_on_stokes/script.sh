#!/bin/bash

ENV_NAME="dockbiotic"

CURRENT_DIR=$(dirname "$(readlink -f "$0")")
MODELS_DIR=$CURRENT_DIR/../../saved_models
export DEEPCHEM_DATA_DIR="$CURRENT_DIR"/../../deepchem_data_dir


printf "\n\nFinetune on stokes after pretraining on dockstring\n"
COMMAND="""
python $CURRENT_DIR/script.py \
    --random_seed 42 \
    --dataset stokes \
    --mode regression \
    --num_epochs 50  \
    --lr 0.0001 \
    --model_dir $MODELS_DIR/finetuned_stokes \
    --pretrained_model_dir $MODELS_DIR/pretrained_dockstring \
    --pretraining_dataset dockstring
"""
conda run --no-capture-output --name $ENV_NAME $COMMAND