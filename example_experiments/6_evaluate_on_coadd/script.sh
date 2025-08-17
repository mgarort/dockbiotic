#!/bin/bash

ENV_NAME="dockbiotic"

CURRENT_DIR=$(dirname "$(readlink -f "$0")")
MODELS_DIR=$CURRENT_DIR/../../saved_models
export DEEPCHEM_DATA_DIR="$CURRENT_DIR"/../../deepchem_data_dir

COMMAND="""
python $CURRENT_DIR/script.py \
    --model_type attentive_fp \
    --dataset stokes \
    --mode regression \
    --model_dir $MODELS_DIR/finetuned_stokes \
    --results_path $CURRENT_DIR/enrichment_factor.txt
"""
conda run --no-capture-output --name $ENV_NAME $COMMAND