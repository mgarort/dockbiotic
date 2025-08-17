#!/bin/bash

ENV_NAME="dockbiotic"


CURRENT_DIR=$(dirname "$(readlink -f "$0")")
MODELS_DIR=$CURRENT_DIR/../../saved_models
DATA_DIR=$CURRENT_DIR/../../data
export DEEPCHEM_DATA_DIR="$CURRENT_DIR"/../../deepchem_data_dir

COMMAND="""
python $CURRENT_DIR/script.py \
    --model_type attentive_fp \
    --dataset stokes \
    --mode regression \
    --model_dir $MODELS_DIR/finetuned_stokes \
    --smiles_path $DATA_DIR/chemdiv/chemdiv_sample.tsv \
    --preds_path $CURRENT_DIR/chemdiv_sample_preds.tsv
"""
conda run --no-capture-output --name $ENV_NAME $COMMAND