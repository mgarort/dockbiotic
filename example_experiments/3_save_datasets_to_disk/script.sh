#!/bin/bash

ENV_NAME="dockbiotic"

CURRENT_DIR=$(dirname "$(readlink -f "$0")")
export DEEPCHEM_DATA_DIR=$CURRENT_DIR/../../deepchem_data_dir

COMMAND="python $CURRENT_DIR/script.py"
conda run --no-capture-output --name $ENV_NAME $COMMAND