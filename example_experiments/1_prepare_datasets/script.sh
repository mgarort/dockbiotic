#!/bin/bash

ENV_NAME="dockbiotic"
CURRENT_DIR=$(dirname "$(readlink -f "$0")")
DATA_DIR=$CURRENT_DIR/../../data


echo 'All datasets will be prepared in debug mode, i.e. using up to 5000 samples. ' \
     'To use the whole dataset, remove the --debug flags. ' \
     'However, this will make the example experiments much slower.'

# pre-training datasets

printf "\n\nPreparing dockstring dataset\n\n"
DOCKSTRING_DIR="$DATA_DIR/dockstring"
xz -d -k $DOCKSTRING_DIR/dockstring-dataset.tsv.xz
COMMAND="python $DOCKSTRING_DIR/prepare_dockstring_dataset.py --debug"
conda run --no-capture-output --name $ENV_NAME $COMMAND

printf "\n\nPreparing ExCAPE dataset\n\n"
EXCAPE_DIR="$DATA_DIR/excape"
bash $EXCAPE_DIR/download_excape.sh
COMMAND="python $EXCAPE_DIR/prepare_excape_dataset.py --excape $EXCAPE_DIR/excape.tsv --debug"
conda run --no-capture-output --name $ENV_NAME $COMMAND

printf "\n\nPreparing RDKit dataset\n\n"
RDKIT_DIR="$DATA_DIR/rdkit"
COMMAND="python $RDKIT_DIR/prepare_rdkit_dataset.py --debug"
conda run --no-capture-output --name $ENV_NAME $COMMAND

# antibiotic datasets

printf "\n\nPreparing Stokes dataset\n\n"
STOKES_DIR="$DATA_DIR/stokes"
COMMAND="python $STOKES_DIR/prepare_stokes_dataset.py"
conda run --no-capture-output --name $ENV_NAME $COMMAND

printf "\n\nPreparing COADD dataset\n\n"
COADD_DIR="$DATA_DIR/coadd"
xz -k -d $COADD_DIR/CO-ADD_InhibitionData_r03_01-02-2020_CSV.csv.xz
COMMAND="python $COADD_DIR/prepare_coadd_dataset.py --debug"
conda run --no-capture-output --name $ENV_NAME $COMMAND

# chemdiv

printf "\n\nPreparing ChemDiv dataset\n\n"
CHEMDIV_DIR="$DATA_DIR/chemdiv"
COMMAND="python $CHEMDIV_DIR/prepare_chemdiv_dataset.py"
conda run --no-capture-output --name $ENV_NAME $COMMAND