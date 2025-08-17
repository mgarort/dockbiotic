#!/bin/bash

ENV_NAME="dockbiotic"
CURRENT_DIR=$(dirname "$(readlink -f "$0")")


for ORIGIN in stokes coadd ; do
    for TARGET in stokes coadd ; do

        # skip intra-dataset similarities
        if [[ "$ORIGIN" == "$TARGET" ]]; then
            continue 
        fi

        printf "\nCalculating similarities for: origin $ORIGIN, target $TARGET"
        COMMAND="""
        python $CURRENT_DIR/script.py \
                --origin_dataset $ORIGIN \
                --target_dataset $TARGET
        """
        conda run --no-capture-output --name $ENV_NAME $COMMAND

    done
done