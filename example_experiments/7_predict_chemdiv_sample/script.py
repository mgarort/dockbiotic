"""
Script to compute predictions and save in an output file.

The higher the prediction p, the stronger the predicted activity.
Predictions have been trained on labels y from Stokes and COADD
that lie in the following ranges:

- 0 <= y < 0.8   ->  no or low antibiotic activity (binarized as negative)
- 0.8 < y <= 1  ->  strong antibiotic activity (binarized as positive)

The range of predictions may be interpreted similarly, with the only
caveat that a regression model could predict values p < 0 or p > 1.
If so, you can clip predictions to the [0, 1] range.
"""

import argparse
from dockbiotic.utils import modeling
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Model type (XGB or AttentiveFP).', required=True)
    parser.add_argument('--dataset', type=str, help='Name of dataset on which model was fine-tuned', required=True)
    parser.add_argument('--model_dir', type=str, help='Directory to load model from', required=True)
    parser.add_argument('--mode', type=str, help='"regression", "classification" or "antibiotic_classification"')
    parser.add_argument('--smiles_path', type=str, help='Path to tsv file with a column called "smiles".', required=True)
    parser.add_argument('--id_column', type=str, help='Optional name of column in the smiles file to use as ids in the predictions file')
    parser.add_argument('--preds_path', type=str, help='Path to save predictions.', required=True)
    return parser.parse_args()


args = parse_args()
model_type = args.model_type
dataset_name = args.dataset
model_dir = args.model_dir
mode = args.mode
smiles_path = args.smiles_path
id_column = args.id_column
preds_path = args.preds_path

# load model
model = modeling.get_model(model_type=model_type,
             dataset_name=dataset_name,
             mode=mode,
             pretrained_model_dir=model_dir)

# load smiles and ids
smiles_df = pd.read_csv(smiles_path, sep='\t')
smiles = smiles_df['smiles']
if id_column is not None:
    ids = smiles_df[id_column]
else:
    ids = None

# make predictions with
preds = modeling.predict_smiles(smiles=smiles,
                                model=model,
                                ids=ids)

# save predictions
preds.to_csv(preds_path, sep='\t', index=False)