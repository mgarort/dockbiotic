"""Script to evaluate the enrichment factor (EF) of a model on the COADD dataset. """

import argparse
from dockbiotic.utils import modeling
from dockbiotic.utils import evaluation
from dockbiotic.data import dataframes
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Model type (XGB or AttentiveFP).', required=True)
    parser.add_argument('--dataset', type=str, help='Name of dataset on which model was fine-tuned', required=True)
    parser.add_argument('--model_dir', type=str, help='Directory to load model from', required=True)
    parser.add_argument('--mode', type=str, help='"regression", "classification" or "antibiotic_classification"')
    parser.add_argument('--results_path', type=str, help='Path to save results.', required=True)
    return parser.parse_args()


args = parse_args()
model_type = args.model_type
dataset_name = args.dataset
model_dir = args.model_dir
mode = args.mode
results_path = args.results_path


# load model
model = modeling.get_model(model_type=model_type,
             dataset_name=dataset_name,
             mode=mode,
             pretrained_model_dir=model_dir)

# evaluate ef on coadd
ef = evaluation.evaluate_on_coadd(model=model)

# write results
with open(results_path, 'w') as f:
    f.write(f'{ef}\n')