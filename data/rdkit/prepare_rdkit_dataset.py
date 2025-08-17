from dockbiotic.data import dataframes
from dockbiotic.data import helpers
from dockbiotic import constants
import numpy as np
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', help='If provided, will create RDKit dataset with just 5000 molecules.',
                        required=False, default=False, action='store_true')
    return parser.parse_args()

args = parse_args()
debug = args.debug


def make_rdkit_dataframe():
    # Load smiles
    excape = dataframes.load_excape_dataframe(debug=False, usecols=[0,1])
    standard_smiles = excape['standard_smiles'].values.reshape(-1,1)
    minimal_standard_smiles = excape['minimal_standard_smiles'].values.reshape(-1,1)
    # Get RDKit descriptors
    y = helpers.get_y(smiles=standard_smiles,original_data=None,y_type='rdkit',parallelize=True)
    # Make dataframe
    data = np.hstack([standard_smiles, minimal_standard_smiles, y])
    columns = ['standard_smiles', 'minimal_standard_smiles'] + [f'rdkit_{i}' for i in range(y.shape[1])]
    df = pd.DataFrame(data,columns=columns)
    return df


df = make_rdkit_dataframe()
if debug:
    n = min(5_000,df.shape[0])
    df = df.sample(n=n, random_state=0)
path = constants.DATA_DIR / 'rdkit' / 'processed_rdkit.tsv'
df.to_csv(path,sep='\t',header=True,index=False)

