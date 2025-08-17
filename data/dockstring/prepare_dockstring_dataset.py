import argparse
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
from dockbiotic.utils.chemistry import standardize_smiles
from dockbiotic import constants

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', help='If provided, will create RDKit dataset with just 5000 molecules.',
                        required=False, default=False, action='store_true')
    return parser.parse_args()

args = parse_args()
debug = args.debug


dockstring = pd.read_csv(constants.DATA_DIR / 'dockstring' / 'dockstring-dataset.tsv',sep='\t')
if debug:
    dockstring = dockstring.sample(n=5_000, random_state=0)
dockstring['numeric_index'] = 'dockstring_' + dockstring.index.astype(str)
dockstring['standard_smiles'] = dockstring['smiles'].parallel_apply(standardize_smiles)
dockstring.to_csv(constants.DATA_DIR / 'dockstring' / 'processed_dockstring.tsv',sep='\t',header=True,index=False)
