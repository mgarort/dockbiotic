import argparse
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
from dockbiotic.utils import chemistry
from dockbiotic import constants
from dockbiotic.data import dataframes


# Get excape.tsv input file location
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--excape', help='path to input excape.tsv file produced with download_dataset.sh script',
                        required=True)
    parser.add_argument('--debug', help='If provided, will create RDKit dataset with just 5000 molecules.',
                        required=False, default=False, action='store_true')
    return parser.parse_args()

args = parse_args()
excape = args.excape
debug = args.debug


# Load ExCAPE data
df = pd.read_table(excape).rename(columns={'Ambit_InchiKey':'inchikey',
                                           'SMILES':'smiles',
                                           'Gene_Symbol':'gene',
                                           'Activity_Flag':'label'})

if debug:
    df = df.sample(n=5_000, random_state=0)

# Standardize ExCAPE smiles and delete records with invalid smiles
correspondence = pd.DataFrame(df['smiles'].unique(), columns=['smiles'])
correspondence['standard_smiles'] = correspondence['smiles'].parallel_apply(chemistry.standardize_smiles)
correspondence['minimal_standard_smiles'] = correspondence['smiles'].parallel_apply(chemistry.minimal_standardize_smiles)
correspondence = correspondence.set_index('smiles')


def smiles_to_standard_smiles(smiles):
    return correspondence.loc[smiles,'standard_smiles']

def smiles_to_minimal_standard_smiles(smiles):
    return correspondence.loc[smiles,'minimal_standard_smiles']

df['standard_smiles'] = df['smiles'].progress_map(smiles_to_standard_smiles)
df['minimal_standard_smiles'] = df['smiles'].progress_map(smiles_to_minimal_standard_smiles)

mask_null = df['standard_smiles'].isna()
df = df.loc[~mask_null]


# Merge the records for the same molecule into the same row
standard_smiles = df['standard_smiles'].unique()
genes = list(df['gene'].unique())
merged = pd.DataFrame(index=standard_smiles,
                      columns=['minimal_standard_smiles'] + genes)


for name, row in tqdm(df.iterrows(), total=df.shape[0]):
    standard_smiles = row['standard_smiles']
    gene = row['gene']
    label = row['label'] == 'A'
    merged.loc[standard_smiles,gene] = label
    minimal_standard_smiles = row['minimal_standard_smiles']
    merged.loc[standard_smiles,'minimal_standard_smiles'] = minimal_standard_smiles


# Name the index to "standard smiles"
merged = merged.rename_axis('standard_smiles')


# Put the dockstring ExCAPE molecules at the beginning so that loading the first
# 1000 rows when debugging = True, we obtain the same molecules in dockstring,
# ExCAPE and rdkit
dockstring = dataframes.load_dockstring_dataframe(debug=False)
mask_in_dockstring = merged.index.isin(dockstring['standard_smiles'])
mask_not_in_dockstring = ~mask_in_dockstring
excape_index = merged.index[mask_in_dockstring].append(merged.index[mask_not_in_dockstring])
merged = merged.loc[excape_index].reset_index()


# Save
processed_path = constants.DATA_DIR / 'excape' / 'processed_excape.tsv'
merged.to_csv(processed_path,sep='\t',header=True,index=False)
