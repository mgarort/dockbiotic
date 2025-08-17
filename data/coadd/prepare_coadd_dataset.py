import argparse
from functools import partial
import pandas as pd
from rdkit.Chem.MolStandardize.tautomer import TautomerCanonicalizer
from tqdm import tqdm
tqdm.pandas()
from dockbiotic import constants
from dockbiotic.utils import chemistry

def delistify(row):
    assert isinstance(row, list)
    assert len(row) == 1
    return row[0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', help='If provided, will create RDKit dataset with just 5000 molecules.',
                        required=False, default=False, action='store_true')
    return parser.parse_args()

args = parse_args()
debug = args.debug


allowed_fragments = pd.read_csv(constants.DATA_DIR / 'coadd' / 'allowed_fragments.txt',
                                sep='\t', header=None).iloc[:,0].tolist()
df = pd.read_csv(constants.DATA_DIR / 'coadd' / 'CO-ADD_InhibitionData_r03_01-02-2020_CSV.csv',
                    sep=',', header=0)

# mask to the only strain that we will use
mask = df['STRAIN'] == 'ATCC 25922'
df = df.loc[mask]

if debug:
    df = df.sample(n=5_000, random_state=0)


# Create a map of non-standard to standard SMILES. This is useful for the conversion
# because SMILES are repeated, and standardization is very slow. So if we standardized
# the repeated SMILES directly it would take much longer
correspondence = df[['SMILES']].drop_duplicates().reset_index(drop=True)
correspondence.columns = ['smiles']
enumerate_allowed_fragments = partial(chemistry.enumerate_allowed_fragments,
                                      allowed_fragments=allowed_fragments)
correspondence['allowed_fragments'] = correspondence['smiles'].parallel_apply(enumerate_allowed_fragments)
correspondence['num_allowed_fragments'] = correspondence['allowed_fragments'].map(len)
# Keep compounds with a single unique SMILES
correspondence = correspondence.loc[correspondence['num_allowed_fragments'] == 1]
# Now that we have a single SMILES, return from list to string format
correspondence['allowed_fragments'] = correspondence['allowed_fragments'].map(delistify)
# Standardize fragments in the same way as DOCKSTRING
correspondence = correspondence.rename({'allowed_fragments':'nonstandard_fragment'},
                                       axis=1)
correspondence['standard_smiles'] = correspondence['nonstandard_fragment'].parallel_apply(chemistry.standardize_smiles)
correspondence['minimal_standard_smiles'] = correspondence['nonstandard_fragment'].progress_map(chemistry.minimal_standardize_smiles)
# canonicalize tautomers
tautomer_canonicalizer = TautomerCanonicalizer(max_tautomers=100)
canon_tautom_smiles = partial(chemistry.canonicalize_tautomer_smiles,
                              canonicalizer=tautomer_canonicalizer)
correspondence['tautomer_standard_smiles'] = correspondence['standard_smiles'].parallel_apply(canon_tautom_smiles)


# Add standard_smiles and minimal_standard_smiles to the original COADD dataframe
correspondence = correspondence.set_index('smiles')
correspondence_index = correspondence.index.tolist()

def map_to_standard(row):
    if row in correspondence_index:
        return correspondence.loc[row,'standard_smiles']

def map_to_minimal_standard(row):
    if row in correspondence_index:
        return correspondence.loc[row,'minimal_standard_smiles']

def map_to_tautomer(row):
    if row in correspondence_index:
        return correspondence.loc[row,'tautomer_standard_smiles']

df['standard_smiles'] = df['SMILES'].progress_map(map_to_standard)
df['minimal_standard_smiles'] = df['SMILES'].progress_map(map_to_minimal_standard)
df['tautomer_standard_smiles'] = df['SMILES'].progress_map(map_to_tautomer)

mask_null = df['standard_smiles'].isnull() | df['tautomer_standard_smiles'].isnull()
df = df.loc[~mask_null]


# Select only the fraction of COADD with actual data
df = df.loc[~df['INHIB_AVE'].isnull()]

# Process inhibition data to make compatible with Stokes
df['scaled_INHIB_AVE'] = df['INHIB_AVE'] / 100
df['processed_inhibition'] = df.groupby(['COADD_ID'])['scaled_INHIB_AVE'].transform('max')
df['max_processed_inhibition'] = df.groupby(['COADD_ID'])['scaled_INHIB_AVE'].transform('max')
df = df.sort_values(['max_processed_inhibition','processed_inhibition'],
                                      ascending=False).set_index(['COADD_ID','CONC']).reset_index()
df = df.drop(['max_processed_inhibition','scaled_INHIB_AVE'],axis=1)

# Add a single index for each record (not for the molecules, but for the records)
df['numeric_index'] = 'coadd_' + df.index.astype(str)

# Add inchikeys
df['inchikey'] = df['tautomer_standard_smiles'].progress_map(chemistry.smiles_to_inchikey)


df.to_csv(constants.DATA_DIR / 'coadd' / 'processed_coadd.tsv',
                   sep='\t',header=True,index=False)


print('Confirm that each compound has only been tested at a single concentration')
print(df.groupby(['COADD_ID']).agg({'CONC':'nunique'}).sort_values(by='CONC',ascending=False))
