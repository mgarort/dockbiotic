from functools import partial
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.MolStandardize.tautomer import TautomerCanonicalizer
from dockbiotic import constants
from dockbiotic.utils import chemistry

tqdm.pandas()

allowed_fragments = pd.read_csv(constants.DATA_DIR / 'stokes' / 'allowed_fragments.txt',
                                sep='\t', header=None).iloc[:,0].tolist()
stokes = pd.read_excel(constants.DATA_DIR / 'stokes/mmc1.xlsx',
                   sheet_name='S1B', skiprows=1)

# Identify the unique SMILES in each compounds
enumerate_allowed_fragments = partial(chemistry.enumerate_allowed_fragments, allowed_fragments=allowed_fragments)
stokes['allowed_fragments'] = stokes['SMILES'].map(enumerate_allowed_fragments)
stokes['num_allowed_fragments'] = stokes['allowed_fragments'].map(len)
# Keep compounds with a single unique SMILES
stokes = stokes.loc[stokes['num_allowed_fragments'] == 1]
# Now that we have a single SMILES, return from list to string format
def delistify(row):
    assert isinstance(row,list)
    assert len(row) == 1
    return row[0]
stokes['allowed_fragments'] = stokes['allowed_fragments'].map(delistify)


# Normalize compounds in the same way as DOCKSTRING
stokes['standard_smiles'] = stokes['allowed_fragments'].parallel_apply(chemistry.standardize_smiles)
stokes['minimal_standard_smiles'] = stokes['allowed_fragments'].parallel_apply(chemistry.minimal_standardize_smiles)
mask_null = stokes['standard_smiles'].isnull()
stokes = stokes.loc[~mask_null]


# canonicalize tautomers
tautomer_canonicalizer = TautomerCanonicalizer(max_tautomers=100)
stokes['mol'] = stokes['standard_smiles'].map(Chem.MolFromSmiles)
stokes['tautomer_mol'] = stokes['mol'].parallel_apply(tautomer_canonicalizer.canonicalize)
stokes['tautomer_standard_smiles'] = stokes['tautomer_mol'].map(Chem.MolToSmiles)


mask_null = stokes['tautomer_standard_smiles'].isnull()
stokes = stokes.loc[~mask_null]


stokes = stokes.drop(columns=['mol', 'tautomer_mol'])


# Process inhibition values so that Stokes and COADD are comparable
stokes['processed_inhibition'] = 1 - stokes['Mean_Inhibition']


stokes = stokes[['Mean_Inhibition', 'SMILES', 'Name', 'Activity',
                 'standard_smiles', 'minimal_standard_smiles',
                 'tautomer_standard_smiles',
                 'processed_inhibition']]
stokes['numeric_index'] = 'stokes_' + stokes.reset_index().index.astype(str)
stokes['inchikey'] = stokes['tautomer_standard_smiles'].map(chemistry.smiles_to_inchikey)
stokes.to_csv(constants.DATA_DIR / 'stokes' / 'processed_stokes.tsv',
              sep='\t', header=True, index=False)