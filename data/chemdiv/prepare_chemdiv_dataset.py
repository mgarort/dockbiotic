from dockbiotic import constants
from dockbiotic.utils import chemistry
from tqdm import tqdm
import pandas as pd
from pandarallel import pandarallel
from rdkit import Chem
from rdkit.Chem.MolStandardize.tautomer import TautomerCanonicalizer
from rdkit import RDLogger
from functools import partial

tqdm.pandas()
pandarallel.initialize(progress_bar=True)
tautomer_canonicalizer = TautomerCanonicalizer(max_tautomers=100)
RDLogger.DisableLog('rdApp.*')

# process smiles
chemdiv_path = constants.DATA_DIR / 'chemdiv' / 'chemdiv_sample.tsv'
df = pd.read_csv(chemdiv_path,sep='\t', header=0)
df['standard_smiles'] = df['smiles'].parallel_apply(chemistry.standardize_smiles)
mask_null = df['standard_smiles'].isnull()
df = df.loc[~mask_null]

# canonicalize tautomerization of smiles
tautomer_canonicalizer = TautomerCanonicalizer(max_tautomers=100)
canon_tautom_smiles = partial(chemistry.canonicalize_tautomer_smiles,
                              canonicalizer=tautomer_canonicalizer)
df['tautomer_standard_smiles'] = df['standard_smiles'].parallel_apply(canon_tautom_smiles)
mask_null = df['tautomer_standard_smiles'].isnull()
df = df.loc[~mask_null]

# qssign numeric_index for consistency with other datasets
df['numeric_index'] = 'chemdiv_' + df.reset_index().index.astype(str)
df['inchikey'] = df['tautomer_standard_smiles'].parallel_apply(chemistry.smiles_to_inchikey)

processed_chemdiv_path = constants.DATA_DIR / 'chemdiv' / 'processed_chemdiv.tsv'
df.to_csv(processed_chemdiv_path,sep='\t', header=True, index=False)
