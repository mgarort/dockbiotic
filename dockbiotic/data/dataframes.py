from typing import Optional, List
import pandas as pd
from dockbiotic import constants
import dockbiotic.data.helpers as data_helpers
from dockbiotic.utils.similarities import dataframes as sim_dataframes


def load_stokes_dataframe(debug: bool = False) -> pd.DataFrame:
    """
    Load processed Stokes dataset as a pandas dataframe.

    Args:
        debugging(bool): if True, load small (n=1000) dataset for fast debugging.

    Returns:
        pandas dataframe.
    """
    df = pd.read_csv(constants.DATA_DIR / 'stokes' / 'processed_stokes.tsv', sep='\t')
    if debug:
        df = df.sample(n=1000,random_state=101)
    return df.sample(frac=1, random_state=101)


strain_dict = {'atcc25922':'ATCC 25922',
        # not currently used. if you'd like to use,
        # modify the script prepare_coadd_dataset.py
        # 'mut_tolc':'tolC; MB5747',
        # 'mut_lpxc':'lpxC; MB4902'
        }


def load_coadd_dataframe(strain: Optional[str] = None, debug: bool = False, 
                         discard_above: Optional[float] = None) -> pd.DataFrame:
    """
    Load processed COADD dataset as a pandas dataframe.

    Args:
        strain: currently only 'atcc25922' is available.
        debugging: if True, load small (n=1000) dataset for fast debugging.
        discard_above: discard COADD molecules closer to Stokes than this fingerprint
            similarity threshold. If None, don't discard.
    """
    # Check strain is valid
    if strain not in [None, 'atcc25922']:
        raise ValueError(f'strain should be one of None or "atcc25922". Currently it is {strain}')
    df = pd.read_csv(constants.DATA_DIR / 'coadd' / 'processed_coadd.tsv', sep='\t')
    # Load data
    if strain is not None:
        strain_name = strain_dict[strain]
        mask_strain = df['STRAIN'] == strain_name
        df = df.loc[mask_strain]
    if discard_above is not None:
        df = df.set_index('numeric_index')
        closest_similarities_rdkit = sim_dataframes.load_closest_similarities(strain=strain,datasets='coadd_stokes',fp_type='rdkit').loc[df.index]
        closest_similarities_morgan = sim_dataframes.load_closest_similarities(strain=strain,datasets='coadd_stokes',fp_type='morgan').loc[df.index]
        similarity_mask = (closest_similarities_rdkit.iloc[:,0] < discard_above) & (closest_similarities_morgan.iloc[:,0] < discard_above)
        df = df.loc[similarity_mask].reset_index()
    if debug:
        df = df.sample(n=1000, random_state=101)
    return df.sample(frac=1, random_state=101)


def load_stokes_and_coadd_dataframe(debug: bool = False,
                                    balance: bool = False) -> pd.DataFrame:
    """
    Load processed columns of Stokes and COADD dataset as a pandas dataframe.
    From COADD, we only load ATCC 25922 since that is the only strain consistent with
    the Stokes' strain.

    Args:
        debugging: if True, load small (n=1000) dataset for fast debugging.
    """
    stokes = load_stokes_dataframe(debug=False)
    coadd = load_coadd_dataframe(strain='atcc25922', debug=False)
    df = pd.concat([stokes,coadd],axis=0)[['SMILES', 'standard_smiles',
                                           'minimal_standard_smiles',
                                           'tautomer_standard_smiles',
                                           'processed_inhibition',
                                           'numeric_index', 'inchikey']]
    if balance:
        df = data_helpers.balance_dataframe(df, resampling_factor=100, activity_threshold = 0.8)
        df = df.sample(frac=1)
    if debug:
        df = df.sample(n=1000,random_state=101)
    return df.sample(frac=1, random_state=101)


def prepend_excape(column: str) -> str:
    if column == 'standard_smiles':
        return column
    elif column == 'minimal_standard_smiles':
        return column
    else:
        return f'excape_{column}'


def load_excape_dataframe(debug: bool = False,
                          usecols: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load ExCAPE dataset as a pandas dataframe.

    Args:
        debugging: if True, load small (n=1000) dataset for fast debugging.
    """
    path = constants.DATA_DIR / 'excape' / 'processed_excape.tsv'
    if debug:
        df = pd.read_csv(path, sep='\t', nrows=1000, usecols=usecols)
    else:
        df = pd.read_csv(path, sep='\t', usecols=usecols)
    df.columns = df.columns.map(prepend_excape)
    return df


def prepend_dockstring(column: str) -> str:
    if column in ['standard_smiles','inchikey','smiles','numeric_index']:
        return column
    else:
        return f'dockstring_{column}'


def load_dockstring_dataframe(debug: bool = False) -> pd.DataFrame:
    """
    Load DOCKSTRING dataset as a pandas dataframe.

    Args:
        debugging: if True, load small (n=1000) dataset for fast debugging.
    """
    path = constants.DATA_DIR / 'dockstring' / 'processed_dockstring.tsv'
    df = pd.read_csv(path, sep='\t')
    if debug:
        smiles = load_excape_dataframe(debug=True)['standard_smiles']
        intersection = list(set(smiles).intersection(set(df['standard_smiles'])))
        df = df.set_index('standard_smiles').loc[intersection].reset_index().drop_duplicates(subset='standard_smiles')
    df.columns = df.columns.map(prepend_dockstring)
    return df


def load_rdkit_dataframe(debug: bool = False) -> pd.DataFrame:
    path = constants.DATA_DIR / 'rdkit' / 'processed_rdkit.tsv'
    df = pd.read_csv(path, sep='\t')
    if debug:
        smiles = load_excape_dataframe(debug=True)['standard_smiles']
        df = df.set_index('standard_smiles').loc[smiles].reset_index().drop_duplicates(subset='standard_smiles')
    return df


def load_red_dataframe(debug: bool = False) -> pd.DataFrame:
    # Load
    dockstring = load_dockstring_dataframe(debug=debug).drop_duplicates(subset='standard_smiles')
    rdkit = load_rdkit_dataframe(debug=debug)
    excape = load_excape_dataframe(debug=debug)
    # Select only dockstring mols and order in the same way
    intersection = list(set(dockstring['standard_smiles']).intersection(
            set(rdkit['standard_smiles'])
        ).intersection(
            set(excape['standard_smiles'])
        ))
    dockstring = dockstring.set_index('standard_smiles').loc[intersection]
    rdkit = rdkit.set_index('standard_smiles').loc[intersection]
    excape = excape.set_index('standard_smiles').loc[intersection]
    # Concatenate
    df = pd.concat([rdkit,excape,dockstring],axis=1).reset_index()
    return df


def load_chemdiv_dataframe(debug: bool = False):
    path = constants.DATA_DIR / 'chemdiv' / f'processed_chemdiv.tsv'
    df = pd.read_csv(path, sep='\t')
    if debug:
        df = df.sample(n=1000,random_state=101)
    return df