import pandas as pd
import numpy as np
from dockbiotic.constants import DATA_DIR
from dockbiotic.data import dataframes
from typing import Tuple


def sim_df_has_header(path: str) -> bool:
    sim_df = pd.read_csv(path, sep='\t', nrows=1)
    return 'numeric_index' in sim_df.columns


def load_closest_similarities(strain=None, debug=False, datasets='coadd_stokes', fp_type='rdkit'):

    path = DATA_DIR / 'similarities' / f'closest_similarities_{fp_type}_{datasets}.tsv'
    if sim_df_has_header(path):
        closest_similarities = pd.read_csv(path, sep='\t', header=0)
    else:
        closest_similarities = pd.read_csv(path, sep='\t', header=None)
        n_cols = len(closest_similarities.columns)
        cols = ['numeric_index'] + [i for i in range(n_cols-1)]
        closest_similarities.columns = cols
    closest_similarities.set_index('numeric_index', inplace=True)
    if strain is not None:
        coadd = dataframes.load_coadd_dataframe(strain=strain).set_index('numeric_index')
        closest_similarities = closest_similarities.loc[coadd.index]
    if debug:
        return closest_similarities.sample(n=1000,random_state=101)
    else:
        return closest_similarities


def load_closest_compounds(strain=None,debug=False,datasets='coadd_stokes',fp_type='morgan'):
    path = DATA_DIR / 'similarities' / f'closest_compounds_{fp_type}_{datasets}.tsv'
    if sim_df_has_header(path):
        closest_compounds = pd.read_csv(path, sep='\t', header=0)
    else:
        closest_compounds = pd.read_csv(path, sep='\t', header=None)
        n_cols = len(closest_compounds.columns)
        cols = ['numeric_index'] + [i for i in range(n_cols-1)]
        closest_compounds.columns = cols
    closest_compounds.set_index('numeric_index', inplace=True)
    if strain is not None:
        coadd = dataframes.load_coadd_dataframe(strain=strain).set_index('numeric_index')
        closest_compounds = closest_compounds.loc[coadd.index]
    if debug:
        return closest_compounds.sample(n=1000,random_state=101)
    else:
        return closest_compounds


def get_origin_and_target_dataframes(
        origin: str, target: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Load all required dataframes
    if origin == 'coadd' or target == 'coadd':
        coadd = dataframes.load_coadd_dataframe(strain='atcc25922',
                                     debug=False).set_index('numeric_index')
    if origin == 'stokes' or target == 'stokes':
        stokes = dataframes.load_stokes_dataframe(debug=False).set_index('numeric_index')
    if origin == 'chemdiv' or target == 'chemdiv':
        chemdiv = dataframes.load_chemdiv_dataframe(version='latest',
                                         debug=False).set_index('numeric_index')
    if origin == 'stokescoadd' or target == 'stokescoadd' or origin == 'stokescoaddactives' or target == 'stokescoaddactives':
        stokes_and_coadd = dataframes.load_stokes_and_coadd_dataframe(debug=False).set_index('numeric_index')
    if origin == 'top_enamine' or target == 'top_enamine':
        top_enamine = dataframes.load_enamine_predictions_above_05(debug=False, shuffle=False).set_index('numeric_index')
    # Assign dataframe to "origin"
    if origin == 'coadd':
        df_origin = coadd
    elif origin == 'stokes':
        df_origin = stokes
    elif origin == 'chemdiv':
        df_origin = chemdiv
    elif origin == 'stokescoadd':
        df_origin = stokes_and_coadd
    elif origin == 'stokescoaddactives':
        mask = stokes_and_coadd['processed_inhibition'] > 0.5
        df_origin = stokes_and_coadd.loc[mask]
    elif origin == 'top_enamine':
        df_origin = top_enamine
    else:
        raise ValueError('"origin" must be either "coadd", "stokes", "chemdiv", "stokescoadd", "stokescoaddactives" or "top_enamine".')
    # Assign dataframe to "target"
    if target == 'coadd':
        df_target = coadd
    elif target == 'stokes':
        df_target = stokes
    elif target == 'chemdiv':
        df_target = chemdiv
    elif target == 'stokescoadd':
        df_target = stokes_and_coadd
    elif target == 'stokescoaddactives':
        mask = stokes_and_coadd['processed_inhibition'] > 0.5
        df_target = stokes_and_coadd.loc[mask]
    elif target == 'top_enamine':
        df_target = top_enamine
    else:
        raise ValueError('"target" must be either "coadd", "stokes", "chemdiv", "stokescoadd", "stokescoaddactives" or "top_enamine".')

    return df_origin, df_target