"""Functions to visualize pairs of tanimoto similarities in COADD and Stokes."""

import numpy as np
import pandas as pd
from functools import partial
from dockbiotic.data import dataframes
from dockbiotic.utils import scripting
from dockbiotic.utils import similarities as sim


def sigmoid(z: float) -> float:
    return 1 / (1 + np.exp(-z))


def get_selection_probability(this_similarity: float, desired_similarity: float) -> float:
    diff = np.abs(desired_similarity - this_similarity)
    return sigmoid(100*(np.exp(3.28*(1 - diff - 0.99))-1))


def get_random_number(_) -> float:
    return np.random.uniform()


def get_selection_decision(selection_probability):
    return np.random.uniform() < selection_probability


def get_example_similarities(desired_similarity,coadd=None,return_size=100,random_seed=101):
    # Load COADD and Stokes datasets
    if coadd is None:
        coadd = dataframes.load_coadd_dataframe(debug=False)
    stokes = dataframes.load_stokes_dataframe(debug=False)
    coadd.index = 'coadd_' + coadd.index.astype(str)
    stokes.index = 'stokes_' + stokes.index.astype(str)
    # Load closest similarities and closest compounds data
    closest_similarities = sim.dataframes.load_closest_similarities(debug=False)
    closest_compounds = sim.dataframes.load_closest_compounds(debug=False)
    # Create dataframe with example similarities
    closest_summary = pd.DataFrame(columns=['similarity','coadd_index','stokes_index',
                                            'coadd_smiles','stokes_smiles','diff'])
    closest_summary['similarity'] = closest_similarities.iloc[:,0]
    closest_summary['coadd_index'] = closest_compounds.index
    closest_summary['stokes_index'] = closest_compounds.iloc[:,0]
    partial_coadd_index_to_smiles = partial(scripting.coadd_index_to_smiles,coadd=coadd)
    closest_summary['coadd_smiles'] = closest_summary['coadd_index'].map(partial_coadd_index_to_smiles)
    partial_stokes_index_to_smiles = partial(scripting.stokes_index_to_smiles,stokes=stokes)
    closest_summary['stokes_smiles'] = closest_summary['stokes_index'].map(partial_stokes_index_to_smiles)
    partial_selection_probability = partial(get_selection_probability,desired_similarity=desired_similarity)
    closest_summary['selection_probability'] = closest_summary['similarity'].map(partial_selection_probability)
    closest_summary['random_number'] = np.random.uniform(size=(closest_summary.shape[0],))
    closest_summary['selection_decision'] = closest_summary['selection_probability'] > closest_summary['random_number']
    mask_decision = closest_summary['selection_decision']
    selected =  closest_summary.loc[mask_decision,['similarity','coadd_smiles','stokes_smiles']]
    if len(selected) < return_size:
        return selected.sort_values(by='similarity')
    else:
        return selected.sample(n=return_size,random_state=random_seed).sort_values(by='similarity')