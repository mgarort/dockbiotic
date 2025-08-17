import pandas as pd
from functools import partial
from tqdm import tqdm
import numpy as np
import rdkit
import rdkit.DataStructs as DataStructs
from dockbiotic.utils import chemistry
from typing import Union


def jaccard_coefficient(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """
    Compute the Jaccard coeffient between two arrays
    """
    array1 = np.asarray(array1).astype(bool)
    array2 = np.asarray(array2).astype(bool)
    intersection = np.logical_and(array1, array2)
    union = np.logical_or(array1, array2)
    return np.sum(intersection,axis=1) / np.sum(union,axis=1).astype(float)


np_or_rdkit = Union[np.ndarray, rdkit.DataStructs.cDataStructs.ExplicitBitVect]
def tanimoto(fp1: np_or_rdkit, fp2: np_or_rdkit) -> float:
    """
    Compute Tanimoto similarity (Jaccard coefficient) between two fingerprints,
    which could be given in RDKit format (rdkit.DataStructs.cDataStructs.ExplicitBitVect)
    or as numpy arrays.
    """
    # fp1 = np.squeeze(fp1)
    # fp2 = np.squeeze(fp2)
    rdkit_format = rdkit.DataStructs.cDataStructs.ExplicitBitVect
    if isinstance(fp1, np.ndarray) and isinstance(fp2, np.ndarray):
        return jaccard_coefficient(fp1,fp2)
    elif isinstance(fp1, rdkit_format) and isinstance(fp2, rdkit_format):
        return DataStructs.FingerprintSimilarity(fp1,fp2)
    else:
        raise RuntimeError('Fingerprints must be either all DataStructs.cDataStructs.ExplicitBitVect' \
                           'or all numpy.arrays, but not a mix.')


def compute_closest_and_similarities(df_origin: pd.DataFrame, df_target: pd.DataFrame,
                                     fp_type: str, smiles_type: str='standard_smiles',
                                     num_closest: int=10) -> None:
    """
    - "origin" refers to the molecules for which we want to find analogues.
    - "target" refers to the pool of molecules where we'll try to find analogues.
    """

    # Get dataframes among which to compute similarities
    # - "origin" refers to the molecules for which we want to find analogues
    # - "target" refers to the pool of molecules where we'll try to find analogues

    # Compute fingerprints of the desired type in the target dataset. One of
    # - RDKit fingerprints, path length 6 (recommended by Greg Landrum to find 
    #   analogues.
    # - Morgan fingerprints.
    smiles_to_fp_filled = partial(chemistry.smiles_to_fp, fp_type=fp_type, keep_rdkit_format=False)
    df_target['fp'] = df_target[smiles_type].parallel_apply(smiles_to_fp_filled)
    df_origin['fp'] = df_origin[smiles_type].parallel_apply(smiles_to_fp_filled)
    similarities = pd.DataFrame(index=df_origin.index, columns=df_target.index)

    for mol_index in tqdm(df_origin.index):
        mol_fp = df_origin.loc[mol_index,'fp']
        target_fp = np.vstack(df_target[['fp']].values.squeeze()).squeeze()
        tani = tanimoto(mol_fp,target_fp)
        similarities.loc[mol_index] = tani
        # Set self-similarity to -1 to avoid selecting self as similar compound
        # if mol_index in similarities.columns:
        #     similarities.loc[mol_index,mol_index] = -1

    # Get only top num_closest compounds and similarity values
    order = np.argsort(-similarities,axis=1)[:,:num_closest]
    closest_compounds = similarities.columns.values[order]
    closest_compounds = pd.DataFrame(closest_compounds, index=similarities.index)
    closest_similarities = np.take_along_axis(similarities.values, order, axis=1)
    closest_similarities = pd.DataFrame(closest_similarities, index=similarities.index)

    return closest_compounds, closest_similarities