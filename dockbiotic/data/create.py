import dockbiotic.data.helpers as data_helpers
from dockbiotic.data import dataframes
from typing import Union, Optional, Tuple, Iterable
from pathlib import Path
import numpy as np
import deepchem as dc
import pandas as pd


def create_stokes_data(debug: bool=False, binarize: bool=False,
                     featurization: str='graph',
                     y_type: str='original', dataset_path: Optional[Union[str, Path]]=None,
                     delete_after: bool=True,
                     smiles_type: str='standard_smiles'
                     ) -> Tuple[dc.data.DiskDataset, pd.Series]:
    """
    Load processed Stokes dataset as a DeepChem dataset.

    Args:
        debug: if True, load small (n=100) dataset for fast debugging.
        binarize: if True, inhibition values >= 0.8 are set to 1 (active), and
            inhibition values < 0.8 are set to 0 (inactive).
        featurization: either one of 'smiles', 'mol', 'graph', 'morgan'
            or 'long_morgan' (default 'graph').
        y_type: type of data in the y values. Either one of the following, or
            a list or tuple of the following:
            - 'original': original type of data in the dataset (e.g. antibiotic activity, docking scores, etc).
            - 'rdkit': RDKit molecular descriptors (number of functional groups, logP, polar surface area, etc).
            - 'fp_morgan': Morgan fingerprints, radius 3, length 2048.
            - 'fp_morgan_long': Morgan fingerprints, radius 3, length 4096.
            - 'fp_rdkit': RDKit fingerprints, max path 6.
            - 'fp_rdkit_long': RDKit fingerprints, max path 6, length 4096.
        dataset_path: directory where dataset should be written.
        delete_after: if True, __del__ will be customized so as to attempt to delete the data directory
            when the dataset is deleted.

    Returns:
        DeepChem DiskDataset with X type determined by 'featurization' and Y type determined by 'y_type'.
    """
    # Load data
    df = dataframes.load_stokes_dataframe(debug=debug)
    smiles = df[smiles_type].values.reshape(-1,1)
    if binarize:
        activity = (df['processed_inhibition'] >= 0.8).astype(int).values.reshape(-1,1)
    else:
        activity = df['processed_inhibition'].values.reshape(-1,1)
    numeric_index = df['numeric_index']
    # Get dataset
    dataset = data_helpers.create_dataset(smiles=smiles, original_data=activity, y_type=y_type,
                           featurization=featurization, ids=numeric_index, data_dir=dataset_path,
                           delete_after=delete_after)
    return dataset, numeric_index


def create_coadd_data(strain, discard_above=None, debug=False, binarize=False,
                    featurization='graph', y_type='original', dataset_path=None,
                    delete_after=True, smiles_type='standard_smiles') -> Tuple[dc.data.DiskDataset, pd.Series]:
    """
    Load processed E. coli COADD dataset as a DeepChem dataset.

    Args:
        discard_above(float): discard COADD molecules closer to Stokes than this fingerprint
            similarity threshold. If None, don't discard.
        debugging(bool): if True, load small (n=100) dataset for fast debugging.
        binarize(bool): if True, inhibition values >= 0.8 are set to 1 (active), and
            inhibition values < 0.8 are set to 0 (inactive).
        featurization(str, optional): either one of 'smiles', 'mol', 'graph', 'morgan'
            or 'long_morgan' (default 'graph').
        y_type(str or list or tuple, optional): type of data in the y values. Either one of the following, or
            a list or tuple of the following:
            - 'original': original type of data in the dataset (e.g. antibiotic activity, docking scores, etc).
            - 'rdkit': RDKit molecular descriptors (number of functional groups, logP, polar surface area, etc).
            - 'fp_morgan': Morgan fingerprints, radius 3, length 2048.
            - 'fp_morgan_long': Morgan fingerprints, radius 3, length 4096.
            - 'fp_rdkit': RDKit fingerprints, max path 6.
            - 'fp_rdkit_long': RDKit fingerprints, max path 6, length 4096.
        dataset_path(str or pathlib.Path, optional): directory where dataset should be written.
        delete_after(bool): if True, __del__ will be customized so as to attempt to delete the data directory
            when the dataset is deleted.

    Returns:
        DeepChem DiskDataset with X type determined by 'featurization' and Y type determined by 'y_type'.
    """
    df = dataframes.load_coadd_dataframe(strain=strain,debug=debug,discard_above=discard_above)
    smiles = df[smiles_type].values.reshape(-1,1)
    if binarize:
        activity = (df['processed_inhibition'] >= 0.8).astype(int).values.reshape(-1,1)
    else:
        activity = df['processed_inhibition'].values.reshape(-1,1)
    numeric_index = df['numeric_index']
    # Get dataset
    dataset = data_helpers.create_dataset(smiles=smiles, original_data=activity, y_type=y_type,
                           featurization=featurization, ids=numeric_index, data_dir=dataset_path,
                           delete_after=delete_after)
    return dataset, numeric_index


def create_stokes_and_coadd_data(debug=False, balance=False, binarize=False,
                               featurization='graph', y_type='original',
                               dataset_path=None, delete_after=True,
                               smiles_type='standard_smiles') -> Tuple[dc.data.DiskDataset, pd.Series]:
    # Load data
    df = dataframes.load_stokes_and_coadd_dataframe(debug=debug, balance=balance)
    smiles = df[smiles_type].values.reshape(-1,1)
    if binarize:
        activity = (df['processed_inhibition'] >= 0.8).astype(int).values.reshape(-1,1)
    else:
        activity = df['processed_inhibition'].values.reshape(-1,1)
    numeric_index = df['numeric_index']
    # Get dataset
    dataset = data_helpers.create_dataset(smiles=smiles, original_data=activity,
                             y_type=y_type, featurization=featurization,
                             ids=numeric_index, data_dir=dataset_path,
                             delete_after=delete_after)
    return dataset, numeric_index


def create_rdkit_data(debug=False, featurization='graph', dataset_path=None,
                    delete_after=True, smiles_type='standard_smiles') -> Tuple[dc.data.DiskDataset, pd.Series]:
    # Load data
    df = dataframes.load_rdkit_dataframe(debug=debug)
    smiles = df[smiles_type].values.reshape(-1,1)
    values = df.drop(columns=['standard_smiles',
                              'minimal_standard_smiles']).astype(float).values
    # Get dataset
    dataset = data_helpers.create_dataset(smiles=smiles, original_data=values, y_type='original',
                           featurization=featurization, data_dir=dataset_path,
                           delete_after=delete_after, should_weigh_out_nans=True)
    numeric_index = None
    return dataset, numeric_index


def create_excape_data(debug=False, featurization='graph', dataset_path=None,
                     delete_after=True, smiles_type='standard_smiles') -> Tuple[dc.data.DiskDataset, pd.Series]:
    # Load data
    df = dataframes.load_excape_dataframe(debug=debug)
    smiles = df[smiles_type].values.reshape(-1,1)
    activities = df.drop(columns=['standard_smiles',
                                  'minimal_standard_smiles']).astype(float).values
    # Get dataset
    dataset = data_helpers.create_dataset(smiles=smiles, original_data=activities,
                           y_type='original', featurization=featurization,
                           data_dir=dataset_path, delete_after=delete_after,
                           should_weigh_out_nans=True)
    numeric_index = None
    return dataset, numeric_index


def create_dockstring_data(debug: bool=False, featurization: str='graph', 
                         y_type: str='original',
                         dataset_path: Union[None, str, Path]=None,
                         delete_after: bool=True, smiles_type='standard_smiles'
                         ) -> Tuple[dc.data.DiskDataset, pd.Series]:
    """
    Load DOCKSTRING dataset as a DeepChem dataset.

    Args:
        debugging(bool): if True, load small (n=100) dataset for fast debugging.
        featurization(str, optional): either one of 'smiles', 'mol', 'graph', 'morgan'
            or 'long_morgan' (default 'graph').
        y_type(str or list or tuple, optional): type of data in the y values. Either one of the following, or
            a list or tuple of the following:
            - 'original': original type of data in the dataset (e.g. antibiotic activity, docking scores, etc).
            - 'rdkit': RDKit molecular descriptors (number of functional groups, logP, polar surface area, etc).
            - 'fp_morgan': Morgan fingerprints, radius 3, length 2048.
            - 'fp_morgan_long': Morgan fingerprints, radius 3, length 4096.
            - 'fp_rdkit': RDKit fingerprints, max path 6.
            - 'fp_rdkit_long': RDKit fingerprints, max path 6, length 4096.
        data_dir(str or pathlib.Path, optional): directory where dataset should be written.
        delete_after(bool): if True, __del__ will be customized so as to attempt to delete the data directory
            when the dataset is deleted.

    Returns:
        DeepChem DiskDataset with X type determined by 'featurization' and Y type determined by 'y_type'.
    """
    # Load data
    df = dataframes.load_dockstring_dataframe(debug=debug)
    smiles = df[smiles_type].values.reshape(-1,1)
    scores = df.drop(columns=['standard_smiles',
                              'smiles', 'numeric_index', 'inchikey']).values
    numeric_index = df['numeric_index']
    # Get dataset
    dataset = data_helpers.create_dataset(smiles=smiles,original_data=scores, y_type=y_type,
                           featurization=featurization, data_dir=dataset_path, ids=numeric_index,
                           delete_after=delete_after, should_weigh_out_nans=True)
    return dataset, numeric_index


def create_red_data(debug=False, featurization='graph', dataset_path=None,
                  delete_after=True, smiles_type='standard_smiles') -> Tuple[dc.data.DiskDataset, pd.Series]:
    # Load data
    df = dataframes.load_red_dataframe(debug=debug)
    smiles = df[smiles_type].values.reshape(-1,1)
    values = df.drop(columns=['standard_smiles', 'minimal_standard_smiles',
                              'inchikey', 'smiles',
                              'numeric_index']).astype(float).values
    numeric_index = df['numeric_index']
    # Get dataset
    dataset = data_helpers.create_dataset(smiles=smiles, original_data=values, y_type='original',
                           featurization=featurization, data_dir=dataset_path, ids=numeric_index,
                           delete_after=delete_after, should_weigh_out_nans=True)
    return dataset, numeric_index


def create_chemdiv_data(debug: bool=False,
                      featurization: str='graph', dataset_path: Union[str, Path]=None,
                      delete_after: bool=True, smiles_type='standard_smiles') -> Tuple[dc.data.DiskDataset, pd.Series]:
    # Load data
    df = dataframes.load_chemdiv_dataframe(debug=debug)
    smiles = df[smiles_type].values.reshape(-1,1)
    values = np.zeros(smiles.shape) # Zeros to distinguish y's from w's, which are ones
    numeric_index = df['numeric_index']
    # Get dataset
    dataset = data_helpers.create_dataset(smiles=smiles, original_data=values, y_type='original',
                           featurization=featurization, ids=numeric_index, data_dir=dataset_path,
                           delete_after=delete_after, should_weigh_out_nans=False)
    return dataset, numeric_index


def create_in_memory_dataset(smiles: Iterable[str],
                             y: Optional[Iterable[float]] = None,
                             ids: Optional[Iterable[str]] = None,
                             featurization: str = 'graph') -> dc.data.NumpyDataset:
    featurizer = data_helpers.get_featurizer(featurization)
    features = featurizer.featurize(smiles)
    dataset = dc.data.NumpyDataset(X=features, y=y, ids=ids)
    return dataset