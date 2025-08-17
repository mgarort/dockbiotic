import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
import numpy as np
import shutil
import deepchem as dc
from dockbiotic.utils import chemistry
import uuid
from pathlib import Path
from typing import Tuple

# Featurizer for X

def get_featurizer(featurization: str) -> dc.feat.Featurizer:
    """
    Get DeepChem featurizer.
    """
    if featurization == 'smiles':
        featurizer = dc.feat.RawFeaturizer(smiles=True)
    elif featurization == 'mol':
        featurizer = dc.feat.RawFeaturizer(smiles=False)
    elif featurization == 'graph':
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    elif featurization == 'morgan':
        featurizer = dc.feat.CircularFingerprint(radius=2, size=2048)
    elif featurization == 'long_morgan':
        featurizer = dc.feat.CircularFingerprint(radius=2, size=4096)
    else:
        raise ValueError(f'argument "featurization" should be one of ' \
                          '"smiles", "mol", "graph", "morgan" or "long_morgan"' \
                          'but you supplied {featurization}.')
    return featurizer


# Function for getting Y data of any type

def get_y(smiles ,original_data,y_type,parallelize: bool = False):
    """
    Get y data from an array of SMILES, from one of the following types:
    - 'original': original type of data in the dataset (e.g. antibiotic activity, docking scores, etc).
    - 'rdkit': RDKit molecular descriptors (number of functional groups, logP, polar surface area, etc).
    - 'fp_morgan': Morgan fingerprints, radius 3, length 2048.
    - 'fp_morgan_long': Morgan fingerprints, radius 3, length 4096.
    - 'fp_rdkit': RDKit fingerprints, max path 6.
    - 'fp_rdkit_long': RDKit fingerprints, max path 6, length 4096.
    """
    # If several y types, return a contatenation of y's
    if isinstance(y_type,list) or isinstance(y_type,tuple):
        y = []
        for each_y_type in y_type:
            each_y = get_y(smiles, original_data, each_y_type)
            y.append(each_y)
        y = np.hstack(y)
    # If a single y type, return the single y
    elif y_type == 'original':
        y = original_data
    elif y_type == 'rdkit':
        y = chemistry.get_rdkit_descriptors(smiles, parallelize=parallelize)
    elif y_type == 'fp_rdkit':
        y = chemistry.get_rdkit_fp(smiles=smiles, length=2048, parallelize=parallelize)
    elif y_type == 'fp_rdkit_long':
        y = chemistry.get_rdkit_fp(smiles=smiles, length=4096, parallelize=parallelize)
    elif y_type == 'fp_morgan':
        y = chemistry.get_morgan_fp(smiles=smiles, length=2048, parallelize=parallelize)
    elif y_type == 'fp_morgan_long':
        y = chemistry.get_morgan_fp(smiles=smiles, length=4096, parallelize=parallelize)
    return y

def filter_out_nans(smiles,y):
    '''
    Function for filtering out NaNs.
    '''
    mask_nan = np.isnan(y).any(axis=1)
    smiles = smiles[~mask_nan]
    y = y[~mask_nan]
    return smiles, y

def weigh_out_nans(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for assigning 0 weights to NaN values in y, and replacing them with 0s.
    """
    mask_nan = np.isnan(y)
    w = (~mask_nan).astype(float)
    y = np.nan_to_num(y)
    return y, w

def create_data_dir(data_dir=None):
    '''
    Method to create and get path to dataset directory.
    '''
    # if not specified by the user, get random dataset path in DEEPCHEM_DATA_DIR
    if data_dir is None:
        deepchem_data_dir = Path(dc.utils.get_data_dir())
        data_dirname = str(uuid.uuid4())
        data_dir = deepchem_data_dir / data_dirname
    # if specified by the user, use that dataset path
    if isinstance(data_dir,str):
        data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def disk_dataset_del_(self):
    '''
    Method to delete the dataset directory once the dataset object stops existing.
    '''
    shutil.rmtree(str(self.data_dir))


def create_dataset(smiles, original_data, y_type, featurization,
                 data_dir, delete_after, ids=None, should_weigh_out_nans=False,
                 parallelize=False) -> dc.data.DiskDataset:
    '''
    Get data given the SMILES, original data, desired y_type (for Y) and
    desired featurization (for X).
    '''
    # Define y values
    y = get_y(smiles=smiles,original_data=original_data,y_type=y_type,parallelize=parallelize)
    # Filter out NaNs only if we won't weigh them out
    if not should_weigh_out_nans:
        smiles, y = filter_out_nans(smiles,y)
    # Define featurizer
    featurizer = get_featurizer(featurization)
    # Define data loader
    all_tasks = []
    for i in range(1,y.shape[1]+1):
        all_tasks.append(f'task{i}')
    loader = dc.data.InMemoryLoader(tasks=all_tasks, featurizer=featurizer)
    # Create dataset directory
    data_dir = create_data_dir(data_dir=data_dir)
    # Define dataset
    if should_weigh_out_nans:
        y, w = weigh_out_nans(y)
    else:
        w = np.ones(y.shape)
    if ids is None:
        ids = np.array(range(len(y)))
    dataset = loader.create_dataset(zip(smiles, y, w, ids), shard_size=1000, data_dir=data_dir)
    if delete_after:
        # Manually set __del__ so that the dataset directory is deleted after usage
        dataset.__class__ = type('AutodeletedDiskDataset',(dc.data.datasets.DiskDataset,),{'__del__':disk_dataset_del_})
    return dataset




# Function for balancing actives and inactives by resampling the actives

def balance_dataframe(df, resampling_factor, activity_threshold):
    """
    Function for balancing actives and inactives by resampling the actives.

    Args:
        - df(pandas.core.frame.DataFrame): dataframe to balance by resampling
            actives. It should contain the column "processed_inhibition".
        - active_threshold(float): value of "processed_inhibition" above
            which compounds should be considered active and resampled.

    Returns:
        pandas dataframe with the actives repeated by the "resampling_factor"
    """
    mask_inactive = df['processed_inhibition'] < activity_threshold
    df_inactive = df.loc[mask_inactive]
    mask_active = df['processed_inhibition'] >= activity_threshold
    df_active = df.loc[mask_active]
    return pd.concat([df_inactive] + [df_active] * resampling_factor).reset_index()
