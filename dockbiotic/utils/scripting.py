import numpy as np
import pandas as pd
import random
import torch
from dockbiotic.data import dataframes


def set_random_seed(random_seed):
    # Get a high-quality random seed
    np.random.seed(seed=random_seed)
    random_seed = np.random.randint(0,20000)
    # Set every random algorithm you can think of to that seed
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)


def coadd_index_to_smiles(coadd_index: str, coadd: pd.DataFrame) -> str:
    return coadd.loc[coadd_index,'standard_smiles']


def stokes_index_to_smiles(stokes_index: str, stokes: pd.DataFrame) -> str:
    return stokes.loc[stokes_index,'standard_smiles']