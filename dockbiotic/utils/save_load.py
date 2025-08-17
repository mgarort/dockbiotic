
from pathlib import Path
import pandas as pd
from typing import Union
import pickle
import logging
from dockbiotic.models import AttentiveFPModel
from deepchem.models.gbdt_models import GBDTModel


def save_dataset_indices(dataset_path: Path, idx: pd.Series) -> None:
    path = dataset_path / 'indices.pkl'
    with open(path, 'bw') as f:
        pickle.dump(idx, f)


def save_parameter_string(dataset_path: Path, param_str: str):
    path = dataset_path / 'parameter_string.txt'
    with open(path, 'w') as f:
        f.write(param_str)


def load_dataset_indices(dataset_path: Path) -> pd.Series:
    path = dataset_path / 'indices.pkl'
    with open(path, 'br') as f:
        idx = pickle.load(f)
    return idx


def save_model(model: Union[GBDTModel, AttentiveFPModel, None],
               model_dir: str) -> None:
    """Save model to disk."""
    
    model_dir = Path(model_dir)
    
    if model is None:
        logging.info('Not saving. Model is None.')

    elif isinstance(model, GBDTModel):
        if not model_dir.exists():
            model_dir.mkdir()
        setattr(model, 'model_dir', model_dir)
        model.save()

    elif isinstance(model, AttentiveFPModel):
        model.save_checkpoint(max_checkpoints_to_keep=1,
                              model_dir=model_dir)

    else:
        raise TypeError('Unknown model class')