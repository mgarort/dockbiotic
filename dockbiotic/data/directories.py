import hashlib
from pathlib import Path
import deepchem as dc
from typing import Union, Tuple


def get_coadd_directory(strain: str, discard_above: Union[None, float] = None, binarize: bool = False,
                        featurization: str = 'graph', y_type: str = 'original',
                        smiles_type: str = 'standard_smiles',
                        debug: bool = False) -> Path:
    deepchem_data_dir = Path(dc.utils.get_data_dir())
    param_str = f'coadd{strain}{discard_above}{binarize}{featurization}{y_type}{smiles_type}{debug}'
    identifier = hashlib.md5(str(param_str).encode()).hexdigest()
    dataset_path = deepchem_data_dir / identifier
    return dataset_path, param_str

def get_stokes_directory(binarize: bool = False, featurization: str = 'graph', y_type: str = 'original',
                         smiles_type: str = 'standard_smiles',
                        debug: bool = False) -> Path:
    deepchem_data_dir = Path(dc.utils.get_data_dir())
    param_str = f'stokes{binarize}{featurization}{y_type}{smiles_type}{debug}'
    identifier = hashlib.md5(str(param_str).encode()).hexdigest()
    dataset_path = deepchem_data_dir / identifier
    return dataset_path, param_str

def get_stokes_and_coadd_directory(binarize: bool = False, balance: bool = False,
                                   featurization: str = 'graph', y_type: str = 'original',
                                   smiles_type: str = 'standard_smiles',
                        debug: bool = False) -> Path:
    deepchem_data_dir = Path(dc.utils.get_data_dir())
    param_str = f'stokes_and_coadd{binarize}{balance}{featurization}{y_type}{smiles_type}{debug}'
    identifier = hashlib.md5(str(param_str).encode()).hexdigest()
    dataset_path = deepchem_data_dir / identifier
    return dataset_path, param_str

def get_rdkit_directory(featurization: str = 'graph', smiles_type: str = 'standard_smiles',
                        debug: bool = False) -> Path:
    deepchem_data_dir = Path(dc.utils.get_data_dir())
    param_str = f'rdkit{featurization}{smiles_type}{debug}'
    identifier = hashlib.md5(str(param_str).encode()).hexdigest()
    dataset_path = deepchem_data_dir / identifier
    return dataset_path, param_str

def get_excape_directory(featurization: str = 'graph', smiles_type: str = 'standard_smiles',
                        debug: bool = False) -> Path:
    deepchem_data_dir = Path(dc.utils.get_data_dir())
    param_str = f'excape{featurization}{smiles_type}{debug}'
    identifier = hashlib.md5(str(param_str).encode()).hexdigest()
    dataset_path = deepchem_data_dir / identifier
    return dataset_path, param_str

def get_dockstring_directory(featurization: str = 'graph', smiles_type: str = 'standard_smiles',
                        debug: bool = False) -> Path:
    deepchem_data_dir = Path(dc.utils.get_data_dir())
    param_str = f'dockstring{featurization}{smiles_type}{debug}'
    identifier = hashlib.md5(str(param_str).encode()).hexdigest()
    dataset_path = deepchem_data_dir / identifier
    return dataset_path, param_str

def get_red_directory(featurization: str = 'graph', smiles_type: str = 'standard_smiles',
                        debug: bool = False) -> Tuple[Path, str]:
    deepchem_data_dir = Path(dc.utils.get_data_dir())
    param_str = f'red{featurization}{smiles_type}{debug}'
    identifier = hashlib.md5(str(param_str).encode()).hexdigest()
    dataset_path = deepchem_data_dir / identifier
    return dataset_path, param_str

def get_chemdiv_directory(featurization: str='graph',
                          smiles_type='standard_smiles',
                          debug=False) -> Tuple[Path, str]:
    deepchem_data_dir = Path(dc.utils.get_data_dir())
    param_str = f'chemdiv{featurization}{smiles_type}{debug}'
    identifier = hashlib.md5(str(param_str).encode()).hexdigest()
    dataset_path = deepchem_data_dir / identifier
    return dataset_path, param_str