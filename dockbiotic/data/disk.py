from typing import Union, Tuple
import pandas as pd
import deepchem as dc
from dockbiotic.data import create
from dockbiotic.data import directories
from dockbiotic.utils import save_load


def load_stokes_data_from_disk(binarize: bool=False, featurization: str='graph', 
                               y_type: str='original',
                               smiles_type: str='standard_smiles',
                               debug: bool=False) -> Tuple[dc.data.DiskDataset, pd.Series]:
    # locate the dataset
    dataset_path, param_str = directories.get_stokes_directory(binarize=binarize,
                                        featurization=featurization,
                                        y_type=y_type, smiles_type=smiles_type,
                                        debug=debug)
    # if it exists in disk, load it and return it
    if dataset_path.exists():
        ds = dc.data.DiskDataset(data_dir=dataset_path)
        idx = save_load.load_dataset_indices(dataset_path=dataset_path)
    # if not, create it
    else:
        print('Creating Stokes dataset')
        ds, idx = create.create_stokes_data(binarize=binarize, featurization=featurization,
                         y_type=y_type, smiles_type=smiles_type,
                         delete_after=False,
                         debug=debug,
                         dataset_path=dataset_path)
        save_load.save_dataset_indices(dataset_path=dataset_path, idx=idx)
        save_load.save_parameter_string(dataset_path=dataset_path, param_str=param_str)
        print('Finished creating Stokes dataset')

    return ds, idx



def load_coadd_data_from_disk(strain: str, discard_above: Union[None, float]=None,
                              binarize: bool=False, featurization: str='graph',
                              y_type: str='original',
                              smiles_type: str='standard_smiles',
                              debug: bool=False) -> Tuple[dc.data.DiskDataset, pd.Series]:
    # locate the dataset
    dataset_path, param_str = directories.get_coadd_directory(strain, discard_above=discard_above,
                                       binarize=binarize,
                                       featurization=featurization,
                                       y_type=y_type,
                                       smiles_type=smiles_type,
                                       debug=debug)
    # if it exists in disk, load it
    if dataset_path.exists():
        ds = dc.data.DiskDataset(data_dir=dataset_path)
        idx = save_load.load_dataset_indices(dataset_path=dataset_path)
    # if not, create it
    else:
        print('Creating COADD dataset')
        ds, idx = create.create_coadd_data(binarize=binarize, strain=strain,
                        discard_above=discard_above,
                        featurization=featurization,
                        y_type=y_type, smiles_type=smiles_type,
                        delete_after=False,
                        debug=debug,
                        dataset_path=dataset_path)
        save_load.save_dataset_indices(dataset_path=dataset_path, idx=idx)
        save_load.save_parameter_string(dataset_path=dataset_path, param_str=param_str)
        print('Finished creating COADD dataset')

    return ds, idx


def load_stokes_and_coadd_data_from_disk(binarize: bool=False,
                            balance: bool=False,
                            featurization: str='graph', y_type: str='original',
                            smiles_type: str='standard_smiles',
                            debug: bool=False) -> Tuple[dc.data.DiskDataset, pd.Series]:
    # locate the dataset
    dataset_path, param_str = directories.get_stokes_and_coadd_directory(binarize=binarize,
                                                  balance=balance,
                                                  featurization=featurization,
                                                  y_type=y_type,
                                                  smiles_type=smiles_type,
                                                  debug=debug)
    # if it exists in disk, load it
    if dataset_path.exists():
        ds = dc.data.DiskDataset(data_dir=dataset_path)
        idx = save_load.load_dataset_indices(dataset_path=dataset_path)
    # if not, create it
    else:
        print('Creating Stokes+COADD dataset')
        ds, idx = create.create_stokes_and_coadd_data(binarize=binarize,
                        balance=balance,
                        featurization=featurization,
                        y_type=y_type, smiles_type=smiles_type,
                        delete_after=False,
                        debug=debug,
                        dataset_path=dataset_path)
        save_load.save_dataset_indices(dataset_path=dataset_path, idx=idx)
        save_load.save_parameter_string(dataset_path=dataset_path, param_str=param_str)
        print('Finished creating Stokes+COADD dataset')

    return ds, idx


def load_rdkit_data_from_disk(featurization: str='graph',
                              smiles_type: str='standard_smiles',
                              debug: bool=False) -> Tuple[dc.data.DiskDataset, pd.Series]:
    # Locate the dataset
    dataset_path, param_str = directories.get_rdkit_directory(featurization=featurization,
                                       smiles_type=smiles_type,
                                       debug=debug)
    # if it exists in disk, load it
    if dataset_path.exists():
        ds = dc.data.DiskDataset(data_dir=dataset_path)
        idx = save_load.load_dataset_indices(dataset_path=dataset_path)
    # if not, create it
    else:
        print('Creating RDKit dataset')
        ds, idx = create.create_rdkit_data(featurization=featurization,
                        smiles_type=smiles_type,
                        delete_after=False,
                        debug=debug,
                        dataset_path=dataset_path)
        save_load.save_dataset_indices(dataset_path=dataset_path, idx=idx)
        save_load.save_parameter_string(dataset_path=dataset_path, param_str=param_str)
        print('Finished creating RDKit dataset')

    return ds, idx


def load_excape_data_from_disk(featurization: str='graph',
                               smiles_type: str='standard_smiles',
                              debug: bool=False) -> Tuple[dc.data.DiskDataset, pd.Series]:
    # Locate the dataset
    dataset_path, param_str = directories.get_excape_directory(featurization=featurization,
                                        smiles_type=smiles_type,
                                        debug=debug)
    # if it exists in disk, load it
    if dataset_path.exists():
        ds = dc.data.DiskDataset(data_dir=dataset_path)
        idx = save_load.load_dataset_indices(dataset_path=dataset_path)
    # if not, create it
    else:
        print('Creating EXCAPE dataset')
        ds, idx = create.create_excape_data(featurization=featurization,
                        smiles_type=smiles_type,
                        delete_after=False,
                        debug=debug,
                        dataset_path=dataset_path)
        save_load.save_dataset_indices(dataset_path=dataset_path, idx=idx)
        save_load.save_parameter_string(dataset_path=dataset_path, param_str=param_str)
        print('Finished creating EXCAPE dataset')

    return ds, idx


def load_dockstring_data_from_disk(featurization: str='graph',
                                   smiles_type: str='standard_smiles',
                                   debug: bool=False) -> Tuple[dc.data.DiskDataset, pd.Series]:
    # Locate the dataset
    dataset_path, param_str = directories.get_dockstring_directory(featurization=featurization,
                                            smiles_type=smiles_type,
                                            debug=debug)
    # if it exists in disk, load it
    if dataset_path.exists():
        ds = dc.data.DiskDataset(data_dir=dataset_path)
        idx = save_load.load_dataset_indices(dataset_path=dataset_path)
    # if not, create it
    else:
        print('Creating dockstring dataset')
        ds, idx = create.create_dockstring_data(featurization=featurization,
                        smiles_type=smiles_type,
                        delete_after=False,
                        debug=debug,
                        dataset_path=dataset_path)
        save_load.save_dataset_indices(dataset_path=dataset_path, idx=idx)
        save_load.save_parameter_string(dataset_path=dataset_path, param_str=param_str)
        print('Finished creating dockstring dataset')

    return ds, idx


def load_red_data_from_disk(featurization: str='graph',
                            smiles_type: str='standard_smiles',
                            debug: bool=False) -> Tuple[dc.data.DiskDataset, pd.Series]:
    # locate the dataset
    dataset_path, param_str = directories.get_red_directory(featurization=featurization,
                                     smiles_type=smiles_type,
                                       debug=debug)
    # if it exists in disk, load it
    if dataset_path.exists():
        ds = dc.data.DiskDataset(data_dir=dataset_path)
        idx = save_load.load_dataset_indices(dataset_path=dataset_path)
    # if not, create it
    else:
        print('Creating RED (RDKit+EXCAPE+dockstring) dataset')
        ds, idx = create.create_red_data(featurization=featurization,
                        smiles_type=smiles_type,
                        delete_after=False,
                        debug=debug,
                        dataset_path=dataset_path)
        save_load.save_dataset_indices(dataset_path=dataset_path, idx=idx)
        save_load.save_parameter_string(dataset_path=dataset_path, param_str=param_str)
        print('Finished creating RED (RDKit+EXCAPE+dockstring) dataset')

    return ds, idx



def load_chemdiv_data_from_disk(featurization: str='graph',
                                smiles_type: str='standard_smiles',
                                debug: bool=False) -> Tuple[dc.data.DiskDataset, pd.Series]:
    # locate the dataset
    dataset_path, param_str = directories.get_chemdiv_directory(featurization=featurization,
                                         smiles_type=smiles_type,
                                         debug=debug)
    # if it exists in disk, load it
    if dataset_path.exists():
        ds = dc.data.DiskDataset(data_dir=dataset_path)
        try:
            idx = save_load.load_dataset_indices(dataset_path=dataset_path)
        except FileNotFoundError as e:
            print('Indices file indices.pkl does not exist')
            print(e)
            idx = None
    # if not, create it
    else:
        print('Creating ChemDiv dataset')
        ds, idx = create.create_chemdiv_data(featurization=featurization,
                                    smiles_type=smiles_type,
                                    delete_after=False, debug=debug,
                                    dataset_path=dataset_path)
        save_load.save_dataset_indices(dataset_path=dataset_path, idx=idx)
        save_load.save_parameter_string(dataset_path=dataset_path, param_str=param_str)
        print('Finished creating ChemDiv dataset')
    
    return ds, idx