from typing import Optional, Tuple, List, Union, Iterable
from copy import deepcopy
from deepchem.models.gbdt_models import GBDTModel
import torch
from xgboost import XGBRegressor, XGBClassifier
import deepchem as dc
import pandas as pd
import numpy as np
import logging
from dockbiotic.models import AttentiveFPModel
from dockbiotic.data import num_tasks
from dockbiotic.data import disk
from dockbiotic.data import create
import warnings


def get_dataset(dataset_name: str, featurization: str,
                mode: Optional[str]=None,
                smiles_type: str='standard_smiles',
                debug: bool=False) -> Tuple[dc.data.DiskDataset, pd.Series]:
    """
    setting will be ignored if not stokes, coadd or stokes_coadd.
    """

    if dataset_name is None:
        ds, idx = None, None

    # pretraining datasets
    elif dataset_name == 'rdkit':
        if mode is not None:
            warnings.warn('setting ignored')
        ds, idx = disk.load_rdkit_data_from_disk(featurization=featurization,
                                            smiles_type=smiles_type,
                                            debug=debug)

    elif dataset_name == 'excape':
        if mode is not None:
            warnings.warn('setting ignored')
        ds, idx = disk.load_excape_data_from_disk(featurization=featurization,
                                             smiles_type=smiles_type,
                                             debug=debug)

    elif dataset_name == 'dockstring':
        if mode is not None:
            warnings.warn('setting ignored')
        ds, idx = disk.load_dockstring_data_from_disk(featurization=featurization,
                                                 smiles_type=smiles_type,
                                                 debug=debug)

    elif dataset_name == 'red':
        if mode is not None:
            warnings.warn('setting ignored')
        ds, idx = disk.load_red_data_from_disk(featurization=featurization,
                                          smiles_type=smiles_type,
                                          debug=debug)
        
    # finetuning datasets
    elif dataset_name == 'stokes':
        if mode is None:
            raise ValueError('Argument "mode" required to load Stokes dataset.')
        binarize = (mode == 'classification')
        ds, idx = disk.load_stokes_data_from_disk(binarize=binarize,
                                featurization=featurization,
                                smiles_type=smiles_type,
                                debug=debug)

    elif dataset_name == 'coadd':
        if mode is None:
            raise ValueError('Argument "mode" required to load COADD dataset.')
        binarize = (mode == 'classification')
        ds, idx = disk.load_coadd_data_from_disk(strain='atcc25922',
                                discard_above=0.9, binarize=binarize,
                                featurization=featurization,
                                smiles_type=smiles_type,
                                debug=debug)

    elif dataset_name == 'stokes_and_coadd':
        if mode is None:
            raise ValueError('Argument "mode" required to load Stokes+COADD dataset.')
        binarize = (mode == 'classification')
        ds, idx = disk.load_stokes_and_coadd_data_from_disk(
                                binarize=binarize,
                                balance=False,
                                featurization=featurization,
                                smiles_type=smiles_type,
                                debug=debug
                            )

    elif dataset_name == 'stokes_and_coadd_balanced':
        if mode is None:
            raise ValueError('Argument "mode" required to load Stokes+COADD dataset.')
        binarize = (mode == 'classification')
        ds, idx = disk.load_stokes_and_coadd_data_from_disk(
                                binarize=binarize,
                                balance=True,
                                featurization=featurization,
                                smiles_type=smiles_type,
                                debug=debug
                            )

    # libraries
    elif dataset_name == 'chemdiv':
        if mode is not None:
            warnings.warn('Argument "mode" ignored for dataset ChemDiv.')
        ds, idx = disk.load_chemdiv_data_from_disk(featurization=featurization,
                                              smiles_type=smiles_type,
                                              debug=debug)

    else:
        raise ValueError('Incorrect dataset name.')

    return ds, idx


def get_attentive_fp(dataset_name: str,
                     mode: Optional[str]=None,
                     batch_size: int=64,
                     learning_rate: float=0.001,
                     c_fp: Optional[float]=None,
                     c_fn: Optional[float]=None,
                     pretrained_model_dir: Optional[str]=None,
                     ) -> AttentiveFPModel:
    """
    dataset_name: name of dataset to apply the model to. The architecture of the
        model may depend on the dataset chosen (the number of tasks -and so the
        width of the last layer- can change).
    
    mode: "regression", "classification" or "antibiotic_classification". Some
        datasets can be used with a single setting (e.g. RDKit is only used with
        regression). In those cases, the "setting" argument will have no effect.
        Similarly, if setting is not "antibiotic_classification", c_fp and c_fn
        will have no effect in the model. So these are only passed when dealing
        with stokes, coadd or stokes_coadd.

    pretrained_model_dir: to load a model that was previously trained and saved.
    """

    if dataset_name is None:
        model = None

    elif dataset_name in ['stokes', 'coadd', 'stokes_and_coadd', 'stokes_and_coadd_balanced']:
        model = AttentiveFPModel(mode=mode,
                                 n_tasks=1,
                                 batch_size=batch_size,
                                 learning_rate=learning_rate,
                                 c_fp=c_fp,
                                 c_fn=c_fn,
                                 model_dir=pretrained_model_dir,
                                )

    elif dataset_name == 'rdkit':
        if mode is not None:
            warnings.warn('Argument "mode" ignored for dataset RDKit (always regression).')
        n_tasks = num_tasks.get_rdkit_num_tasks()
        model = AttentiveFPModel(mode='regression',
                                 n_tasks=n_tasks,
                                 batch_size=batch_size,
                                 learning_rate=learning_rate,
                                 model_dir=pretrained_model_dir,
                                 )

    elif dataset_name == 'excape':
        if mode is not None:
            warnings.warn('Argument "mode" ignored for dataset ExCAPE (always classification).')
        n_tasks = num_tasks.get_excape_num_tasks()
        model = AttentiveFPModel(mode='classification',
                                 n_tasks=n_tasks,
                                 batch_size=batch_size,
                                 learning_rate=learning_rate,
                                 model_dir=pretrained_model_dir,
                                 )

    elif dataset_name == 'dockstring':
        if mode is not None:
            warnings.warn('Argument "mode" ignored for dataset dockstring (always regression).')
        n_tasks = num_tasks.get_dockstring_num_tasks()
        model = AttentiveFPModel(mode='regression',
                                 n_tasks=n_tasks,
                                 batch_size=batch_size,
                                 learning_rate=learning_rate,
                                 model_dir=pretrained_model_dir,
                                 )

    elif dataset_name == 'red':
        if mode is not None:
            warnings.warn('Argument "mode" ignored for dataset RED (RDKit always regression, ExCAPE always classification, dockstring always regression).')
        r_n_tasks = num_tasks.get_rdkit_num_tasks()
        e_n_tasks = num_tasks.get_excape_num_tasks()
        d_n_tasks = num_tasks.get_dockstring_num_tasks()
        n_tasks = r_n_tasks + e_n_tasks + d_n_tasks
        r_slice = slice(0, r_n_tasks)
        e_slice = slice(r_n_tasks, r_n_tasks + e_n_tasks)
        d_slice = slice(r_n_tasks + e_n_tasks, r_n_tasks + e_n_tasks + d_n_tasks)
        ds, _ = disk.load_red_data_from_disk()
        model = AttentiveFPModel(
                        mode='red',
                        r_slice=r_slice,
                        e_slice=e_slice,
                        d_slice=d_slice,
                        n_tasks=n_tasks,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        model_dir=pretrained_model_dir
                    )

    if pretrained_model_dir is not None:
        model.restore()

    return model


def train_attentive_fp(model: AttentiveFPModel, dataset: dc.data.DiskDataset,
                       num_epochs: int, parameters_to_train: Optional[List[str]] = None,
                       pretrained_model: Optional[AttentiveFPModel] = None,
                       verbose: bool = False, debug: bool = True) -> AttentiveFPModel:
    if verbose:
        original_level = logging.root.level
        logging.getLogger().setLevel(logging.INFO)

    # we allow passsing None for model and dataset. This allows consistency when
    # not pretraining
    if model is None and dataset is None:
        return None
    # load weights of pretrained model
    if pretrained_model is not None:
        pretrained_weights = deepcopy(pretrained_model.model.state_dict())
        initialized_weights = deepcopy(model.model.state_dict())
        pretrained_weights['model.predict.1.weight'] = initialized_weights['model.predict.1.weight']
        pretrained_weights['model.predict.1.bias'] = initialized_weights['model.predict.1.bias']
        model.model.load_state_dict(pretrained_weights)

    # train
    # allow interrupting training gracefully with Ctrl-C
    try:
        if parameters_to_train is None:
            variables = None
        else:
            variables = []
            for name, param in model.model.named_parameters():
                if name in parameters_to_train:
                    variables.append(param)
        state_dict_before = deepcopy(model.model.state_dict())
        model.fit(dataset, nb_epoch=num_epochs, variables=variables)
        state_dict_after = deepcopy(model.model.state_dict())

    except KeyboardInterrupt:
        pass

    if debug:
        unfrozen_values_before = []
        frozen_values_before = []
        for name, param in state_dict_before.items():
            if 'readout' in name or 'predict' in name:
                unfrozen_values_before.append(param.detach().cpu().numpy())
            else:
                frozen_values_before.append(param.detach().cpu().numpy())
        unfrozen_values_after = []
        frozen_values_after = []
        for name, param in state_dict_after.items():
            if 'readout' in name or 'predict' in name:
                unfrozen_values_after.append(param.detach().cpu().numpy())
            else:
                frozen_values_after.append(param.detach().cpu().numpy())

        unfrozen_change = [np.abs(elem_after - elem_before).sum()
                        for elem_after, elem_before in 
                        zip(unfrozen_values_after, unfrozen_values_before)]
        frozen_change = [np.abs(elem_after - elem_before).sum()
                        for elem_after, elem_before in 
                        zip(frozen_values_after, frozen_values_before)]
        unfrozen_change = np.array(unfrozen_change).sum()
        frozen_change = np.array(frozen_change).sum()
        print('Unfrozen change', unfrozen_change)
        print('Frozen change', frozen_change)

    if verbose:
        logging.getLogger().setLevel(original_level)

    return model


def get_xgb(mode: str) -> GBDTModel:
    objective_catalog = {'regression':'reg:squarederror',
                            'classification':'binary:logistic'}
    xgb_model_catalog = {'regression':XGBRegressor,
                            'classification':XGBClassifier}
    objective = objective_catalog[mode]
    model = xgb_model_catalog[mode]
    model = GBDTModel(model(objective=objective))
    return model


def get_model(model_type: str,
             dataset_name: Optional[str]=None,
             mode: Optional[str]=None,
             batch_size: int=64,
             learning_rate: float=0.001,
             c_fp: Optional[float]=None,
             c_fn: Optional[float]=None,
             pretrained_model_dir: Optional[str]=None,
            ) -> Union[GBDTModel, AttentiveFPModel]:
    
    if model_type == 'xgb':
        if mode is None:
            raise ValueError('Argument "mode" required for XGB model.')
        model = get_xgb(mode=mode)

    elif model_type == 'attentive_fp':
        if dataset_name is None:
            raise ValueError('Argument "dataset_name" required for AttentiveFP model.')
        model = get_attentive_fp(dataset_name=dataset_name,
                                 mode=mode,
                                 batch_size=batch_size,
                                 learning_rate=learning_rate,
                                 c_fp=c_fp,
                                 c_fn=c_fn,
                                 pretrained_model_dir=pretrained_model_dir,
                                )
    else:
        raise ValueError('Incorrect model_type')
    
    return model


def train_model(model_name: str,
                dataset_name: str,
                featurization: str = 'graph',
                mode: Optional[str] = None,
                smiles_type: str = 'standard_smiles',
                num_epochs: Optional[int] = None,
                batch_size: int = 64,
                lr: float = 1e-3,
                c_fp: Optional[float] = None,
                c_fn: Optional[float] = None,
                parameters_to_train: Optional[List[torch.nn.Parameter]] = None,
                debug: bool = False,
                verbose: bool = False,
                pretrained_model_dir: Optional[str] = None,
                pretraining_dataset_name: Optional[str] = None,
               ) -> Union[GBDTModel, AttentiveFPModel]:

    """
    - parameters_to_train: names of parameters to train during the fine-tuning
        (during the pre-training all variables will be trained).
    - model_dir: if provided, this model will be loaded previous to
        pre-training. Assumes that the model type is AttentiveFP.
    """

    if model_name == 'xgb':
        # if xgb, we ignore: num_epochs,
        # batch size, lr, c_fp and c_fn...
        model = get_xgb(mode=mode)
        ds, _ = get_dataset(dataset_name=dataset_name,
                              featurization=featurization,
                              mode=mode,
                              smiles_type=smiles_type,
                              debug=debug)
        model.fit(dataset=ds)

    elif model_name == 'attentive_fp':
        
        # initialize pretrained model (if applicable)
        if pretrained_model_dir is not None:
            logging.info(f'Loading pretrained model {pretrained_model_dir},' \
                         f'which was trained with dataset {pretraining_dataset_name}')
            pretrained_model = get_attentive_fp(
                                 dataset_name=pretraining_dataset_name,
                                 pretrained_model_dir=pretrained_model_dir
                                )
        else:
            pretrained_model = None

        # get dataset and model
        ds, _ = get_dataset(dataset_name=dataset_name,
                            featurization=featurization,
                            mode=mode,
                            smiles_type=smiles_type,
                            debug=debug)
        model = get_attentive_fp(dataset_name=dataset_name,
                                 batch_size=batch_size,
                                 learning_rate=lr,
                                 mode=mode,
                                 c_fp=c_fp, c_fn=c_fn,
                                )

        # train
        model = train_attentive_fp(
                    model=model,
                    dataset=ds,
                    num_epochs=num_epochs,
                    parameters_to_train=parameters_to_train,
                    pretrained_model=pretrained_model,
                    verbose=verbose,
                    debug=debug
                    )

    else:
        raise ValueError('Incorrect model_name')

    return model


def predict_dataset(ds: dc.data.DiskDataset,
                    model: Union[GBDTModel, AttentiveFPModel]) -> np.ndarray:

    with torch.no_grad():
        return model.predict(ds)


def predict_smiles(smiles: Iterable[str],
                   model: Union[GBDTModel, AttentiveFPModel],
                   ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Predict a list of smiles and return predictions in a dataframe.
    
    The list of smiles should not be too long since we'll use an in-memory dataset.
    """

    df = pd.DataFrame({'smiles': smiles})
    if ids is not None:
        df['id'] = ids
    ds = create.create_in_memory_dataset(
                        smiles=smiles,
                        ids=ids,
                        y=None
                    )
    df['pred'] = predict_dataset(ds=ds, model=model)
    return df