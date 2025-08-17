from dockbiotic.data import disk

def get_rdkit_num_tasks(featurization: str='graph',
                      smiles_type: str='standard_smiles',
                      debug: bool=False) -> int:
    ds, _ =  disk.load_rdkit_data_from_disk(
                        featurization=featurization,
                        smiles_type=smiles_type,
                        debug=debug
                    )
    return ds.get_shard(0)[1].shape[1]


def get_excape_num_tasks(featurization: str='graph',
                      smiles_type: str='standard_smiles',
                      debug: bool=False) -> int:
    ds, _ =  disk.load_excape_data_from_disk(
                        featurization=featurization,
                        smiles_type=smiles_type,
                        debug=debug
                    )
    return ds.get_shard(0)[1].shape[1]


def get_dockstring_num_tasks(featurization: str='graph',
                      smiles_type: str='standard_smiles',
                      debug: bool=False) -> int:
    ds, _ =  disk.load_dockstring_data_from_disk(
                        featurization=featurization,
                        smiles_type=smiles_type,
                        debug=debug
                    )
    return ds.get_shard(0)[1].shape[1]


def get_red_num_tasks(featurization: str='graph',
                      smiles_type: str='standard_smiles',
                      debug: bool=False) -> int:
    ds, _ =  disk.load_red_data_from_disk(
                        featurization=featurization,
                        smiles_type=smiles_type,
                        debug=debug
                    )
    return ds.get_shard(0)[1].shape[1]