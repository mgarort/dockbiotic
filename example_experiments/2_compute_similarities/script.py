from dockbiotic.utils.similarities import dataframes as sim_dataframes
from dockbiotic.utils.similarities import computation as sim_computation
from dockbiotic import constants
from tqdm import tqdm
tqdm.pandas()
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_dataset', type=str, help='Dataset of origin/rows', required=True)
    parser.add_argument('--target_dataset', type=str, help='Dataset target/columns', required=True)
    return parser.parse_args()

args = parse_args()
origin = args.origin_dataset
target = args.target_dataset


df_origin, df_target = sim_dataframes.get_origin_and_target_dataframes(origin=origin, target=target)


for fp_type in ['morgan', 'rdkit']:

    # compute similarities
    results = sim_computation.compute_closest_and_similarities(df_origin=df_origin,
                                    df_target=df_target, fp_type=fp_type,
                                    smiles_type='standard_smiles')
    closest_compounds, closest_similarities = results

    # save
    if not (constants.DATA_DIR / 'similarities').exists():
        (constants.DATA_DIR / 'similarities').mkdir()

    compounds_path = constants.DATA_DIR / 'similarities' / f'closest_compounds_{fp_type}_{origin}_{target}.tsv'
    closest_compounds.to_csv(compounds_path, sep='\t', header=False, index=True)
    similarities_path = constants.DATA_DIR / 'similarities' / f'closest_similarities_{fp_type}_{origin}_{target}.tsv'
    closest_similarities.to_csv(similarities_path, sep='\t', header=False, index=True)
