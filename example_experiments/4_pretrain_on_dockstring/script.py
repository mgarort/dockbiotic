from dockbiotic.utils import modeling
from dockbiotic.utils import save_load
from dockbiotic.utils import scripting
import logging
import argparse

# Log deepchem info messages during x_featurization and training
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Name of pretraining dataset', required=True)
    parser.add_argument('--num_epochs', type=str, help='Number of pretraining epochs', required=True)
    parser.add_argument('--random_seed', type=int, help='Random seed', required=True)
    parser.add_argument('--model_dir', type=str, help='Directory to save the pre-trained model', required=True)
    return parser.parse_args()

args = parse_args()
random_seed = args.random_seed
dataset_name = args.dataset
num_epochs = int(args.num_epochs)
model_dir = args.model_dir

scripting.set_random_seed(random_seed)

logging.info(f'Pretraining params: dataset {dataset_name}, num_epochs {num_epochs}')
model = modeling.train_model(model_name='attentive_fp',
                             dataset_name=dataset_name,
                             num_epochs=num_epochs,
                             lr=1e-3
                    )
                     
logging.info('Save model')
save_load.save_model(model=model, model_dir=model_dir)
