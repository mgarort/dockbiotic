# Experiments on transfer learning for virtual screening of antibacterials

Code accompanying the paper "Transfer learning enables discovery of sub-micromolar antibacterials for ESKAPE pathogens from ultra-large chemical spaces".


## Installation


1. Clone the repo: `git clone git@github.com:mgarort/dockbiotic.git`

2. Enter the repo: `cd dockbiotic`

3. Install:

YAML installation for Ubuntu 18.0:

```
conda env create -f environment.yml
conda activate dockbiotic
pip cache purge
pip install deepchem[torch]
pip install -e .
```

Manual installation for other OS:

```
conda create --name dockbiotic python=3.9
conda activate dockbiotic
conda install conda-forge::dockstring
conda install pytorch::pytorch
conda install pyg::pyg
conda install conda-forge::tensorflow
conda install conda-forge::pandarallel
conda install anaconda::ipywidgets
conda install conda-forge::py-xgboost
conda install dglteam/label/th21_cpu::dgl
conda install conda-forge::openpyxl
pip cache purge
pip install --pre deepchem[torch]
conda install -c dglteam/label/th21_cpu dgl
pip install -e .
```


## Example experiments

The `example_experiments` folder contain example scripts showing how to pretrain, fine-tune, evaluate and make predictions with models.

Enter the example experiments folder: `cd example_experiments`

### 1. Preparing datasets

`bash example_experiments/1_prepare_datasets/script.sh`

(This script assumes that you have `wget` installed.)

The first step is to process datasets in `dockbiotic/data` from their raw form to a clean `processed_{dataset}.tsv` file.

If you inspect `script.sh`, you'll notice that some datasets are prepared with a `--debug` flag. This flag restricts the number of samples processed to a maximum of 5000. If you use it, each script in `example_experiments` should take between 10 and 30 minutes to complete.

Alternatively, if you'd like to prepare and use datasets in full, you'll need to:

1. Delete the DeepChem sharded datasets in `$DEEPCHEM_DATA_DIR` (see below).
2. Recreate the `processed_{dataset}.tsv` files without the `--debug` flag.
3. Recreate the DeepChem sharded datasets in `$DEEPCHEM_DATA_DIR`.


### 2. Computing similarities and getting closest compounds

`bash example_experiments/2_compute_similarities/script.sh`

For evaluation purposes, we will use [Stokes](https://www.cell.com/cell/fulltext/S0092-8674(20)30102-1) as training set and [COADD](https://db.co-add.org/) as held-out test set. We will try to avoid data leakage by discarding from COADD every molecule with a Tanimoto similarity greater than 0.9 to any Stokes molecule.

Running this script will create a new folder `dockbiotic/data/similarities` where the Tanimoto values and indices of closest molecules between Stokes and COADD are saved. We calculate Tanimoto similarities on both Morgan fingerprints and RDKit path fingerprints and take the highest value of the two.


### 3. Saving DeepChem datasets to disk

`bash example_experiments/3_save_to_disk/script.sh`

DeepChem provides convenient sharding functionality to handle datasets that are too large to fit in memory. Shards store already-featurized molecules in disk, so loading them becomes much faster.

In this repo, we take advantage of DeepChem sharding as follows: the first time a dataset is used with certain parameters, it is saved to disk, and subsequent times that the dataset is used with the same parameters, it is loaded from that location. The location where shards are saved is determined by the global variable `DEEPCHEM_DATA_DIR`, which can be set in the command line with 

`export DEEPCHEM_DATA_DIR=/path/to/directory`

(This is already taken care of in `script.sh`, so you don't need to set it manually during the example experiments.)

The name of each dataset and its parameters are hashed, and the shards of that dataset are saved to `$DEEPCHEM_DATA_DIR/hash/`. You can inspect the name of the dataset and its parameters at

`$DEEPCHEM_DATA_DIR/hash/parameter_string.txt`


### 4. Pretraining model on DOCKSTRING

`bash example_experiments/4_pretrain_on_dockstring/script.sh`

We pretrain an AttentiveFP model on (a 5000-subset of) [the DOCKSTRING dataset](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01334), for 20 epochs and with a learning rate of 1e-3.

The pretrained model will be saved at `dockbiotic/saved_models/pretrained_dockstring`.

### 5. Finetuning model on Stokes

`bash example_experiments/5_finetune_on_stokes/script.sh`

We fine-tune the pretrained model on [the Stokes dataset](https://www.cell.com/cell/fulltext/S0092-8674(20)30102-1), for 50 epochs and with a smaller learning rate of 1e-4. We keep all parameters trainable (without freezing any weights), so all of them can be adapted. Intuitively, we can imagine the pretraining stage moves the weights to a region of parameter space that minimizes the loss of the pretraining molecular tasks, and the fine-tuning stage takes small steps in the vicinity of that region.

The fine-tuned model will be saved at `dockbiotic/saved_models/finetuned_stokes`.


### 6. Evaluating model on COADD

`bash example_experiments/6_evaluate_on_coadd/script.sh`

We evaluate the pretrained and fine-tuned AttentiveFP model by calculating the enrichment factor (EF) that it achieves on the (5000-subset of) [the COADD dataset](https://db.co-add.org/) when we select the top 200 molecules according to the model. The EF will be saved at `dockbiotic/example_experiments/6_evaluate_on_coadd/enrichment_factor.txt`.


### 7. Making predictions on ChemDiv sample

`bash example_experiments/7_predict_chemdiv_sample/script.sh`

Finally, we make predictions on a sample of [the ChemDiv library](https://www.chemdiv.com). Predictions will be saved at `dockbiotic/example_experiments/7_predict_chemdiv_sample/chemdiv_sample_preds.tsv`. The higher the value, the stronger the predicted activity. (Note that the model has been trained on y labels that range from 0 to 1, but the model's outputs are not clipped, so they may be smaller than 0 or larger than 1.)

