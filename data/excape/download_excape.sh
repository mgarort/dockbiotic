CURRENT_DIR=$(dirname "$(readlink -f "$0")")

# Download whole dataset
wget https://zenodo.org/record/2543724/files/pubchem.chembl.dataset4publication_inchi_smiles_v2.tsv.xz
# Decompress (required space ~18GB)
unxz pubchem.chembl.dataset4publication_inchi_smiles_v2.tsv.xz
# Get desired columns (SMILES, gene name and activity label)
awk 'BEGIN{FS="\t";OFS="\t"} {print $12, $9, $4}' pubchem.chembl.dataset4publication_inchi_smiles_v2.tsv > $CURRENT_DIR/excape.tsv
# Remove whole dataset
rm $CURRENT_DIR/pubchem.chembl.dataset4publication_inchi_smiles_v2.tsv