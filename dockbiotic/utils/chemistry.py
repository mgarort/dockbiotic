# general imports
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
import numpy as np
from functools import partial
import logging
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors, rdmolops
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles, MakeScaffoldGeneric
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator
import rdkit.DataStructs as DataStructs


# project imports
from dockstring.utils import (smiles_to_mol, sanitize_mol, check_mol,
                              check_charges, check_obabel_install,
                              protonate_mol, DockstringError)



def get_rdkit_descriptors_for_single_smiles(single_smiles: str) -> np.ndarray:
    """
    Compute RDKit descriptors for a single molecule.
    """
    mol = Chem.MolFromSmiles(single_smiles)
    features = []
    for name, function in Descriptors.descList:
        feature = function(mol)
        features.append(feature)
    return np.asarray(features).reshape(1,-1)

# RDKit descriptors for Y (208 basic descriptors like number of functional groups, logP, total surface polar area, etc)

def get_rdkit_descriptors(smiles,parallelize):
    """
    Compute RDKit descriptors for a N x 1 array of SMILES.
    """
    # Get RDKit descriptors
    df = pd.DataFrame(smiles,columns=['smiles'])
    if parallelize:
        df['descriptors'] = df['smiles'].parallel_apply(get_rdkit_descriptors_for_single_smiles)
    else:
        df['descriptors'] = df['smiles'].map(get_rdkit_descriptors_for_single_smiles)
    descriptors = np.stack(df['descriptors'].values).squeeze()
    df = pd.DataFrame(descriptors,index=df['smiles'])
    # Normalize them
    df = (df - df.mean()) / (df.std() + 0.0001)
    descriptors = df.iloc[:,:].values
    return descriptors

# RDKit fingerprints for Y

def get_rdkit_fp_for_single_smiles(single_smiles,length,keep_rdkit_format):
    """
    Compute RDKit fingerprint for a single molecule. Can choose whether to return
    the fingerprint in the original RDKit format or as a numpy array.
    """
    mol = Chem.MolFromSmiles(single_smiles)
    fp_gen = GetRDKitFPGenerator(maxPath=6, fpSize=length)
    fp = fp_gen.GetFingerprint(mol)
    if keep_rdkit_format:
        return fp
    else:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.reshape(1,-1)

def get_rdkit_fp(smiles, length, parallelize):
    """
    Compute RDKit fingerprints for a N x 1 array of SMILES.
    """
    df = pd.DataFrame(smiles,columns=['smiles'])
    partial_function = partial(get_rdkit_fp_for_single_smiles, length=length,
                               keep_rdkit_format=False)
    if parallelize:
        df['fp'] = df['smiles'].parallel_apply(partial_function)
    else:
        df['fp'] = df['smiles'].map(partial_function)
    fp = np.stack(df['fp'].values).squeeze()
    return fp


def get_morgan_fp_for_single_smiles(single_smiles,length,keep_rdkit_format):
    """
    Compute morgan fingerprint for a single molecule. Can choose whether to
    return the fingerprint in the original RDKit format or as a numpy array.
    """
    mol = Chem.MolFromSmiles(single_smiles)
    fp_gen = GetMorganGenerator(radius=2, fpSize=length)
    fp = fp_gen.GetFingerprint(mol)
    if keep_rdkit_format:
        return fp
    else:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.reshape(1,-1)


def get_morgan_fp(smiles, length, parallelize):
    """
    Compute morgan fingerprints for a N x 1 array of SMILES.
    """
    df = pd.DataFrame(smiles,columns=['smiles'])
    partial_function = partial(get_morgan_fp_for_single_smiles, length=length,
                               keep_rdkit_format=False)
    if parallelize:
        df['fp'] = df['smiles'].parallel_apply(partial_function)
    else:
        df['fp'] = df['smiles'].map(partial_function)
    fp = np.stack(df['fp'].values).squeeze()
    return fp


def enumerate_allowed_fragments(smiles,allowed_fragments):
    """
    Lists all the fragments in a given SMILES string that are not considered to be salts
    (because they belong to a set of non-salt SMILES that has been semi-manually prepared).

    Args:
        smiles(str): query SMILES string.
        allowed_fragments(list): list containing all the allowed fragments.

    Returns:
        list of all allowed fragments in the query SMILES string.
    """
    allowed = []
    fragments = smiles.split('.')
    for fragment in fragments:
        if fragment in allowed_fragments:
            allowed.append(fragment)
    return allowed


class StandardizationError(DockstringError):
    '''Error during standardization'''
    pass


def standardize_smiles(smiles):
    '''
    Same standardization process as in DOCKSTRING, except that isomeric information
    is removed from canonical SMILES.
    '''
    try:
        # Make sure user input is standardized
        try:
            canonical_smiles = Chem.CanonSmiles(smiles, useChiral=False)
        except Exception:
            raise StandardizationError
        # Read and check input
        mol = smiles_to_mol(canonical_smiles)
        mol = sanitize_mol(mol)
        check_mol(mol)
        check_charges(mol)
        # Check that the right Open Babel version is available
        check_obabel_install()
        # Protonate ligand
        protonated_mol = protonate_mol(mol, pH=7.4)
        check_mol(protonated_mol)
        # Convert to SMILES
        return Chem.MolToSmiles(protonated_mol)
    except DockstringError:
        logging.info(smiles, ' standardization failed')
        return None


def canonicalize_tautomer_smiles(smiles, canonicalizer):
    try:
        mol = Chem.MolFromSmiles(smiles)
        tautomer_mol = canonicalizer.canonicalize(mol)
        tautomer_smiles = Chem.MolToSmiles(tautomer_mol)
        return tautomer_smiles
    except Exception:
        logging.info(smiles, ' standardization failed')
        return None    


def minimal_standardize_smiles(smiles):
    try:
        try:
            canonical_smiles = Chem.CanonSmiles(smiles, useChiral=False)
            return canonical_smiles
        except Exception as e:
            raise StandardizationError
    except DockstringError:
        logging.info(smiles, ' standardization failed')
        return None


def smiles_to_inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    inchikey = Chem.inchi.MolToInchiKey(mol)
    return inchikey


def smiles_to_fp(smiles,fp_type,keep_rdkit_format=False):
    """
    Simple function to compute fingerprints from one smiles.
    """
    if fp_type == 'rdkit':
        return get_rdkit_fp_for_single_smiles(single_smiles=smiles, length=2048, keep_rdkit_format=keep_rdkit_format)
    elif fp_type == 'rdkit_long':
        return get_rdkit_fp_for_single_smiles(single_smiles=smiles, length=4096, keep_rdkit_format=keep_rdkit_format)
    elif fp_type == 'morgan':
        return get_morgan_fp_for_single_smiles(single_smiles=smiles, length=2048, keep_rdkit_format=keep_rdkit_format)
    elif fp_type == 'morgan_long':
        return get_morgan_fp_for_single_smiles(single_smiles=smiles, length=4096, keep_rdkit_format=keep_rdkit_format)
    else:
        raise ValueError(f'Argument fp_type is incorrect: value given is {fp_type}')


def get_frags(smiles):
    """
    Inspired by https://www.rdkit.org/docs/Cookbook.html
    """
    mol = Chem.MolFromSmiles(smiles)
    mol_frags = rdmolops.GetMolFrags(mol, asMols = True)
    smiles_frags = [Chem.MolToSmiles(each_frag_mol) for each_frag_mol in mol_frags]
    return smiles_frags


def get_largest_frag(smiles):
    """
    Inspired by https://www.rdkit.org/docs/Cookbook.html
    """
    mol = Chem.MolFromSmiles(smiles)
    mol_frags = rdmolops.GetMolFrags(mol, asMols = True)
    largest_mol_frag = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
    return Chem.MolToSmiles(largest_mol_frag)


def smiles_to_scaffold_smiles(smiles):
    scaffold_smiles = MurckoScaffoldSmilesFromSmiles(smiles)
    try:
        return Chem.CanonSmiles(scaffold_smiles)
    except Exception:
        return scaffold_smiles


def smiles_to_generic_scaffold_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        generic_scaffold_mol = MakeScaffoldGeneric(mol)
        generic_scaffold_smiles = Chem.MolToSmiles(generic_scaffold_mol)
        return generic_scaffold_smiles
    except Exception:
        return None