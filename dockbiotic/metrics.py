import numpy as np


def rank_by_prediction(y, p, smiles=None):
    y = y.squeeze()
    p = p.squeeze()
    ranking = np.argsort(p)[::-1]
    y_ranked = y[ranking]
    p_ranked = p[ranking]
    if smiles is None:
        return y_ranked, p_ranked
    else:
        smiles_ranked = smiles[ranking]
        return y_ranked, p_ranked, smiles_ranked


def num_actives_selected(y, p, num_selected=200, threshold=0.8):
    """
    Metric that counts the number of actives in a top-ranking selected subset.

    Args:
        y (numpy array): non-binary antibiotic activity labels (to be binarized).
        hat_y (numpy array): activity predictions.
        num_selected (int, optional): size of top-ranking selected subset.
        threshold (float, optional): threshold to binarize labels y.

    Returns:
        Integer that represents the number of actives in the selected subset.
    """
    y_ranked, p_ranked = rank_by_prediction(y, p)
    y_ranked_binary = (y_ranked >= threshold).astype(int)
    num_actives = sum(y_ranked_binary[:num_selected])
    return num_actives


def enrichment_factor(y, p, num_selected=200, threshold=0.8):
    """
    Metric that computes the enrichment factor.

    Args:
        y (numpy array): non-binary antibiotic activity labels (to be binarized).
        hat_y (numpy array): activity predictions.
        num_selected (int, optional): size of top-ranking selected subset.
        threshold (float, optional): threshold to binarize labels y.

    Returns:
        Integer that represents the number of actives in the selected subset.
    """
    # Get number of actives selected
    n_actives_selected = num_actives_selected(y, p, num_selected=num_selected, threshold=threshold)
    # Compute EF
    rate_before = (y >= threshold).astype(int).sum() / len(y)
    rate_after = n_actives_selected / num_selected
    return rate_after / rate_before