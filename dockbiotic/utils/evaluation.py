from typing import Union
from dockbiotic import metrics
from dockbiotic.utils import modeling
from dockbiotic.data.disk import load_coadd_data_from_disk
from deepchem.models.gbdt_models import GBDTModel
from dockbiotic.models import AttentiveFPModel


def evaluate_on_coadd(model: Union[GBDTModel, AttentiveFPModel]) -> float:
    """Evaluate enrichment factor (EF) of a model on the COADD dataset."""
    ds, _ = load_coadd_data_from_disk(strain='atcc25922',
                                      discard_above=0.9,
                                      binarize=True)
    y = ds.y
    p = modeling.predict_dataset(ds, model=model)
    return metrics.enrichment_factor(y,p)