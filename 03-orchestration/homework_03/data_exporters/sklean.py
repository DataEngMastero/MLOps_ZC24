from typing import Callable, Dict, Tuple, Union  

from mlops.utils.models.sklearn import load_class, tune_hyperparameters 
from pandas import Series 
from scipy.sparse._csr import csr_matrix 
from sklearn.base import BaseEstimator

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(
    settings, *args, **kwargs
):
    hyperparameters, X, Y, model_info = settings

    model_class = model_info['cls']
    model = model_class(**hyperparameters)
    model.fit(X, Y)

    print(model.intercept_)

    return model, model_info

