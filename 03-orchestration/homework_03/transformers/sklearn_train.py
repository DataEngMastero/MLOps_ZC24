from sklearn.linear_model import LinearRegression
from typing import Tuple 
from pandas import DataFrame, Series

from mlops.utils.models.sklearn import load_class, tune_hyperparameters 


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(
    data: Tuple[DataFrame, DataFrame, DataFrame], *args, **kwargs 
):
    X, X_train, X_val, Y, Y_train, Y_val, dv = data  
    model_name : str = 'linear_model.LinearRegression'
    child_metadata: Dict = dict(block_uuid=model_name.split('.')[-1]) 
    model_class = load_class(model_name)
    print(model_class)

    hyperparameters = tune_hyperparameters(
        model_class,
        X_train=X_train,
        y_train=Y_train,
        X_val=X_val,
        y_val=Y_val
        # max_evaluations=kwargs.get('max_evaluations', 50),
        # random_state=kwargs.get('random_state', 7)
    )

    model_info = dict(cls=model_class, name=model_name)

    return hyperparameters, X, Y, model_info



@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'