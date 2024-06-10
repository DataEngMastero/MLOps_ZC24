import requests
from io import BytesIO
from typing import List

import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data(*args, **kwargs):
    """
    Template code for loading data from any source.

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    dfs: List[pd.DataFrame] = []

    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(response.text)

    print("Successful Response : OK")

    df = pd.read_parquet(BytesIO(response.content))
    dfs.append(df)

    print(f"Shape of DFs : {df.shape}")

    return pd.concat(dfs)