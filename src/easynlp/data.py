import datasets
import pandas as pd
from typing import Dict, List, Union


def handle_data(
    data: Union[List[Dict[str, str]], Dict[str, List], pd.DataFrame, datasets.Dataset]
) -> datasets.Dataset:

    assert (
        (isinstance(data, list) and all(isinstance(d, dict) for d in data))
        or (isinstance(data, dict) and all(isinstance(d, list) for d in data.values()))
        or isinstance(data, pd.DataFrame)
        or isinstance(data, datasets.Dataset)
    ), f"data should be list[dict[str]], dict[str, list], pd.DataFrame, or datasets.Dataset"

    if isinstance(data, list):
        dataset = convert_list_to_dataset(data)
    elif isinstance(data, dict):
        dataset = convert_dict_to_dataset(data)
    elif isinstance(data, pd.DataFrame):
        dataset = convert_df_to_dataset(data)
    else:
        assert isinstance(data, datasets.Dataset)
        dataset = data

    return dataset


def convert_list_to_dataset(data: List[Dict]) -> datasets.Dataset:
    df = pd.DataFrame(data)
    dataset = convert_df_to_dataset(df)
    return dataset


def convert_dict_to_dataset(data: Dict[List]) -> datasets.Dataset:
    df = pd.DataFrame(data)
    dataset = convert_df_to_dataset(df)
    return dataset


def convert_df_to_dataset(data: pd.DataFrame) -> datasets.Dataset:
    dataset = datasets.Dataset.from_pandas(data)
    return dataset
