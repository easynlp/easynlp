import datasets
import pandas as pd
from typing import Dict, List, Union


def handle_data(
    data: Union[
        List[Dict[str, str]], Dict[str, List[str]], pd.DataFrame, datasets.Dataset
    ]
) -> datasets.Dataset:
    """Handles converting data into a Dataset."""

    # checks data is suitable type
    assert (
        (isinstance(data, list) and all(isinstance(d, dict) for d in data))
        or (isinstance(data, dict) and all(isinstance(d, list) for d in data.values()))
        or isinstance(data, pd.DataFrame)
        or isinstance(data, datasets.Dataset)
    ), f"data should be List[dict[str]], Dict[str, list], pd.DataFrame, or datasets.Dataset"

    # performs the appropriate data -> Dataset conversion
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


def convert_list_to_dataset(data: List[Dict[str, str]]) -> datasets.Dataset:
    """Convert data given in the form of:
    data = [{"text": "I love playing soccer."}, {"text":"It is really sunny today."}]
    into a Dataset.
    """
    df = pd.DataFrame(data)
    dataset = convert_df_to_dataset(df)
    return dataset


def convert_dict_to_dataset(data: Dict[str, List[str]]) -> datasets.Dataset:
    """Convert data given in the form of:
    data = {"text": ["I love playing soccer.", "It is really sunny today."]}
    into a Dataset.
    """
    df = pd.DataFrame(data)
    dataset = convert_df_to_dataset(df)
    return dataset


def convert_df_to_dataset(data: pd.DataFrame) -> datasets.Dataset:
    """Convert pandas DataFrame into a Dataset."""
    dataset = datasets.Dataset.from_pandas(data)
    return dataset
