import datasets
import pandas as pd


def convert_list_to_dataset(data: list[dict]) -> datasets.Dataset:
    df = pd.DataFrame(data)
    dataset = convert_df_to_dataset(df)
    return dataset


def convert_dict_to_dataset(data: dict[list]) -> datasets.Dataset:
    df = pd.DataFrame(data)
    dataset = convert_df_to_dataset(df)
    return dataset


def convert_df_to_dataset(data: pd.DataFrame) -> datasets.Dataset:
    dataset = datasets.Dataset.from_pandas(data)
    return dataset
