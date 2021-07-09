from easynlp.data import (
    handle_data,
    convert_list_to_dataset,
    convert_dict_to_dataset,
    convert_df_to_dataset,
)
import datasets
import pandas as pd
import pytest

list_data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]
dict_data = {"a": [1, 3, 5], "b": [2, 4, 6]}
df_data = pd.DataFrame(dict_data)
ds_data = datasets.Dataset.from_pandas(df_data)


@pytest.mark.parametrize("data", [list_data, dict_data, df_data, ds_data])
def test_handle_data(data):
    dataset = handle_data(data)
    assert len(dataset) == 3
    assert dataset["a"] == [1, 3, 5]
    assert dataset["b"] == [2, 4, 6]


def test_convert_list_to_dataset():
    dataset = convert_list_to_dataset(list_data)
    assert len(dataset) == 3
    assert dataset["a"] == [1, 3, 5]
    assert dataset["b"] == [2, 4, 6]


def test_convert_dict_to_dataset():
    dataset = convert_dict_to_dataset(dict_data)
    assert len(dataset) == 3
    assert dataset["a"] == [1, 3, 5]
    assert dataset["b"] == [2, 4, 6]


def test_convert_df_to_dataset():
    dataset = convert_df_to_dataset(df_data)
    assert len(dataset) == 3
    assert dataset["a"] == [1, 3, 5]
    assert dataset["b"] == [2, 4, 6]
