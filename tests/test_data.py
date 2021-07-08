from easynlp.data import (
    convert_list_to_dataset,
    convert_dict_to_dataset,
    convert_df_to_dataset,
)
import pandas as pd


def test_convert_list_to_dataset():
    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]
    dataset = convert_list_to_dataset(data)
    assert len(dataset) == 3
    assert dataset["a"] == [1, 3, 5]
    assert dataset["b"] == [2, 4, 6]


def test_convert_dict_to_dataset():
    data = {"a": [1, 3, 5], "b": [2, 4, 6]}
    dataset = convert_dict_to_dataset(data)
    assert len(dataset) == 3
    assert dataset["a"] == [1, 3, 5]
    assert dataset["b"] == [2, 4, 6]


def test_convert_df_to_dataset():
    data = pd.DataFrame({"a": [1, 3, 5], "b": [2, 4, 6]})
    dataset = convert_df_to_dataset(data)
    assert len(dataset) == 3
    assert dataset["a"] == [1, 3, 5]
    assert dataset["b"] == [2, 4, 6]
