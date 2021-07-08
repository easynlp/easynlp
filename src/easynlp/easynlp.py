import datasets
import pandas as pd
from .data import (
    convert_list_to_dataset,
    convert_dict_to_dataset,
    convert_df_to_dataset,
)

AVAILABLE_TASKS = ["classification", "ner", "pos"]

AVAILABLE_LANGUAGES = ["en"]


class EasyNLP:
    def __init__(self, task, data, data_args, model=None, language="en"):

        assert (
            task in AVAILABLE_TASKS
        ), f"{task} is not a valid task, should be one of {AVAILABLE_TASKS}"

        assert (
            language in AVAILABLE_LANGUAGES
        ), f"{language} is not a valid language, should be one of {AVAILABLE_LANGUAGES}"

        assert (
            (isinstance(data, list) and all(isinstance(d, dict) for d in data))
            or (isinstance(data, dict) and all(isinstance(d, list) for d in data))
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

        self.dataset = dataset
