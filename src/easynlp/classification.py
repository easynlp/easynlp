from easynlp.data import handle_data
import transformers
from typing import Union
import pandas as pd
import datasets


def classification(
    data: Union[list[dict[str, str]], dict[str, list], pd.DataFrame, datasets.Dataset],
    input_column: str,
    labels: list[str],
    output_column: str = "classification",
    model_name: str = "typeform/distilbert-base-uncased-mnli",
):
    """Does zero-shot classification on data."""

    assert (
        input_column != output_column
    ), f"input and output columns must be different, both are {input_column}"

    assert isinstance(labels, list) and all(
        isinstance(s, str) for s in labels
    ), "labels must be list[str]"

    dataset = handle_data(data)

    columns_to_remove = [f for f in dataset.features if f != input_column]
    dataset = dataset.remove_columns(columns_to_remove)

    model = transformers.AutoModel.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    pipe = transformers.ZeroShotClassificationPipeline(model=model, tokenizer=tokenizer)

    dataset = dataset.map(
        get_classification,
        fn_kwargs={
            "input_column": input_column,
            "pipe": pipe,
            "labels": labels,
            "output_column": output_column,
        },
    )

    return dataset


def get_classification(example, input_column, pipe, labels, output_column):
    output = pipe(example[input_column], labels)
    predicted_class = output["labels"][0]
    return {output_column: predicted_class}
