from easynlp.data import handle_data
import transformers
from typing import Dict, List, Union
import pandas as pd
import datasets


def classification(
    data: Union[List[Dict[str, str]], Dict[str, List], pd.DataFrame, datasets.Dataset],
    labels: List[str],
    input_column: str = "text",
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

    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    pipe = transformers.ZeroShotClassificationPipeline(model=model, tokenizer=tokenizer)

    dataset = dataset.map(
        get_classification,
        fn_kwargs={
            "pipe": pipe,
            "input_column": input_column,
            "labels": labels,
            "output_column": output_column,
        },
        batched=True,
        batch_size=len(dataset) // 100,
    )

    return dataset


def get_classification(examples, pipe, input_column, labels, output_column):
    outputs = pipe(examples[input_column], labels)
    predicted_classes = [outputs["labels"][0] for output in outputs]
    return {output_column: predicted_classes}
