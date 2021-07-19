from easynlp.data import handle_data
import transformers
from typing import Dict, List, Optional, Union
import pandas as pd
import datasets


def classification(
    data: Union[List[Dict[str, str]], Dict[str, List], pd.DataFrame, datasets.Dataset],
    labels: List[str],
    input_column: str = "text",
    output_column: str = "classification",
    model_name: Optional[str] = None,
) -> datasets.Dataset:
    """Performs zero-shot classification on given data."""

    # get default model name
    if model_name is None:
        model_name = "typeform/distilbert-base-uncased-mnli"

    # check input and output columns are different
    assert (
        input_column != output_column
    ), f"input and output columns must be different, both are {input_column}"

    # ensure labels are list of strings
    assert isinstance(labels, list) and all(
        isinstance(s, str) for s in labels
    ), "labels must be list[str]"

    # convert data to datasets.Dataset
    dataset = handle_data(data)

    # remove all columns that aren't the `input_column`
    columns_to_remove = [f for f in dataset.features if f != input_column]
    dataset = dataset.remove_columns(columns_to_remove)

    # load model and tokenizer
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # create pipeline
    pipe = transformers.ZeroShotClassificationPipeline(model=model, tokenizer=tokenizer)

    # perform classification
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


def get_classification(
    examples: List[Dict[str, List[str]]],
    pipe: transformers.Pipeline,
    input_column: str,
    labels: List[str],
    output_column: str,
) -> Dict[str, List[str]]:
    """Performs classification on a batch of examples."""
    outputs = pipe(examples[input_column], labels)
    if isinstance(outputs, dict):  # handle case where input is a single example
        outputs = [outputs]
    predicted_labels = [output["labels"][0] for output in outputs]
    return {output_column: predicted_labels}
