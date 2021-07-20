from easynlp.data import handle_data
import transformers
from typing import Dict, List, Optional, Union
import pandas as pd
import datasets


def ner(
    data: Union[list[dict[str, str]], dict[str, list], pd.DataFrame, datasets.Dataset],
    input_column: str = "text",
    output_column: str = "ner",
    model_name: Optional[str] = None,
) -> datasets.Dataset:
    """Performs NER on given data."""

    # get default model name
    if model_name is None:
        model_name = "dslim/bert-base-NER"

    # check input and output columns are different
    assert (
        input_column != output_column
    ), f"input and output columns must be different, both are {input_column}"

    # convert data to datasets.Dataset
    dataset = handle_data(data)

    # remove all columns that aren't the `input_column`
    columns_to_remove = [f for f in dataset.features if f != input_column]
    dataset = dataset.remove_columns(columns_to_remove)

    # load model and tokenizer
    model = transformers.AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # create pipeline
    pipe = transformers.TokenClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )

    # perform NER
    dataset = dataset.map(
        get_ner_tags,
        fn_kwargs={
            "pipe": pipe,
            "input_column": input_column,
            "output_column": output_column,
        },
        batched=True,
        batch_size=len(dataset) // 100,
    )

    return dataset


def get_ner_tags(
    examples: List[Dict[str, List[str]]],
    pipe: transformers.Pipeline,
    input_column: str,
    output_column: str,
) -> Dict[str, List[str]]:
    """Performs NER on a batch of examples."""
    outputs = pipe(
        examples[input_column],
    )
    if isinstance(outputs[0], dict):  # handle case where input is a single example
        outputs = [outputs]
    predicted_tags = [[o["entity_group"] for o in output] for output in outputs]
    predicted_start_offsets = [[o["start"] for o in output] for output in outputs]
    predicted_end_offsets = [[o["end"] for o in output] for output in outputs]
    return {
        f"{output_column}_tags": predicted_tags,
        f"{output_column}_start_offsets": predicted_start_offsets,
        f"{output_column}_end_offsets": predicted_end_offsets,
    }
