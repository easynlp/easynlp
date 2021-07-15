from easynlp.data import handle_data
import transformers
from typing import Optional, Union
import pandas as pd
import datasets


def ner(
    data: Union[list[dict[str, str]], dict[str, list], pd.DataFrame, datasets.Dataset],
    input_column: str = "text",
    output_column: str = "ner",
    model_name: Optional[str] = None,
):
    """Does named entity recognition on data."""

    if model_name is None:
        model_name = "dslim/bert-base-NER"

    assert (
        input_column != output_column
    ), f"input and output columns must be different, both are {input_column}"

    dataset = handle_data(data)

    columns_to_remove = [f for f in dataset.features if f != input_column]
    dataset = dataset.remove_columns(columns_to_remove)

    model = transformers.AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    pipe = transformers.TokenClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )

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


def get_ner_tags(examples, pipe, input_column, output_column):
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
