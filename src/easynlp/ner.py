from easynlp.data import handle_data
import transformers
from typing import Optional, Union
import pandas as pd
import datasets
import collections


def ner(
    data: Union[list[dict[str, str]], dict[str, list], pd.DataFrame, datasets.Dataset],
    input_column: str = "text",
    output_column: str = "ner_tags",
    model_name: Optional[str] = "dslim/bert-base-NER",
):
    """Does named entity recognition on data."""

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
    predicted_tags = [[o["entity_group"] for o in output] for output in outputs]
    predicted_tag_starts = [[o["start"] for o in output] for output in outputs]
    predicted_tag_ends = [[o["end"] for o in output] for output in outputs]
    return {
        output_column: predicted_tags,
        f"{output_column}_starts": predicted_tag_starts,
        f"{output_column}_ends": predicted_tag_ends,
    }
