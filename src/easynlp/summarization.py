from easynlp.data import handle_data
import transformers
from typing import Dict, List, Optional, Union
import pandas as pd
import datasets


def summarization(
    data: Union[List[Dict[str, str]], Dict[str, List], pd.DataFrame, datasets.Dataset],
    input_column: str = "text",
    output_column: str = "summarization",
    model_name: Optional[str] = None,
):
    """Does summarization on data."""

    if model_name is None:
        model_name = f"google/pegasus-xsum"

    assert (
        input_column != output_column
    ), f"input and output columns must be different, both are {input_column}"

    dataset = handle_data(data)

    columns_to_remove = [f for f in dataset.features if f != input_column]
    dataset = dataset.remove_columns(columns_to_remove)

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    pipe = transformers.SummarizationPipeline(
        model=model,
        tokenizer=tokenizer,
    )

    dataset = dataset.map(
        get_summarization,
        fn_kwargs={
            "pipe": pipe,
            "input_column": input_column,
            "output_column": output_column,
        },
        batched=True,
        batch_size=len(dataset) // 100,
    )

    return dataset


def get_summarization(examples, pipe, input_column, output_column):
    outputs = pipe(
        examples[input_column],
        clean_up_tokenization_spaces=True,
    )
    predicted_summarizations = [output["summary_text"] for output in outputs]
    return {output_column: predicted_summarizations}
