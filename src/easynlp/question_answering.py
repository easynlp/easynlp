from easynlp.data import handle_data
import transformers
from typing import Dict, List, Optional, Union
import pandas as pd
import datasets


def question_answering(
    data: Union[List[Dict[str, str]], Dict[str, List], pd.DataFrame, datasets.Dataset],
    input_column: str = "text",
    context_column: str = "context",
    output_column: str = "answer",
    model_name: Optional[str] = None,
):
    """Performs question answering on given data."""

    # get default model name
    if model_name is None:
        model_name = f"distilbert-base-cased-distilled-squad"

    # check input and output columns are different
    assert (
        input_column != output_column
    ), f"input and output columns must be different, both are {input_column}"

    # check input and context columns are different
    assert (
        input_column != context_column
    ), f"input and context columns must be different, both are {input_column}"

    # check context and output columns are different
    assert (
        context_column != output_column
    ), f"context and output columns must be different, both are {context_column}"

    # convert data to datasets.Dataset
    dataset = handle_data(data)

    # remove all columns that aren't the `input_column`
    columns_to_remove = [
        f for f in dataset.features if f not in {input_column, context_column}
    ]
    dataset = dataset.remove_columns(columns_to_remove)

    # load model and tokenizer
    model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # create pipeline
    pipe = transformers.QuestionAnsweringPipeline(
        model=model,
        tokenizer=tokenizer,
    )

    # perform question answering
    dataset = dataset.map(
        get_answer,
        fn_kwargs={
            "pipe": pipe,
            "input_column": input_column,
            "context_column": context_column,
            "output_column": output_column,
        },
        batched=True,
        batch_size=len(dataset) // 100,
    )

    return dataset


def get_answer(
    examples: List[Dict[str, List[str]]],
    pipe: transformers.Pipeline,
    input_column: str,
    context_column: str,
    output_column: str,
) -> Dict[str, List[str]]:
    """Performs question answering on a batch of examples."""
    outputs = pipe(
        question=examples[input_column],
        context=examples[context_column],
        clean_up_tokenization_spaces=True,
    )
    if isinstance(outputs, dict):  # handle case where input is a single example
        outputs = [outputs]
    predicted_answers = [output["answer"] for output in outputs]
    return {output_column: predicted_answers}
