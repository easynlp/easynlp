from easynlp.data import handle_data
import transformers
from typing import Optional, Union
import pandas as pd
import datasets


def translation(
    data: Union[list[dict[str, str]], dict[str, list], pd.DataFrame, datasets.Dataset],
    input_column: str,
    input_language: str,
    output_language: str,
    output_column: str = "translation",
    model_name: Optional[str] = None,
):
    """Does translation on data."""

    if model_name is None:
        model_name = f"Helsinki-NLP/opus-mt-{input_language}-{output_language}"

    assert (
        input_column != output_column
    ), f"input and output columns must be different, both are {input_column}"

    assert isinstance(
        input_language, str
    ), f"input_language must be str, got {type(input_language)}"

    assert isinstance(
        output_language, str
    ), f"output_language must be str, got {type(output_language)}"

    dataset = handle_data(data)

    columns_to_remove = [f for f in dataset.features if f != input_column]
    dataset = dataset.remove_columns(columns_to_remove)

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    pipe = transformers.TranslationPipeline(
        model=model,
        tokenizer=tokenizer,
        task=f"translation_{input_language}_to_{output_language}",
    )

    dataset = dataset.map(
        get_translation,
        fn_kwargs={
            "pipe": pipe,
            "input_column": input_column,
            "input_language": input_language,
            "output_language": output_language,
            "output_column": output_column,
        },
    )

    return dataset


def get_translation(
    example, pipe, input_column, input_language, output_language, output_column
):
    output = pipe(
        example[input_column],
        src_lang=input_language,
        tgt_lang=output_language,
        clean_up_tokenization_spaces=True,
    )
    predicted_class = output[0]["translation_text"]
    return {output_column: predicted_class}
