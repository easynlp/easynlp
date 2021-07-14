from easynlp.data import handle_data
import transformers
from typing import Dict, List, Optional, Union
import pandas as pd
import datasets


def translation(
    data: Union[List[Dict[str, str]], Dict[str, List], pd.DataFrame, datasets.Dataset],
    output_language: str,
    input_language: str = "en",
    input_column: str = "text",
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
        batched=True,
        batch_size=len(dataset) // 100,
    )

    return dataset


def get_translation(
    examples, pipe, input_column, input_language, output_language, output_column
):
    outputs = pipe(
        examples[input_column],
        src_lang=input_language,
        tgt_lang=output_language,
        clean_up_tokenization_spaces=True,
    )
    predicted_translations = [output["translation_text"] for output in outputs]
    return {output_column: predicted_translations}
