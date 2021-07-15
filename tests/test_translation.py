import easynlp


def test_single_translation():
    data = {
        "text": [
            "I love playing soccer.",
        ]
    }
    output_language = "de"
    input_language = "en"
    input_column = "text"
    output_column = "translation"
    output_dataset = easynlp.translation(
        data, output_language, input_language, input_column, output_column
    )
    translated_text = [
        "Ich spiele gern Fußball.",
    ]
    assert len(output_dataset) == 1
    assert output_dataset[output_column] == translated_text


def test_translation():
    data = {
        "text": [
            "I love playing soccer.",
            "It is really sunny today.",
            "The stock market is down 10% today.",
        ]
    }
    output_language = "de"
    input_language = "en"
    input_column = "text"
    output_column = "translation"
    output_dataset = easynlp.translation(
        data, output_language, input_language, input_column, output_column
    )
    translated_text = [
        "Ich spiele gern Fußball.",
        "Heute ist es wirklich sonnig.",
        "Die Börse ist heute um 10% gesunken.",
    ]
    assert len(output_dataset) == 3
    assert output_dataset[output_column] == translated_text
