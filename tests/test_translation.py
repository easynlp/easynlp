import easynlp


def test_translation():
    data = {
        "text": [
            "I love playing soccer.",
            "It is really sunny today.",
            "The stock market is down 10% today.",
        ]
    }
    input_column = "text"
    input_language = "en"
    output_language = "de"
    output_column = "translation"
    output_dataset = easynlp.translation(
        data, input_column, input_language, output_language, output_column
    )
    assert len(output_dataset) == 3
    assert output_dataset["translation"] == [
        "Ich liebe Fußball zu spielen.",
        "Heute ist es wirklich sonnig.",
        "Der Aktienmarkt ist heute um 10 % gefallen.",
    ]
