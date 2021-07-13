import easynlp


def test_classification():
    data = {
        "text": [
            "I love playing soccer.",
            "It is really sunny today.",
            "The stock market is down 10% today.",
        ]
    }
    input_column = "text"
    labels = ["sport", "weather", "business"]
    output_column = "classification"
    output_dataset = easynlp.classification(data, input_column, labels, output_column)
    assert len(output_dataset) == 3
    assert output_dataset[output_column] == labels
