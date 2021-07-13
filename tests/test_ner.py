import easynlp


def test_ner():
    data = {
        "text": [
            "My name is Ben. I live in Scotland and work for Microsoft.",
            "My name is Ben.",
            "I live in Scotland.",
            "I work for Microsoft.",
        ]
    }
    input_column = "text"
    output_column = "ner_tags"
    output_dataset = easynlp.ner(data, input_column, output_column)
    ner_tags = [["PER", "LOC", "ORG"], ["PER"], ["LOC"], ["ORG"]]
    ner_tags_starts = [[11, 26, 48], [11], [10], [11]]
    ner_tags_ends = [[14, 34, 57], [14], [18], [20]]
    assert len(output_dataset) == 4
    assert output_dataset[output_column] == ner_tags
    assert output_dataset[f"{output_column}_starts"] == ner_tags_starts
    assert output_dataset[f"{output_column}_ends"] == ner_tags_ends
