import easynlp


def test_single_ner():
    data = {
        "text": [
            "My name is Ben. I live in Scotland and work for Microsoft.",
        ]
    }
    input_column = "text"
    output_column = "ner"
    output_dataset = easynlp.ner(data, input_column, output_column)
    ner_tags = [["PER", "LOC", "ORG"]]
    ner_start_offsets = [[11, 26, 48]]
    ner_end_offsets = [[14, 34, 57]]
    assert len(output_dataset) == 1
    assert output_dataset[f"{output_column}_tags"] == ner_tags
    assert output_dataset[f"{output_column}_start_offsets"] == ner_start_offsets
    assert output_dataset[f"{output_column}_end_offsets"] == ner_end_offsets


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
    output_column = "ner"
    output_dataset = easynlp.ner(data, input_column, output_column)
    ner_tags = [["PER", "LOC", "ORG"], ["PER"], ["LOC"], ["ORG"]]
    ner_start_offsets = [[11, 26, 48], [11], [10], [11]]
    ner_end_offsets = [[14, 34, 57], [14], [18], [20]]
    assert len(output_dataset) == 4
    assert output_dataset[f"{output_column}_tags"] == ner_tags
    assert output_dataset[f"{output_column}_start_offsets"] == ner_start_offsets
    assert output_dataset[f"{output_column}_end_offsets"] == ner_end_offsets
