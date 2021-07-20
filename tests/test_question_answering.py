import easynlp


def test_single_question_answering():
    data = {
        "text": [
            "What is extractive question answering?",
        ],
        "context": [
            """Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
           question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
           a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.""",
        ],
    }
    input_column = "text"
    context_column = "context"
    output_column = "answer"
    output_dataset = easynlp.question_answering(
        data, input_column, context_column, output_column
    )
    answers = [
        "the task of extracting an answer from a text given a question",
    ]
    assert len(output_dataset) == 1
    assert output_dataset[output_column] == answers


def test_question_answering():
    data = {
        "text": [
            "What is extractive question answering?",
            "What is a good example of a question answering dataset?",
        ],
        "context": [
            """Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
           question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
           a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.""",
            """Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
           question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
           a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.""",
        ],
    }
    input_column = "text"
    context_column = "context"
    output_column = "answer"
    output_dataset = easynlp.question_answering(
        data, input_column, context_column, output_column
    )
    answers = [
        "the task of extracting an answer from a text given a question",
        "SQuAD dataset",
    ]
    assert len(output_dataset) == 2
    assert output_dataset[output_column] == answers
