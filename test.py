import easynlp

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

data = {
    "text": ["What is extractive question answering?"],
    "context": [
        """Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
           question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
           a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.""",
    ],
}
output_dataset = easynlp.question_answering(
    data,
)

print(output_dataset[:])
