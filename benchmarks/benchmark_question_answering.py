import easynlp
import time


def benchmark_question_answering(n_examples: int = 1_000):

    data = {"text": ["This is some text which contains a question"] * n_examples,
            "context": ["This is some text which contains the context which is used to answer the question."] * n_examples}
    t0 = time.monotonic()
    _ = easynlp.question_answering(data)
    dt = time.monotonic() - t0
    return dt


print(f"Question answering benchmark: {benchmark_question_answering()} seconds")
