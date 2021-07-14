import easynlp
import time


def benchmark_ner(n_examples: int = 1_000):

    data = {"text": ["This is some text which contains named entities."] * n_examples}
    t0 = time.monotonic()
    _ = easynlp.ner(data)
    dt = time.monotonic() - t0
    return dt


print(f"NER benchmark: {benchmark_ner()} seconds")
