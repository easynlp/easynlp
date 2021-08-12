import easynlp
import time


def benchmark_summarization(n_examples: int = 1_000):

    data = {"text": ["This is some very long text which must be summarized because it is a very long and contains unnecessary words."] * n_examples}
    t0 = time.monotonic()
    _ = easynlp.summarization(data)
    dt = time.monotonic() - t0
    return dt


print(f"Summarization benchmark: {benchmark_summarization()} seconds")
