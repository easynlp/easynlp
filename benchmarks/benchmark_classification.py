import easynlp
import time


def benchmark_classification(n_examples: int = 1_000):

    data = {
        "text": ["This is some text about sport which must be classified."] * n_examples
    }
    labels = ["sport", "weather", "business"]
    t0 = time.monotonic()
    _ = easynlp.classification(data, labels)
    dt = time.monotonic() - t0
    return dt


print(f"Classification benchmark: {benchmark_classification()} seconds")
