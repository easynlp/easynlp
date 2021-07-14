import easynlp
import time


def benchmark_translation(n_examples: int = 1_000):

    data = {"text": ["This is some text which must be translated."] * n_examples}
    output_language = "de"
    t0 = time.monotonic()
    _ = easynlp.translation(data, output_language)
    dt = time.monotonic() - t0
    return dt


print(f"Translation benchmark: {benchmark_translation()} seconds")
