# easynlp

## Introduction

`easynlp` is a library for performing natural language processing (NLP) with minimal friction -- import the library and call the task-specific function and you're done! `easynlp` currently supports the following tasks:

- text classification
- translation
- named entity recogition
- summarization
- question answering

`easynlp` is essentially is a wrapper around the [transformers](https://github.com/huggingface/transformers) library making heavy use of their [pipelines](https://huggingface.co/transformers/main_classes/pipelines.html).

`easynlp` only supports inference, and does not support training or fine-tuning a model. However, it can use any task-appropriate model provided by the `transformers` library, see the list of all models [here](https://huggingface.co/models).

**Note**: the first time each `easynlp` function is called it will need to download the appropriate pre-trained model. These can be anywhere from ~250MB to ~2GB, depending on the model used.

## Installation

```bash
pip install git+https://github.com/easynlp/easynlp
```

Alternatively, if you wish to edit `easynlp`:

```bash
git clone https://github.com/easynlp/easynlp.git
cd easynlp
pip install -e .
```

## Usage

The data passed to an `easynlp` function can either be: a list of dictionaries, a dictionary of lists, a `pandas.DataFrame` or a `datasets.Dataset`.

```python
list_data = [{"text": "I love playing soccer."}, {"text":"It is really sunny today."}]
dict_data = {"text": ["I love playing soccer.", "It is really sunny today."]}
```

Note that even with a single example the data must be in the same format:

```python
# INCORRECT
incorrect_data = {"text": "I love playing soccer."}

# CORRECT
list_data = [{"text": "I love playing soccer."}]
dict_data = {"text": ["I love playing soccer."]}
```

The `key` of the data you want to process is called an `input_column` in `easynlp`. All `easynlp` functions use a default `input_column` of `"text"`. This can be changed by passing an `input_column` parameter to the corresponding `easynlp` function, i.e. if the above examples used `"words"` instead of `"text"`, then the `input_column` should be `"words"`.

All `easynlp` functions return a `datasets.Dataset` with an `input_column` and `output_column` key. No other keys from the input data are kept. The default `output_column` depends on the `easynlp` function being called, but can be specified by passing an `output_column` parameter to the corresponding `easynlp` function.

### Sequence Classification

`easynlp` performs zero-shot classification. We have to give it the data we want to classify and list of labels, in natural language, i.e. a list of strings. The default `output_column` is `"classification"`.

```python
import easynlp

data = {
        "text": [
            "The stock market is down 10% today.",
            "It is really sunny today.",
            "I love playing soccer.",
            ]
        }
labels = ["sport", "weather", "business"]

output_dataset = easynlp.classification(data, labels)

actual_labels = ["business", "weather", "sports"]

assert output_dataset["classification"] == actual_labels
```

### Translation

`easynlp.translation` uses a default `input_language` of `"en"` (English), we can change this by passing an `input_language` argument to `easynlp.translation`. Languages are specified by their two-letter codes. The `output_column` defaults to `"translation"`.

```python
import easynlp

data = {
    "text": [
        "I love playing soccer.",
        "It is really sunny today.",
        "The stock market is down 10% today.",
            ]
        }
output_language = "de"

output_dataset = easynlp.translation(data, output_language)

translated_text = [
    "Ich spiele gern Fu??ball.",
    "Heute ist es wirklich sonnig.",
    "Die B??rse ist heute um 10% gesunken.",
    ]

assert output_dataset["translation"] == translated_text
```

### Named Entity Recognition (NER)

`easynlp.ner` recognizes the following NER tags: `PER` (person), `ORG` (organization), `LOC` (location), and `MIS` (miscellaneous). For each example in the input, `easynlp.ner` returns: a list of tags, a list of start offsets to the beginning of those tagged entities, and a list of end offsets to the end of those tagged entities. The `output_column` defaults to `"ner"` and the output keys are given by `f"{output_column}_tags"`, `f"{output_column}_start_offsets"`, and `f"{output_column}_end_offsets"`, for the tags, start offsets and end offsets, respectively.

```python
import easynlp

data = {
    "text": [
        "My name is Ben. I live in Scotland and work for Microsoft.",
        "My name is Ben.",
        "I live in Scotland.",
        "I work for Microsoft.",
            ]
        }

output_dataset = easynlp.ner(data)

ner_tags = [["PER", "LOC", "ORG"], ["PER"], ["LOC"], ["ORG"]]
ner_tags_starts = [[11, 26, 48], [11], [10], [11]]
ner_tags_ends = [[14, 34, 57], [14], [18], [20]]

assert output_dataset["ner_tags"] == ner_tags
assert output_dataset["ner_start_offsets"] == ner_tags_starts
assert output_dataset["ner_end_offsets"] == ner_tags_ends
```

### Summarization

`easynlp.summarization` performs automatic summarization on given text. Summarization models are usually relatively large, and quite slow. The default `output_column` is `"summarization"`.

```python
import easynlp

data = {
        "text": [
            """The warning begins at 22:00 GMT on Saturday and
               ends at 10:00 on Sunday. The ice could lead to
               difficult driving conditions on untreated roads
               and slippery conditions on pavements, the weather
               service warned. Only the southernmost counties and
               parts of the most westerly counties are expected
               to escape. Counties expected to be affected are
               Carmarthenshire, Powys, Ceredigion, Pembrokeshire,
               Denbighshire, Gwynedd, Wrexham, Conwy, Flintshire,
               Anglesey, Monmouthshire, Blaenau Gwent,
               Caerphilly, Merthyr Tydfil, Neath Port Talbot,
               Rhondda Cynon Taff and Torfaen.""",
                ]
        }

output_dataset = easynlp.summarization(data)

summarized_text = ['The Met Office has issued a yellow "be aware" warning for ice across much of Wales.']

assert output_dataset[output_column] == summarized_text
```

### Question Answering

`easynlp.question_answering` performs extractive question answering given a question and a context. It can only answer questions where the answer appears within the given context. The default input and context columns are `"text"` and `"context"`, respectively. The default output column is `"answer"`.

```python
import easynlp

data = {
        "text": [
            "What is extractive question answering?",
                ],
        "context": [
            """Extractive Question Answering is the task of extracting an answer from a text given a question.
               An example of a question answering dataset is the SQuAD dataset, which is entirely based on that task.
               If you would like to fine-tune a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.""",
                   ],
       }

output_dataset = easynlp.question_answering(data)

answers = ["the task of extracting an answer from a text given a question"]

assert output_dataset["answers"] == answers
```

## Server

We can also use `easynlp` as a server using the `easynlp` command line interface.

```bash
easynlp --port 1234
```

The `port` is optional and defaults to `8888`. This starts an `uvicorn` server at `http://127.0.0.1:<port>` (which is the same thing as `http://localhost:<port>`). We can now send a `POST` request to `http://127.0.0.1:<port>/<task>` (or `http://localhost:<port>/<task>`) to run the desired `easynlp` function on the given data.

The data must be a `JSON` object which is formatted in the same way as the dictionary of lists format used for `easynlp` as a Python module.

The main thing to note is that the `input_column` and `output_column` names cannot be specified when using the server. We must use the default values from the respective `easynlp` function. The `input_column` defaults to `"text"` and the `output_column` depends on the task being performed.

An example of how to use the server with the `requests` module in Python:

```python
import requests

r = requests.post("http://localhost:1234/classification", json={"text": ["I love playing soccer."],
                                                                "labels": ["sports", 
                                                                           "weather",
                                                                           "business"]
                                                               })
assert r.status_code == 200
assert r.json() == {"classification": ["sports"]}

r = requests.post("http://localhost:1234/translation", json={"text": ["I love playing soccer."],
                                                             "output_language": "de"
                                                            })
assert r.status_code == 200
assert r.json() == {"translation": ["Ich spiele gern Fu??ball."]}

r = requests.post("http://localhost:1234/ner", json={"text": ["My name is Ben.",
                                                              "I live in Scotland.",
                                                              "I work for Microsoft."]
                                                    })

assert r.status_code == 200
assert r.json() == {"ner_tags": [["PER"], ["LOC"], ["ORG"]],
                    "ner_tags_starts": [[11], [10], [11]],
                    "ner_tags_ends": [[14], [18], [20]]}

r = requests.post("http://localhost:1234/summarization", json={"text": ["""Arsenal beat Chelsea 2-0 in
                                                                           the FA Cup Final last night."""]
                                                              })

assert r.status_code == 200
assert r.json() == {'summarization': ["Watch highlights of Arsenal's victory over Chelsea in the FA Cup Final."]}


r = requests.post("http://localhost:1234/question_answering", json={"text": ["What is John's favourite number?"],
                                                                    "context": ["""John's favourite color is red,
                                                                                   and his favourite number is seven."""]
                                                                   })

assert r.status_code == 200
assert r.json() == {"answer": ["seven"]}
```

An example of how to use the server with `curl` using the `easynlp` command line interface on port `8888`:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8888/classification' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": [
    "This is some text about soccer."
  ],
  "labels": [
    "sports", "weather", "business"
  ]
}'
```

```bash
curl -X 'POST' \
  'http://127.0.0.1:8888/translation' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": [
    "I love sitting outside."
  ],
  "output_language": "fr"
}'
```

```bash
curl -X 'POST' \
  'http://127.0.0.1:8888/ner' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": ["My name is Ben.", "I live in Scotland.", "I work for Microsoft."]}'
```

```bash
curl -X 'POST' \
  'http://localhost:8888/summarization' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": [
    "Arsenal beat Chelsea 2-0 in the FA Cup Final last night."
  ]
}'
```

```bash
curl -X 'POST' \
  'http://localhost:8888/question_answering' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": [
    "What is John'\''s favourite number?"
  ],
  "context": [
    "John'\''s favourite color is red, and his favourite number is seven."
  ]
}'
```
