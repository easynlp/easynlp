import easynlp
import fastapi
import pydantic
from typing import List, Optional

app = fastapi.FastAPI()


class ClassificationRequest(pydantic.BaseModel):
    text: List[str]
    labels: List[str]
    model_name: Optional[str] = None


class ClassificationResponse(pydantic.BaseModel):
    classification: List[str]


class TranslationRequest(pydantic.BaseModel):
    text: List[str]
    output_language: str
    input_language: str = "en"
    model_name: Optional[str] = None


class TranslationResponse(pydantic.BaseModel):
    translation: List[str]


class NERRequest(pydantic.BaseModel):
    text: List[str]
    model_name: Optional[str] = None


class NERResponse(pydantic.BaseModel):
    ner_tags: List[List[str]]
    ner_tags_starts: List[List[int]]
    ner_tags_ends: List[List[int]]


@app.get("/")
def root():
    return {"message": "hello world!"}


@app.get("/ping")
def ping():
    return {"message": "pong"}


@app.post("/classification", response_model=ClassificationResponse)
def classification(request: ClassificationRequest):
    data = {"text": request.text}
    labels = request.labels
    model_name = request.model_name
    outputs = easynlp.classification(data=data, labels=labels, model_name=model_name)
    predicted_labels = [output["classification"] for output in outputs]
    response = ClassificationResponse(classification=predicted_labels)
    return response


@app.post("/translation", response_model=TranslationResponse)
def translation(request: TranslationRequest):
    data = {"text": request.text}
    output_language = request.output_language
    input_language = request.input_language
    model_name = request.model_name
    outputs = easynlp.translation(
        data=data,
        output_language=output_language,
        input_language=input_language,
        model_name=model_name,
    )
    predicted_translations = [output["translation"] for output in outputs]
    response = TranslationResponse(translation=predicted_translations)
    return response


@app.post("/ner", response_model=NERResponse)
def ner(request: NERRequest):
    data = {"text": request.text}
    model_name = request.model_name
    outputs = easynlp.ner(data=data, model_name=model_name)
    predicted_ner_tags = [output["ner_tags"] for output in outputs]
    predicted_ner_tags_starts = [output["ner_tags_starts"] for output in outputs]
    predicted_ner_tags_ends = [output["ner_tags_ends"] for output in outputs]
    response = NERResponse(
        ner_tags=predicted_ner_tags,
        ner_tags_starts=predicted_ner_tags_starts,
        ner_tags_ends=predicted_ner_tags_ends,
    )
    return response
