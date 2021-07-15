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
