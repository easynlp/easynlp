import easynlp
import fastapi
import pydantic
from typing import List, Union

app = fastapi.FastAPI()


class ClassificationRequest(pydantic.BaseModel):
    text: List[str]
    labels: List[str]
    model_name: str = "typeform/distilbert-base-uncased-mnli"


class ClassificationResponse(pydantic.BaseModel):
    classification: List[str]


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
