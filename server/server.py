import easynlp
import fastapi
import pydantic
from typing import List, Union

app = fastapi.FastAPI()


class ClassificationRequest(pydantic.BaseModel):
    text: List[str]
    labels: List[str]


class ClassificationResponse(pydantic.BaseModel):
    predicted_label: List[str]


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
    outputs = easynlp.classification(data, labels)
    predicted_label = [output["classification"] for output in outputs]
    response = ClassificationResponse(predicted_label=predicted_label)
    return response
