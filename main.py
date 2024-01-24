from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from training.fastText.predict import fasttext_predict

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/status")
def status():
    return {}


class Data(BaseModel):
    text: str


@app.post("/detect")
async def detect(data: Data):
    prediction, confidence = fasttext_predict(data.text)
    return {"prediction": f"{str(prediction).replace('__label__','')}", "confidence": f"{confidence}"}


