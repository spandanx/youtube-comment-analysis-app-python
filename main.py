from typing import List
from typing import Union
from fastapi import FastAPI, Request, HTTPException, status
from pydantic import BaseModel
from YoutubeSearch import YoutubeSearch
# from DataProcessing.TextSummarization import TextSummarizer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

origins = [
    "http://localhost:3000"
]

class Sentence(BaseModel):
    text: List[str]

class VideoIds(BaseModel):
    ids: List[str]

class QuesAnsDType(BaseModel):
    context: List[str]
    questions: List[str]
    qaModel: str

class SummarizationDType(BaseModel):
    texts: List[str]
    summModel: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from SentenceDetectionGeneratorDetector import SentenceTypeDetection

ys = YoutubeSearch()

@app.get("/heathcheck")
async def root():
    return {"message": "I am Alive!"}

@app.post("/check-sentence/")
async def get_sentence_type(sentence: Sentence):
    return SentenceTypeDetection.TestSentenceDetectionModel(sentence.text)

@app.get("/video-search/")
async def get_video(searchText: str, max_results: int | None = 10):
    return ys.youtube_get_videos(searchText, max_results)

@app.post("/extract-text/")
async def extract_comments(videoIds: VideoIds):
    try:
        response = ys.extract_youtube_comments(videoIds.ids, max_results_comments = 2, max_results_replies = 20)
        return response

    except Exception as e:
        print('Something went wrong')
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Error has occurred -  {e}'
        )

@app.post("/summarize-text/")
async def summarize_text(summarizationDType: SummarizationDType):
    try:
        # response = ys.summarize_youtube_comments(videoIds.ids, max_results_comments = 2, max_results_replies = 20)
        response = ys.summarize_comments(summarizationDType.texts, summarizationDType.summModel)
        return response

    except Exception as e:
        print('Something went wrong')
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Error has occurred -  {e}'
        )

@app.post("/answer-question/")
async def answer_questions(quesAnsDType: QuesAnsDType):
    try:
        return ys.answer_questions(quesAnsDType.questions, quesAnsDType.context, quesAnsDType.qaModel)

    except Exception as e:
        print('Something went wrong')
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Error has occurred -  {e}'
        )

@app.exception_handler(Exception)
async def validation_exception_handler(request: Request, exc: Exception):
    # Change here to Logger
    return JSONResponse(
        status_code=500,
        content={
            "message": (
                f"Failed method {request.method} at URL {request.url}."
                f" Exception message is {exc!r}."
            )
        },
    )
