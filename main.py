from typing import List
from typing import Union
from fastapi import FastAPI, Request
from pydantic import BaseModel
from YoutubeSearch import YoutubeSearch
from DataProcessing.TextSummarizer import TextSummarizer
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000"
]

class Sentence(BaseModel):
    text: List[str]

class VideoIds(BaseModel):
    ids: List[str]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from ModelGenerator import SentenceTypeDetection

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

@app.post("/summarize-text/")
async def get_sentence_type(videoIds: VideoIds):
    return ys.summarize_youtube_comments(videoIds.ids, max_results_comments = 2, max_results_replies = 20)
