from datetime import timedelta
from typing import List
from fastapi import FastAPI, Request, HTTPException, status, Depends
from pydantic import BaseModel

from YoutubeSearch import YoutubeSearch
# from DataProcessing.TextSummarization import TextSummarizer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Annotated

from Security.OAuth2Security import User, \
    authenticate_user, ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token, add_user, \
    get_current_active_user, RegisterUser, get_settings, mysqlDB
# from Security.OAuth2Security import
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from src.config.CommonVariables import Token, property_var

# from src.db.MySQLDB import MysqlDB

props = property_var.get_property_data()

origins = [
    props["origin"]["frontend"],
    "*"
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
    expose_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

from DataProcessing.SentenceDetectionGeneratorDetector import SentenceTypeDetection

ys = YoutubeSearch()

@app.get("/healthcheck-draft")
async def root(token: Annotated[str, Depends(oauth2_scheme)]):
    return {"message": "I am Alive!"}

@app.get("/healthcheck")
async def healthcheck(current_user: Annotated[User, Depends(get_current_active_user)]):
    return {"message": "I am Alive!"}

@app.post("/check-sentence/")
async def get_sentence_type(sentence: Sentence):
    return SentenceTypeDetection.TestSentenceDetectionModel(sentence.text)

@app.get("/video-search/")
async def get_video(current_user: Annotated[User, Depends(get_current_active_user)],
                    searchText: str, max_results: int | None = 10):
    try:
        return ys.youtube_get_videos(searchText, max_results)
    except Exception as e:
        print('Something went wrong while searching video')
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Error has occurred while searching video -  {e}'
        )

@app.get("/video-search-by-token/")
async def get_video(current_user: Annotated[User, Depends(get_current_active_user)],
                    searchText: str, pageToken: str, max_results: int | None = 10):
    try:
        return ys.youtube_get_videos_by_token(searchText, pageToken, max_results)
    except Exception as e:
        print('Something went wrong while searching video')
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Error has occurred while searching video -  {e}'
        )

@app.post("/extract-text/")
async def extract_comments(current_user: Annotated[User, Depends(get_current_active_user)],
                           videoIds: VideoIds):
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
async def summarize_text(current_user: Annotated[User, Depends(get_current_active_user)],
                         summarizationDType: SummarizationDType):
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

@app.get("/summarize-models/")
async def get_summarization_model(current_user: Annotated[User, Depends(get_current_active_user)]):
    try:
        # response = ys.summarize_youtube_comments(videoIds.ids, max_results_comments = 2, max_results_replies = 20)
        response = ys.get_summarizer_model_list()
        return response

    except Exception as e:
        print('Something went wrong')
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Error has occurred -  {e}'
        )

@app.get("/question-answering-models/")
async def get_question_answering_model(current_user: Annotated[User, Depends(get_current_active_user)]):
    try:
        # response = ys.summarize_youtube_comments(videoIds.ids, max_results_comments = 2, max_results_replies = 20)
        response = ys.get_question_answering_model_list()
        return response

    except Exception as e:
        print('Something went wrong')
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Error has occurred -  {e}'
        )

@app.post("/answer-question/")
async def answer_questions(current_user: Annotated[User, Depends(get_current_active_user)],
                           quesAnsDType: QuesAnsDType):
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

@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    try:
        print("/token endpoint called")
        user = authenticate_user(form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["username"]}, expires_delta=access_token_expires
        )
        return Token(access_token=access_token, token_type="bearer")
    except Exception as e:
        print('Something went wrong while creating token')
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Error has occurred -  {e}'
        )


@app.get("/users/me/", response_model=User)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    return current_user

# @app.get("/users/all/")
# async def read_users_all():
#     return await get_all_user()

@app.get("/users/me/items/")
async def read_own_items(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    return [{"owner": current_user.username}]

@app.post("/users/register/")
async def register_user(user: RegisterUser):
    return await add_user(user)

@app.get("/settings/")
async def get_settings_property():
    settings = dict(get_settings())
    return settings

@app.on_event("startup")
async def startup_event():
    print("Executing on startup")
    # mysqlDB.start_connection()
    # res = mysqlDB.get_user_by_username("admin2")
    # print(res)
    # Perform any necessary setup here

@app.on_event("shutdown")
async def shutdown_event():
    mysqlDB.close_connection()
    print("Executing on shutdown")
    # Perform any necessary cleanup here