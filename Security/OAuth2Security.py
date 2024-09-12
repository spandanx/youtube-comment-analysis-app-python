from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from pydantic import BaseModel

from pydantic_settings import BaseSettings, SettingsConfigDict

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 5


fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    }
}


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None

class Settings(BaseSettings):
    REFFERAL_CODE: str
    ACCESS_LEVEL: str

    model_config = SettingsConfigDict(env_file=".env")


class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None

class RegisterUser(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    password: str | None = None
    disabled: bool | None = None
    referral_code: str | None = None


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print(payload)
        username: str = payload.get("sub")
        print('username', username)
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
        print(token_data)
    except InvalidTokenError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    print(user)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_all_user():
    return sorted(list(fake_users_db.keys()))

async def add_user(user):
    forbidden_exception = HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid refferal code!",
        headers={"WWW-Authenticate": "Bearer"},
    )
    settings = get_settings()
    temp_user = dict(user)
    if 'referral_code' in temp_user:
        if (temp_user['referral_code'] != settings.REFFERAL_CODE):
            raise forbidden_exception
    else:
        raise forbidden_exception
    if 'password' in temp_user:
        temp_user['hashed_password'] = get_password_hash(temp_user['password'])
        del temp_user['password']
    fake_users_db[temp_user['username']] = temp_user
    print(fake_users_db)
    return fake_users_db[temp_user["username"]]["username"]

def get_settings():
    return Settings()



if __name__ == "__main__":
    # search_text = "kolkata restaurants"
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJqb2huZG9lIiwiZXhwIjoxNzI2MDI1MzQxfQ.YNH8DsMuSw4N8JpS1C1lmIc9Ph6XmRyZi5k0EMvmiaM"
    # response = await get_current_user(token)
    # res = get_current_user_classic(token)
    # print(res)