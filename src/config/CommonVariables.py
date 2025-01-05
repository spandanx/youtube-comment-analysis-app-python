from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config.ExtractProperty import Property

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None

class Settings(BaseSettings):
    REFFERAL_CODE: str
    ACCESS_LEVEL: str
    YOUTUBE_DEVELOPER_KEY: str
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: str
    ENCODING_SALT: str

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
    hashed_password: str | None = None
    disabled: bool | None = None
    referral_code: str | None = None


class UserInDB(User):
    hashed_password: str

def get_settings():
    return Settings()

property_var = Property()