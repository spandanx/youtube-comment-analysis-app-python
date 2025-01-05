from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import inspect
import os

from src.config.ExtractProperty import Property


def get_caller_file_name():
    call_stack = inspect.stack()
    call_filenames = [stack.filename for stack in call_stack]
    # call_filenames = ['C:\\Users\\spand\\Downloads\\PROJECTS\\Youtube Comment Analysis\\youtube-comment-analysis-app-python\\src\\config\\ExtractProperty.py', '<frozen importlib._bootstrap>', '<frozen importlib._bootstrap_external>', '<frozen importlib._bootstrap>', '<frozen importlib._bootstrap>', '<frozen importlib._bootstrap>', 'C:\\Users\\spand\\Downloads\\PROJECTS\\Youtube Comment Analysis\\youtube-comment-analysis-app-python\\src\\db\\MySQLDB.py', '<frozen importlib._bootstrap>', '<frozen importlib._bootstrap_external>', '<frozen importlib._bootstrap>', '<frozen importlib._bootstrap>', '<frozen importlib._bootstrap>', 'C:\\Users\\spand\\Downloads\\PROJECTS\\Youtube Comment Analysis\\youtube-comment-analysis-app-python\\Security\\OAuth2Security.py', 'C:\\Program Files\\JetBrains\\PyCharm Community Edition 2024.1.1\\plugins\\python-ce\\helpers\\pydev\\_pydev_imps\\_pydev_execfile.py', 'C:\\Program Files\\JetBrains\\PyCharm Community Edition 2024.1.1\\plugins\\python-ce\\helpers\\pydev\\pydevd.py', 'C:\\Program Files\\JetBrains\\PyCharm Community Edition 2024.1.1\\plugins\\python-ce\\helpers\\pydev\\pydevd.py', 'C:\\Program Files\\JetBrains\\PyCharm Community Edition 2024.1.1\\plugins\\python-ce\\helpers\\pydev\\pydevd.py', 'C:\\Program Files\\JetBrains\\PyCharm Community Edition 2024.1.1\\plugins\\python-ce\\helpers\\pydev\\pydevd.py']
    common_file_name = "youtube-comment-analysis-app-python"
    filtered_filenames = [filename for filename in call_filenames if filename.endswith(".py") and common_file_name in filename]
    if len(filtered_filenames)==0:
        return "Not Found"
    caller_filename = os.path.basename(filtered_filenames[-1])
    return caller_filename

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