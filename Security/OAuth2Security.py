from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext

from src.config.CommonVariables import TokenData, User, RegisterUser, get_settings
from src.db.MySQLDB import MysqlDB


# fake_users_db = {
#     "admin": {
#         "username": "admin",
#         "full_name": "Admin User",
#         "email": "admin@admin.com",
#         "hashed_password": "$2b$12$G6Qw5e.K871doase2mJqgepPaB7frMIWb973E9zspNl3dNrHSik8C",#"$2b$12$cEgVZpjKEu79bo97d.h1muyLfV2U/JqNdslg/T/0cEJwcBvdbyKem",
#         "disabled": False,
#     }
# }



SECRET_KEY = get_settings().SECRET_KEY
ALGORITHM = get_settings().ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = int(get_settings().ACCESS_TOKEN_EXPIRE_MINUTES)

mysqlDB = MysqlDB()


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# class OAuth2Security:
#
#     def __init__(self):
#         self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
#         self.settings = self.get_settings()
#         self.SECRET_KEY = self.settings.SECRET_KEY
#         self.ALGORITHM = self.settings.ALGORITHM
#         self.ACCESS_TOKEN_EXPIRE_MINUTES = self.settings.ACCESS_TOKEN_EXPIRE_MINUTES
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# def oauth2_scheme():
#     return OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(username: str):
    print("Called get_user()")
    result = mysqlDB.get_user_by_username(username)
    if len(result)>0:
        return result[0]
    # if username in db:
    #     user_dict = db[username]
    #     return UserInDB(**user_dict)

def authenticate_user(username: str, password: str):
    print("Called authenticate_user()")
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        print("expires_delta", expires_delta)
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        print("expires in ", 60, "minutes")
        expire = datetime.now(timezone.utc) + timedelta(minutes=1)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials!",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token_expired_exception = HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Token is expired!",
        headers={"WWW-Authenticate": "Bearer", "REASON": "TOKEN_EXPIRED"},
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

    except jwt.ExpiredSignatureError:
        raise token_expired_exception
    except InvalidTokenError:
        raise credentials_exception
    user = get_user(username=token_data.username)
    print(user)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: Annotated[User, Depends(get_current_user)],
):
    print(current_user)
    if current_user['disabled']:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# async def get_all_user():
#     return sorted(list(fake_users_db.keys()))

async def add_user(user):
    forbidden_exception_ref_code = HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid refferal code!",
        headers={"WWW-Authenticate": "Bearer"},
    )
    forbidden_exception_user_exists = HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="User already exists!",
        headers={"WWW-Authenticate": "Bearer"},
    )
    settings = get_settings()
    temp_user = dict(user)
    current_user = get_user(temp_user['username'])
    if (current_user is not None):
        raise forbidden_exception_user_exists
    if 'referral_code' in temp_user:
        if (temp_user['referral_code'] != settings.REFFERAL_CODE):
            raise forbidden_exception_ref_code
    else:
        raise forbidden_exception_ref_code
    if 'password' in temp_user:
        temp_user['hashed_password'] = get_password_hash(temp_user['password'])
        del temp_user['password']
    to_be_registered_user = RegisterUser(**temp_user)
    mysqlDB.create_user(to_be_registered_user)
    # fake_users_db[temp_user['username']] = temp_user
    # print(fake_users_db)
    return temp_user["username"]

if __name__ == "__main__":
    search_text = "kolkata restaurants"
    # token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJqb2huZG9lIiwiZXhwIjoxNzI2NTEwMzUzfQ.YP4oZ6q070sa8owmo9rx20qjaeTYoCL0npImN8rHV4A"
    # token_60 = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJqb2huZG9lIiwiZXhwIjoxNzI2NTE0MzQ4fQ.RmMT_tJkw-MUIuvS0Albou0ReDzg80wiNsa6gd7dD40"
    # token_1 = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJqb2huZG9lIiwiZXhwIjoxNzI2NTEwODQ3fQ.gF5rLNkMNF6KJ5vbARYlm5TivxGiIgfQxSSG1rbQG-w"
    # token_1 = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJqb2huZG9lIiwiZXhwIjoxNzI2NTEwODQ3fQ.gF5rLNkMNF6KJ5vbARYlm5TivxGiIgfQxSSG1rb-w00"
    # res = get_current_user(token_1)
    # token = create_access_token({"sub": "johndoe"})
    # print(token)
    # response = await get_current_user(token)
    # res = get_current_user_classic(token)
    # print(res)
    #$2b$12$cEgVZpjKEu79bo97d.h1muyLfV2U/JqNdslg/T/0cEJwcBvdbyKem
    # print(get_password_hash("admin111"))

    # username, full_name, email, hashed_password, disabled
    # register_user = dict()
    # register_user["username"] = "admin3"
    # register_user["full_name"] = "Admin User2"
    # register_user["email"] = "admin2@admin.com"
    # register_user["hashed_password"] = "$2b$12$hQhRH/casNbFntPFTXn0QOZQ/4PI/krp.kCCS6K0TknXykVualRxm"
    # register_user["disabled"] = 0
    # register_user["referral_code"] = "allowed123"
    # register_user_obj = RegisterUser(**register_user)
    # print(register_user_obj)
    # mysqlDB.start_connection()
    # import asyncio
    # asyncio.run(add_user(register_user_obj))
    # mysqlDB.close_connection()
