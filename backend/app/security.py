import datetime as dt

from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import settings

pwd = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
ALGO = "HS256"


def hash_password(p): return pwd.hash(p)
def verify_password(p, h): return pwd.verify(p, h)


def create_token(sub: str, role: str):
    exp = dt.datetime.utcnow() + dt.timedelta(minutes=settings.access_token_minutes)
    return jwt.encode({"sub": sub, "role": role, "exp": exp}, settings.secret_key, algorithm=ALGO)


def decode_token(t):
    try:
        return jwt.decode(t, settings.secret_key, algorithms=[ALGO])
    except JWTError:
        return None
