# app/services/auth.py
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import jwt, JWTError
from passlib.context import CryptContext
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return _pwd_context.verify(password, password_hash)


def create_access_token(subject: str, expires_minutes: Optional[int] = None) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes if expires_minutes is not None else settings.jwt_expire_minutes
    )
    payload = {"sub": subject, "exp": expire, "type": "admin_access"}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> Optional[str]:
    """Returns the subject (username) if the token is valid, else None"""
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        if payload.get("type") != "admin_access":
            return None
        return payload.get("sub")
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        return None


def authenticate_admin(username: str, password: str) -> bool:
    """
    Checks credentials against settings.admin_username / settings.admin_password_hash.
    If no password hash is configured (fresh install), authentication is refused
    rather than silently allowing any password.
    """
    if username != settings.admin_username:
        return False
    if not settings.admin_password_hash:
        logger.error("ADMIN_PASSWORD_HASH is not configured; refusing admin login")
        return False
    return verify_password(password, settings.admin_password_hash)
