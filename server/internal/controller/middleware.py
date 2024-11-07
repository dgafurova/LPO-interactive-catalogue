from fastapi import Depends, HTTPException, status
from internal.app.config import api_token
from typing import Optional
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer


get_bearer_token = HTTPBearer(auto_error=False)


def verify_token(auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token)):
    if auth.credentials != api_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
