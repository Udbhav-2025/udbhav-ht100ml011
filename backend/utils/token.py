from config import JWT_SECRET
from datetime import datetime, timedelta


def _get_jwt_encoder():
    """Return a callable encoder(payload, key, algorithm=...) or raise a helpful error.

    Tries common JWT libraries: PyJWT (`jwt.encode`) or python-jose (`jose.jwt.encode`).
    """
    try:
        import jwt as pyjwt
        if hasattr(pyjwt, "encode"):
            return pyjwt.encode
    except Exception:
        pyjwt = None

    try:
        # python-jose
        from jose import jwt as jose_jwt

        if hasattr(jose_jwt, "encode"):
            return jose_jwt.encode
    except Exception:
        pass

    raise RuntimeError(
        "No compatible JWT library found. Install 'PyJWT' (`pip install pyjwt`) "
        "or 'python-jose' (`pip install python-jose`) and ensure no conflicting package named 'jwt' is installed."
    )


def _get_jwt_decoder():
    """Return a callable decoder(token, key, algorithms=[...]) or raise a helpful error."""
    try:
        import jwt as pyjwt

        if hasattr(pyjwt, "decode"):
            # PyJWT-style API
            return lambda t, k, algs=None: pyjwt.decode(t, k, algorithms=algs or ["HS256"])
    except Exception:
        pyjwt = None

    try:
        from jose import jwt as jose_jwt

        if hasattr(jose_jwt, "decode"):
            return lambda t, k, algs=None: jose_jwt.decode(t, k, algorithms=algs or ["HS256"])
    except Exception:
        pass

    raise RuntimeError(
        "No compatible JWT library found for decoding. Install 'PyJWT' or 'python-jose'."
    )


def generate_token(email, name=None, role=None):
    payload = {
        "email": email,
        "exp": datetime.utcnow() + timedelta(days=1),
    }
    if name is not None:
        payload["name"] = name
    if role is not None:
        payload["role"] = role

    encoder = _get_jwt_encoder()
    token = encoder(payload, JWT_SECRET, algorithm="HS256")
    # PyJWT may return bytes in older versions; normalize to str
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token


def decode_token(token):
    """Decode a JWT and return its payload.

    Raises an exception if the token is invalid or expired.
    """
    decoder = _get_jwt_decoder()
    return decoder(token, JWT_SECRET, algs=["HS256"])
