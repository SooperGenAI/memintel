"""
app/utils/encryption.py
──────────────────────────────────────────────────────────────────────────────
Fernet symmetric encryption for connector credential storage.

encrypt() / decrypt() operate on plain-text strings.  The Fernet key is read
from the MEMINTEL_ENCRYPTION_KEY environment variable at call time.

Key format: URL-safe base64-encoded 32-byte key as produced by Fernet.generate_key().

Usage:
    from app.utils.encryption import encrypt, decrypt

    token = encrypt('{"password": "secret"}')
    plain = decrypt(token)

Raises RuntimeError when MEMINTEL_ENCRYPTION_KEY is not set.
Raises cryptography.fernet.InvalidToken when the ciphertext is tampered or
the wrong key is used.
"""
from __future__ import annotations

import os

from cryptography.fernet import Fernet


def _fernet() -> Fernet:
    key = os.environ.get("MEMINTEL_ENCRYPTION_KEY")
    if not key:
        raise RuntimeError(
            "MEMINTEL_ENCRYPTION_KEY environment variable is not set. "
            "Generate a key with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt(data: str) -> str:
    """Return a Fernet-encrypted, URL-safe base64 token for *data*."""
    return _fernet().encrypt(data.encode()).decode()


def decrypt(token: str) -> str:
    """Decrypt a Fernet token and return the original plain-text string."""
    return _fernet().decrypt(token.encode()).decode()
