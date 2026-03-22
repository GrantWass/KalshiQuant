"""
kalshi/auth.py — Kalshi RSA-PSS request signer.

Kalshi uses RSA key-based authentication. Every request must include three headers:
  KALSHI-ACCESS-KEY       — the Key ID from your Kalshi account
  KALSHI-ACCESS-TIMESTAMP — current time in milliseconds (string)
  KALSHI-ACCESS-SIGNATURE — base64(RSA-PSS-SHA256(timestamp + METHOD + path))

The path must NOT include query parameters.

Setup:
  1. Go to Kalshi → Profile → API Keys → Create Key
  2. Save the private key to a file (e.g. kalshi_private.pem)
  3. Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH in your .env

Usage:
    headers = signer.get_auth_headers("GET", "/trade-api/v2/markets")
"""

import base64
import logging
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from config.settings import settings

logger = logging.getLogger(__name__)


class KalshiSigner:
    """Signs Kalshi API requests using RSA-PSS/SHA-256."""

    def __init__(self) -> None:
        self._private_key = None

    def _load_key(self):
        if self._private_key is not None:
            return self._private_key

        if settings.KALSHI_PRIVATE_KEY:
            # Key content provided directly via env var.
            # Replace literal \n with real newlines in case the env var was set that way.
            pem = settings.KALSHI_PRIVATE_KEY.replace("\\n", "\n").encode("utf-8")
            logger.debug("Kalshi private key loaded from KALSHI_PRIVATE_KEY env var.")
        else:
            key_path = Path(settings.KALSHI_PRIVATE_KEY_PATH)
            if not key_path.exists():
                raise RuntimeError(
                    f"Kalshi private key not found. Set KALSHI_PRIVATE_KEY (PEM content) "
                    f"or KALSHI_PRIVATE_KEY_PATH (path to file) in your .env."
                )
            pem = key_path.read_bytes()
            logger.debug("Kalshi private key loaded from %s", key_path)

        self._private_key = serialization.load_pem_private_key(pem, password=None)
        return self._private_key

    def get_auth_headers(self, method: str, path: str) -> dict:
        """
        Return the three Kalshi auth headers for a request.

        method — HTTP method (GET, POST, DELETE, …)
        path   — URL path only, no query string (e.g. /trade-api/v2/markets)
        """
        timestamp_ms = str(int(time.time() * 1000))
        message = (timestamp_ms + method.upper() + path).encode("utf-8")

        key = self._load_key()
        signature = key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )

        return {
            "KALSHI-ACCESS-KEY": settings.KALSHI_API_KEY_ID,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
        }


# Module-level singleton shared by KalshiClient and KalshiWebSocketManager.
signer = KalshiSigner()
