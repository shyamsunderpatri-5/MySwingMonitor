"""
auto_token_refresh.py
=====================
Automatically logs into Zerodha Kite and refreshes the access token
WITHOUT any browser or manual intervention.

Requirements:
    pip install kiteconnect pyotp requests

Secrets needed (Streamlit Secrets or GitHub Secrets):
    KITE_API_KEY
    KITE_API_SECRET
    KITE_USER_ID        ← Your Zerodha login ID (e.g. AB1234)
    KITE_PASSWORD       ← Your Zerodha password
    KITE_TOTP_SECRET    ← Base32 TOTP secret from Zerodha's 2FA setup page
"""

import logging
import time
import requests
import pyotp

logger = logging.getLogger(__name__)

# ── Import credentials & token helpers from your existing config ──────────────
from config_kite import (
    KITE_API_KEY, KITE_API_SECRET,
    KITE_USER_ID, KITE_PASSWORD, KITE_TOTP_SECRET,
    save_token, can_auto_refresh
)

try:
    from kiteconnect import KiteConnect
    HAS_KITE = True
except ImportError:
    HAS_KITE = False


# ── Zerodha internal login endpoints (official flow, no browser needed) ───────
_BASE        = "https://kite.zerodha.com"
_LOGIN_URL   = f"{_BASE}/api/login"
_TWOFA_URL   = f"{_BASE}/api/twofa"
_CONNECT_URL = f"{_BASE}/connect/login"


def _get_totp() -> str:
    """Generate a fresh TOTP code from the base32 secret."""
    totp = pyotp.TOTP(KITE_TOTP_SECRET)
    return totp.now()


def _zerodha_login_session() -> str:
    """
    Perform a headless Zerodha login and return the request_token.

    Steps:
      1. POST /api/login  → get request_id (2FA challenge)
      2. POST /api/twofa  → completes login, sets session cookie
      3. GET  /connect/login?api_key=...  → redirects with request_token in URL
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
        "Referer": "https://kite.zerodha.com/",
        "X-Kite-Version": "2.9.3",
    })

    # ── Step 1: Username + Password ──────────────────────────────────────────
    logger.info("Step 1: Sending login credentials…")
    resp = session.post(_LOGIN_URL, data={
        "user_id": KITE_USER_ID,
        "password": KITE_PASSWORD,
    })
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "success":
        raise RuntimeError(f"Login failed: {data.get('message', data)}")

    request_id = data["data"]["request_id"]
    logger.info("Step 1 OK — got request_id")

    # ── Step 2: TOTP ─────────────────────────────────────────────────────────
    totp_code = _get_totp()
    logger.info("Step 2: Submitting TOTP…")
    resp = session.post(_TWOFA_URL, data={
        "user_id":    KITE_USER_ID,
        "request_id": request_id,
        "twofa_value": totp_code,
        "twofa_type": "totp",
        "skip_session": "",
    })
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "success":
        raise RuntimeError(f"2FA failed: {data.get('message', data)}")

    logger.info("Step 2 OK — 2FA passed")

    # ── Step 3: Get request_token via Connect redirect ────────────────────────
    logger.info("Step 3: Fetching request_token from Connect URL…")
    resp = session.get(
        _CONNECT_URL,
        params={"api_key": KITE_API_KEY, "v": "3"},
        allow_redirects=True,
    )
    # The final URL contains ?request_token=XXXX&action=login&status=success
    final_url = resp.url
    if "request_token" not in final_url:
        raise RuntimeError(
            f"request_token not found in redirect URL: {final_url}\n"
            "Make sure your app's redirect URL is set to https://127.0.0.1 "
            "in the Kite Connect developer console."
        )

    from urllib.parse import urlparse, parse_qs
    params     = parse_qs(urlparse(final_url).query)
    req_token  = params["request_token"][0]
    logger.info("Step 3 OK — got request_token")
    return req_token


def refresh_token_auto() -> tuple[bool, str]:
    """
    Public entry point called by your app / scheduler.

    Returns: (success: bool, message: str)
    """
    if not HAS_KITE:
        return False, "kiteconnect not installed. Run: pip install kiteconnect"

    if not can_auto_refresh():
        return False, (
            "Auto-refresh credentials not set. "
            "Add KITE_USER_ID, KITE_PASSWORD, KITE_TOTP_SECRET to secrets."
        )

    try:
        # Get fresh request_token via headless login
        request_token = _zerodha_login_session()

        # Exchange request_token for access_token using kiteconnect
        kite = KiteConnect(api_key=KITE_API_KEY)
        session_data  = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
        access_token  = session_data["access_token"]

        # Persist token (re-uses your existing save_token helper)
        save_token(access_token, request_token)

        logger.info("✅ Token refreshed successfully via auto-login")
        return True, "Token refreshed successfully."

    except Exception as exc:
        logger.error(f"Auto token refresh failed: {exc}", exc_info=True)
        return False, f"Auto-refresh failed: {exc}"


# ── Convenience: run directly to test ────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok, msg = refresh_token_auto()
    print(f"{'✅' if ok else '❌'} {msg}")
