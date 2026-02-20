"""
auto_token_refresh.py
=====================
FIX 6: Automated daily Zerodha Kite token refresh using TOTP (pyotp).

HOW IT WORKS
────────────
Zerodha requires a browser login once per day.  This module automates the
exact same HTTP calls the browser makes:

  1. POST /api/login       → user_id + password
  2. POST /api/twofa       → TOTP code (from your authenticator secret)
  3. Capture request_token from the redirect URL
  4. Exchange request_token → access_token via kiteconnect SDK
  5. Save the token to kite_token.json (read by config_kite.py / kite_connector.py)

SETUP
─────
Install deps:  pip install pyotp requests kiteconnect

Store these in GitHub Secrets / Streamlit Secrets / environment variables:
  KITE_API_KEY        – your Zerodha API key
  KITE_API_SECRET     – your Zerodha API secret
  KITE_USER_ID        – Zerodha login ID  (e.g. "AB1234")
  KITE_PASSWORD       – Zerodha password
  KITE_TOTP_SECRET    – Base32 TOTP secret from Zerodha 2FA setup
                        (shown once when you enable TOTP; looks like "JBSWY3DPEHPK3PXP")

USAGE
─────
  # Manual call (e.g. from GitHub Actions before scanner runs)
  python auto_token_refresh.py

  # Programmatic call from another module
  from auto_token_refresh import refresh_token_auto
  success, msg = refresh_token_auto()

GITHUB ACTIONS INTEGRATION
───────────────────────────
Add a step BEFORE the scanner step:

  - name: Refresh Kite Token
    env:
      KITE_API_KEY:     ${{ secrets.KITE_API_KEY }}
      KITE_API_SECRET:  ${{ secrets.KITE_API_SECRET }}
      KITE_USER_ID:     ${{ secrets.KITE_USER_ID }}
      KITE_PASSWORD:    ${{ secrets.KITE_PASSWORD }}
      KITE_TOTP_SECRET: ${{ secrets.KITE_TOTP_SECRET }}
    run: python auto_token_refresh.py
"""

import os
import re
import sys
import logging
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Optional: use pyotp for TOTP ────────────────────────────────────────────
try:
    import pyotp
    HAS_PYOTP = True
except ImportError:
    HAS_PYOTP = False
    logger.error("pyotp not installed.  Run:  pip install pyotp")

# ── Optional: use requests for HTTP ─────────────────────────────────────────
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.error("requests not installed.  Run:  pip install requests")

# ── Optional: kiteconnect ────────────────────────────────────────────────────
try:
    from kiteconnect import KiteConnect
    HAS_KITE_SDK = True
except ImportError:
    HAS_KITE_SDK = False
    logger.error("kiteconnect not installed.  Run:  pip install kiteconnect")


# ─────────────────────────────────────────────────────────────────────────────
# CORE REFRESH FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def refresh_token_auto(
    api_key: str = "",
    api_secret: str = "",
    user_id: str = "",
    password: str = "",
    totp_secret: str = "",
) -> tuple:
    """
    Fully automated token refresh for Zerodha Kite.

    All parameters fall back to environment variables when not supplied.

    Returns: (success: bool, message: str)
    """

    # ── Load from env if not provided ───────────────────────────────────────
    api_key    = api_key    or os.environ.get("KITE_API_KEY", "")
    api_secret = api_secret or os.environ.get("KITE_API_SECRET", "")
    user_id    = user_id    or os.environ.get("KITE_USER_ID", "")
    password   = password   or os.environ.get("KITE_PASSWORD", "")
    totp_secret = totp_secret or os.environ.get("KITE_TOTP_SECRET", "")

    # ── Dependency check ────────────────────────────────────────────────────
    missing_deps = []
    if not HAS_PYOTP:      missing_deps.append("pyotp")
    if not HAS_REQUESTS:   missing_deps.append("requests")
    if not HAS_KITE_SDK:   missing_deps.append("kiteconnect")
    if missing_deps:
        return False, f"Missing packages: {', '.join(missing_deps)}.  pip install {' '.join(missing_deps)}"

    # ── Credential check ────────────────────────────────────────────────────
    missing_creds = []
    if not api_key:      missing_creds.append("KITE_API_KEY")
    if not api_secret:   missing_creds.append("KITE_API_SECRET")
    if not user_id:      missing_creds.append("KITE_USER_ID")
    if not password:     missing_creds.append("KITE_PASSWORD")
    if not totp_secret:  missing_creds.append("KITE_TOTP_SECRET")
    if missing_creds:
        return False, f"Missing credentials: {', '.join(missing_creds)}"

    # ── Step 1: Check if existing token is still valid ───────────────────────
    try:
        from config_kite import load_token
        access_token, is_valid, token_msg = load_token()
        if is_valid:
            logger.info(f"Token still valid — skipping refresh.  {token_msg}")
            return True, f"Token already valid: {token_msg}"
    except Exception:
        pass  # config_kite not available, proceed with refresh

    logger.info("Starting automated Zerodha token refresh...")

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": "https://kite.zerodha.com/",
        "Origin": "https://kite.zerodha.com",
        "X-Kite-Version": "3",
    })

    # ── Step 2: Username + Password login ────────────────────────────────────
    try:
        logger.info("Step 1/3 — Submitting credentials...")
        resp = session.post(
            "https://kite.zerodha.com/api/login",
            data={"user_id": user_id, "password": password},
            timeout=30,
        )
        resp.raise_for_status()
        login_data = resp.json()

        if login_data.get("status") != "success":
            err = login_data.get("message", "Unknown error")
            return False, f"Login failed: {err}. Check KITE_USER_ID / KITE_PASSWORD."

        request_id = login_data["data"]["request_id"]
        logger.info("Step 1/3 — Credentials accepted")

    except requests.exceptions.HTTPError as e:
        return False, f"Login HTTP error: {e}.  Check credentials."
    except Exception as e:
        return False, f"Login failed unexpectedly: {e}"

    # ── Step 3: Submit TOTP ──────────────────────────────────────────────────
    try:
        logger.info("Step 2/3 — Submitting TOTP...")

        # Generate TOTP from secret
        totp_code = pyotp.TOTP(totp_secret.strip().upper()).now()
        logger.info(f"TOTP generated: {totp_code} (valid for ~{30 - int(time.time()) % 30}s)")

        resp = session.post(
            "https://kite.zerodha.com/api/twofa",
            data={
                "user_id":    user_id,
                "request_id": request_id,
                "twofa_type": "totp",
                "twofa_value": totp_code,
            },
            timeout=30,
        )
        resp.raise_for_status()
        twofa_data = resp.json()

        if twofa_data.get("status") != "success":
            err = twofa_data.get("message", "Unknown error")
            return False, (
                f"2FA failed: {err}.  "
                "Check KITE_TOTP_SECRET — it should be the Base32 key "
                "shown once when you enabled TOTP (not the 6-digit code)."
            )

        logger.info("Step 2/3 — TOTP accepted")

    except requests.exceptions.HTTPError as e:
        return False, f"2FA HTTP error: {e}. Check KITE_TOTP_SECRET."
    except Exception as e:
        return False, f"2FA failed unexpectedly: {e}"

    # ── Step 4: Capture request_token from redirect URL ─────────────────────
    try:
        logger.info("Step 3/3 — Capturing request_token from redirect...")

        # Build the Kite login URL (same as what the browser visits)
        login_url = (
            f"https://kite.zerodha.com/connect/login?"
            f"v=3&api_key={api_key}"
        )
        resp = session.get(login_url, allow_redirects=True, timeout=30)

        # Kite redirects to your registered redirect_url with ?request_token=...
        # Extract it from the final URL
        final_url = resp.url
        match = re.search(r"[?&]request_token=([^&]+)", final_url)

        if not match:
            # Try looking in the HTML body (some setups embed it)
            match = re.search(r"request_token[\":\s]+([A-Za-z0-9]+)", resp.text)

        if not match:
            return False, (
                "Could not find request_token in redirect URL.\n"
                f"Final URL was: {final_url}\n"
                "Make sure your Kite app's redirect URL is set and accessible."
            )

        request_token = match.group(1)
        logger.info(f"request_token captured: {request_token[:8]}...")

    except Exception as e:
        return False, f"Could not capture request_token: {e}"

    # ── Step 5: Exchange request_token for access_token ──────────────────────
    try:
        kite = KiteConnect(api_key=api_key)
        session_data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = session_data["access_token"]

        # Verify it works
        kite.set_access_token(access_token)
        profile = kite.profile()
        user_name = profile.get("user_name", user_id)

        logger.info(f"Token generated successfully for {user_name}")

    except Exception as e:
        return False, f"Token generation failed: {e}"

    # ── Step 6: Save token ───────────────────────────────────────────────────
    try:
        from config_kite import save_token
        save_token(access_token, request_token)
        logger.info("Token saved to kite_token.json")
    except Exception:
        # Manual save fallback if config_kite unavailable
        import json
        from pathlib import Path
        token_data = {
            "access_token": access_token,
            "request_token": request_token,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
            "api_key": api_key,
        }
        with open("kite_token.json", "w") as f:
            json.dump(token_data, f, indent=2)
        logger.info("Token saved manually to kite_token.json")

    return True, f"Token refreshed successfully for {user_name}. Valid 24 hours."


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULED REFRESH  (run as a daily cron / GitHub Actions job)
# ─────────────────────────────────────────────────────────────────────────────

def run_scheduled_refresh(retries: int = 3, retry_delay: int = 30) -> int:
    """
    Entry point for GitHub Actions / cron.
    Returns 0 on success, 1 on failure.
    """
    for attempt in range(1, retries + 1):
        logger.info(f"Token refresh attempt {attempt}/{retries}...")
        success, msg = refresh_token_auto()

        if success:
            logger.info(f"✅ {msg}")
            return 0

        logger.error(f"❌ Attempt {attempt} failed: {msg}")

        if attempt < retries:
            logger.info(f"Retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    logger.critical("All refresh attempts failed.  Manual login required.")
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(run_scheduled_refresh())
