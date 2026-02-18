"""
Zerodha Kite Connect Configuration
===================================
Reads credentials from Streamlit Secrets (Cloud) or
environment variables (local/GitHub Actions).
"""

import os
import json
import logging
import streamlit as st
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# LOAD CREDENTIALS — Streamlit Secrets → Env Vars → Empty
# ============================================================================

def _load_credentials():
    """
    Load API credentials safely from Streamlit secrets or env vars.
    Never hardcode keys here.
    """
    api_key = ""
    api_secret = ""

    # Try Streamlit Secrets first (Streamlit Cloud)
    try:
        api_key = st.secrets.get("KITE_API_KEY", "")
        api_secret = st.secrets.get("KITE_API_SECRET", "")
    except Exception:
        pass

    # Fallback to environment variables (local terminal / GitHub Actions)
    if not api_key:
        api_key = os.environ.get("KITE_API_KEY", "")
    if not api_secret:
        api_secret = os.environ.get("KITE_API_SECRET", "")

    return api_key, api_secret


KITE_API_KEY, KITE_API_SECRET = _load_credentials()

# ============================================================================
# TOKEN STORAGE
# ============================================================================

TOKEN_FILE = "kite_token.json"


def save_token(access_token, request_token=""):
    token_data = {
        "access_token": access_token,
        "request_token": request_token,
        "created_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
        "api_key": KITE_API_KEY
    }
    try:
        with open(TOKEN_FILE, "w") as f:
            json.dump(token_data, f, indent=2)
        logger.info(f"Token saved at {datetime.now()}")
        return True
    except Exception as e:
        logger.error(f"Failed to save token: {e}")
        return False


def load_token():
    try:
        if not Path(TOKEN_FILE).exists():
            return None, False, "No token file found. Please login."

        with open(TOKEN_FILE, "r") as f:
            token_data = json.load(f)

        access_token = token_data.get("access_token", "")
        expires_at_str = token_data.get("expires_at", "")

        if not access_token:
            return None, False, "Token file is empty. Please login."

        try:
            expires_at = datetime.fromisoformat(expires_at_str)
            if datetime.now() >= expires_at:
                return None, False, "Token expired. Please login again."
            remaining = expires_at - datetime.now()
            hours_left = remaining.total_seconds() / 3600
            return access_token, True, f"Token valid. Expires in {hours_left:.1f} hours."
        except (ValueError, TypeError):
            return access_token, False, "Cannot verify expiry."

    except json.JSONDecodeError:
        return None, False, "Token file corrupted. Please login again."
    except Exception as e:
        return None, False, f"Error loading token: {e}"


def clear_token():
    try:
        if Path(TOKEN_FILE).exists():
            os.remove(TOKEN_FILE)
        return True
    except Exception as e:
        logger.error(f"Failed to clear token: {e}")
        return False


def is_configured():
    return bool(KITE_API_KEY and KITE_API_SECRET)


def get_login_url():
    if not KITE_API_KEY:
        return None, "API Key not configured"
    url = f"https://kite.zerodha.com/connect/login?v=3&api_key={KITE_API_KEY}"
    return url, "OK"


# ============================================================================
# ORDER TYPE CONSTANTS
# ============================================================================

class OrderType:
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"

class TransactionType:
    BUY = "BUY"
    SELL = "SELL"

class ProductType:
    CNC = "CNC"
    MIS = "MIS"
    NRML = "NRML"

class Exchange:
    NSE = "NSE"
    BSE = "BSE"
    NFO = "NFO"

class OrderStatus:
    COMPLETE = "COMPLETE"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    OPEN = "OPEN"
    PENDING = "PENDING"
    TRIGGER_PENDING = "TRIGGER PENDING"
