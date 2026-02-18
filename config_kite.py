"""
Zerodha Kite Connect Configuration
===================================
This file manages API keys, tokens, and session state for Zerodha integration.

SETUP INSTRUCTIONS:
1. Subscribe to Kite Connect API (https://kite.trade)
2. Get your api_key and api_secret from Kite Console
3. Fill in KITE_API_KEY and KITE_API_SECRET below
4. Every day at 9:00 AM, generate a request_token via the login URL
5. The app will auto-exchange it for an access_token
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# YOUR ZERODHA CREDENTIALS - FILL THESE IN
# ============================================================================

# Option 1: Hardcode (Simple but less secure)
KITE_API_KEY = ""         # e.g., "abc123xyz"
KITE_API_SECRET = ""      # e.g., "def456uvw789"

# Option 2: Environment Variables (More secure - recommended for deployment)
# Set these in your terminal or Streamlit secrets:
#   export KITE_API_KEY="your_key"
#   export KITE_API_SECRET="your_secret"
if not KITE_API_KEY:
    KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
if not KITE_API_SECRET:
    KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")

# ============================================================================
# TOKEN STORAGE (Persists across app restarts)
# ============================================================================

TOKEN_FILE = "kite_token.json"  # Stores token on disk


def save_token(access_token, request_token=""):
    """
    Save access token to file with timestamp.
    Called after successful login.
    """
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
        logger.info(f"Token saved successfully at {datetime.now()}")
        return True
    except Exception as e:
        logger.error(f"Failed to save token: {e}")
        return False


def load_token():
    """
    Load access token from file.
    Returns: (access_token, is_valid, message)
    """
    try:
        if not Path(TOKEN_FILE).exists():
            return None, False, "No token file found. Please login."
        
        with open(TOKEN_FILE, "r") as f:
            token_data = json.load(f)
        
        access_token = token_data.get("access_token", "")
        expires_at_str = token_data.get("expires_at", "")
        
        if not access_token:
            return None, False, "Token file is empty. Please login."
        
        # Check expiry
        try:
            expires_at = datetime.fromisoformat(expires_at_str)
            now = datetime.now()
            
            if now >= expires_at:
                return None, False, "Token expired. Please login again."
            
            remaining = expires_at - now
            hours_left = remaining.total_seconds() / 3600
            
            return (
                access_token,
                True,
                f"Token valid. Expires in {hours_left:.1f} hours."
            )
        except (ValueError, TypeError):
            return access_token, False, "Cannot verify expiry. Token may be invalid."
    
    except json.JSONDecodeError:
        return None, False, "Token file corrupted. Please login again."
    except Exception as e:
        return None, False, f"Error loading token: {e}"


def clear_token():
    """Delete saved token (for logout)"""
    try:
        if Path(TOKEN_FILE).exists():
            os.remove(TOKEN_FILE)
            logger.info("Token cleared")
        return True
    except Exception as e:
        logger.error(f"Failed to clear token: {e}")
        return False


def is_configured():
    """Check if API credentials are configured"""
    return bool(KITE_API_KEY and KITE_API_SECRET)


def get_login_url():
    """Generate the Kite Connect login URL"""
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
    SL = "SL"              # Stop Loss (with trigger price)
    SL_M = "SL-M"          # Stop Loss Market (trigger only, no limit)


class TransactionType:
    BUY = "BUY"
    SELL = "SELL"


class ProductType:
    CNC = "CNC"            # Cash and Carry (Delivery)
    MIS = "MIS"            # Margin Intraday Settlement
    NRML = "NRML"          # Normal (F&O)


class Exchange:
    NSE = "NSE"
    BSE = "BSE"
    NFO = "NFO"            # F&O segment


class OrderStatus:
    COMPLETE = "COMPLETE"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    OPEN = "OPEN"
    PENDING = "PENDING"
    TRIGGER_PENDING = "TRIGGER PENDING"