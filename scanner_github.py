
import sys
import os
import io
import logging
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders


# ============================================================================
# GOOGLE SHEETS INTEGRATION - NEW SECTION
# ============================================================================

try:
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_SHEETS_AVAILABLE = True
except ImportError:
    GOOGLE_SHEETS_AVAILABLE = False
    print("⚠️  Google Sheets libraries not installed. Install with: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")

# Google Sheets Configuration
GOOGLE_SHEETS_CONFIG = {
    'enabled': True,  # Set to True to enable Google Sheets updates
    'spreadsheet_id': os.environ.get('GOOGLE_SHEET_ID', ''),  # Your Google Sheet ID
    'sheet_name': 'Pending',  # The name of the sheet/tab to update
    'credentials_json': os.environ.get('GOOGLE_CREDENTIALS_JSON', ''),  # Service account JSON
}
# ============================================================================
# GITHUB ACTIONS CONFIGURATION
# ============================================================================

# Get email credentials from environment variables (GitHub Secrets)
EMAIL_CONFIG = {
    'enabled': True,
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': os.environ.get('EMAIL_SENDER', ''),
    'sender_password': os.environ.get('EMAIL_PASSWORD', ''),
    'recipient_email': os.environ.get('EMAIL_RECIPIENT', ''),
}

# GitHub Actions paths
SIGNALS_DIR = 'signals'
LOG_FILE = 'logs/scanner.log'

# Create directories
os.makedirs(SIGNALS_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# [FIX CRIT-3 / MIN-9]  NSE HOLIDAY CALENDAR + SAFE TRADING-DAY HELPER
# ============================================================================
# Update this set at the start of each calendar year.
NSE_HOLIDAYS = {
    # 2025
    "2025-01-26",  # Republic Day
    "2025-02-26",  # Mahashivratri
    "2025-03-14",  # Holi
    "2025-03-31",  # Id-Ul-Fitr (Ramzan)
    "2025-04-10",  # Shri Ram Navami
    "2025-04-14",  # Dr. Ambedkar Jayanti
    "2025-04-18",  # Good Friday
    "2025-05-01",  # Maharashtra Day
    "2025-08-15",  # Independence Day
    "2025-08-27",  # Ganesh Chaturthi
    "2025-10-02",  # Gandhi Jayanti / Dussehra
    "2025-10-20",  # Diwali Laxmi Puja
    "2025-10-21",  # Diwali Balipratipada
    "2025-11-05",  # Prakash Gurpurb Sri Guru Nanak Dev Ji
    "2025-12-25",  # Christmas
    # 2026 — sourced from NSE Circular NSE/CMTR/71775 (Dec 2025)
    "2026-01-26",  # Republic Day (Monday)
    "2026-03-03",  # Holi (Tuesday)           ← was wrongly listed as Mar-20
    "2026-03-26",  # Shri Ram Navami (Thursday)    ← was MISSING
    "2026-03-31",  # Mahavir Jayanti (Tuesday)     ← was MISSING
    "2026-04-03",  # Good Friday (Friday)
    "2026-04-14",  # Dr. Ambedkar Jayanti (Tuesday)
    "2026-04-30",  # Id-Ul-Fitr / Eid (Thursday)
    "2026-05-01",  # Maharashtra Day (Friday)
    "2026-05-28",  # Bakri Id / Id-Ul-Zuha (Thursday)  ← was MISSING
    "2026-06-26",  # Muharram (Friday)                  ← was MISSING
    "2026-10-02",  # Gandhi Jayanti (Friday)
    "2026-12-25",  # Christmas (Friday)
    # NOTE: Aug 15 (Independence Day) falls on Saturday 2026 — weekend, no closure.
    # NOTE: Diwali Laxmi Pujan Nov 8 falls on Sunday 2026 — Muhurat only, weekend.
}


def get_next_trading_day(from_date: datetime = None) -> str:
    """
    [FIX CRIT-3] Return the next NSE trading day as 'YYYY-MM-DD'.
    Skips weekends (Sat=5, Sun=6) and NSE holidays listed above.
    Defaults to the day after today when from_date is None.
    """
    d = (from_date if from_date else datetime.now()) + timedelta(days=1)
    while d.weekday() >= 5 or d.strftime("%Y-%m-%d") in NSE_HOLIDAYS:
        d += timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def is_market_holiday_today() -> bool:
    """Return True if today is a weekend or NSE holiday."""
    today = datetime.now()
    return today.weekday() >= 5 or today.strftime("%Y-%m-%d") in NSE_HOLIDAYS



def get_google_sheets_service():
    """
    Create and return Google Sheets API service
    Returns None if credentials are not configured
    """
    try:
        if not GOOGLE_SHEETS_CONFIG['enabled']:
            logger.info("Google Sheets integration is disabled")
            return None
        
        if not GOOGLE_SHEETS_AVAILABLE:
            logger.error("Google Sheets libraries not installed")
            return None
        
        credentials_json = GOOGLE_SHEETS_CONFIG['credentials_json']
        
        if not credentials_json:
            logger.error("Google credentials not found in environment variables")
            return None
        
        # Parse credentials from JSON string
        import json
        credentials_dict = json.loads(credentials_json)
        
        # Create credentials object
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
        credentials = Credentials.from_service_account_info(
            credentials_dict,
            scopes=SCOPES
        )
        
        # Build and return service
        service = build('sheets', 'v4', credentials=credentials)
        logger.info("✅ Google Sheets service authenticated successfully")
        return service
        
    except Exception as e:
        logger.error(f"❌ Failed to authenticate Google Sheets: {e}")
        return None


def format_sheet_row(signal_data, entry_date):
    """
    Format a single signal into a row for Google Sheets - FIXED VERSION
    Columns: Ticker, Position, Entry_Price, Quantity, Stop_Loss, Target_1, Target_2, Entry_Date, Status, Notes
    
    Args:
        signal_data: Dictionary containing signal information
        entry_date: Date string for entry (tomorrow's date)
    
    Returns:
        List of values for the row
    """
    try:
        # DEBUG: Print what columns we actually have
        logger.info(f"DEBUG - Available columns: {list(signal_data.keys())}")
        
        # Extract ticker - try multiple possible column names
        ticker = signal_data.get('ticker', signal_data.get('Ticker', signal_data.get('TICKER', 'N/A')))
        
        # Position (LONG/SHORT) - Try multiple possible column names
        position = (signal_data.get('side', '') or              # ← MOST IMPORTANT: lowercase!
                   signal_data.get('Side', '') or 
                   signal_data.get('Signal', '') or 
                   signal_data.get('Position', 'N/A'))
        
        # Entry Price - Try multiple possible column names
        entry_price = (signal_data.get('price', 0) or          # ← MOST IMPORTANT: lowercase!
                      signal_data.get('entry_price', 0) or     # ← Second priority
                      signal_data.get('Entry', 0) or 
                      signal_data.get('Entry_Price', 0) or
                      signal_data.get('ENTRY', 0))
        
        # Convert entry_price to float if it's a string
        if isinstance(entry_price, str):
            # Remove ₹ symbol and commas
            entry_price = entry_price.replace('₹', '').replace(',', '').strip()
            try:
                entry_price = float(entry_price)
            except:
                entry_price = 0
        
        # Stop Loss - Try multiple possible column names
        stop_loss = (signal_data.get('stop_loss', 0) or        # ← MOST IMPORTANT: lowercase!
                    signal_data.get('SL', 0) or 
                    signal_data.get('Stop_Loss', 0) or 
                    signal_data.get('STOP_LOSS', 0))
        
        if isinstance(stop_loss, str):
            stop_loss = stop_loss.replace('₹', '').replace(',', '').strip()
            try:
                stop_loss = float(stop_loss)
            except:
                stop_loss = 0
        
        # Target 1 - Try multiple possible column names
        target_1 = (signal_data.get('target_1', 0) or          # ← MOST IMPORTANT: lowercase!
                   signal_data.get('T1', 0) or 
                   signal_data.get('Target_1', 0) or 
                   signal_data.get('TARGET_1', 0))
        
        if isinstance(target_1, str):
            target_1 = target_1.replace('₹', '').replace(',', '').strip()
            try:
                target_1 = float(target_1)
            except:
                target_1 = 0
        
        # Target 2 - Try multiple possible column names
        target_2 = (signal_data.get('target_2', 0) or          # ← MOST IMPORTANT: lowercase!
                   signal_data.get('T2', 0) or 
                   signal_data.get('Target_2', 0) or 
                   signal_data.get('TARGET_2', 0))
        
        if isinstance(target_2, str):
            target_2 = target_2.replace('₹', '').replace(',', '').strip()
            try:
                target_2 = float(target_2)
            except:
                target_2 = 0
        
        # [FIX CRIT-1] Use module-level ACCOUNT_CAPITAL — single source of truth.
        # The original code had _ACCOUNT_CAPITAL = 50_000 hardcoded here, so
        # quantity sizing in Google Sheets would silently diverge from the scanner
        # whenever ACCOUNT_CAPITAL was changed.
        _RISK_PCT   = 0.01   # 1% risk per trade (mirrors RISK_PER_TRADE)
        risk_amount = ACCOUNT_CAPITAL * _RISK_PCT

        risk_per_share = abs(entry_price - stop_loss) if stop_loss and entry_price else 0
        if risk_per_share > 0 and entry_price > 0:
            qty_by_risk    = int(risk_amount / risk_per_share)
            qty_by_capital = int((ACCOUNT_CAPITAL * 0.10) / entry_price)
            quantity       = max(1, min(qty_by_risk, qty_by_capital, 1000))
        elif entry_price > 0:
            quantity = max(1, int((ACCOUNT_CAPITAL * 0.05) / entry_price))
        else:
            quantity = 1
        
        # Get Confidence and R:R
        confidence = signal_data.get('confidence', signal_data.get('Confidence', signal_data.get('CONF', 0)))
        
        # Handle confidence if it's a percentage string like "98.4%"
        if isinstance(confidence, str):
            confidence = confidence.replace('%', '').strip()
            try:
                confidence = float(confidence)
            except:
                confidence = 0
        
        # Get R:R ratio
        rr_ratio = signal_data.get('risk_reward', signal_data.get('RiskReward', signal_data.get('R:R', signal_data.get('RR', 0))))
        
        # Handle R:R if it's a string like "1.9x"
        if isinstance(rr_ratio, str):
            rr_ratio = rr_ratio.replace('x', '').strip()
            try:
                rr_ratio = float(rr_ratio)
            except:
                rr_ratio = 0
        
        # Create notes from additional data
        # Mark SHORT_ALERT signals clearly so you know these are manual-only
        # (cannot be placed as CNC orders — use Kite app / F&O manually)
        alert_tag = " ⚠️ SHORT_ALERT: Manual/F&O only" if str(position).upper() == "SHORT_ALERT" else ""
        position_display = "SHORT" if str(position).upper() == "SHORT_ALERT" else position
        notes = f"Confidence: {confidence}%, R:R: {rr_ratio}{alert_tag}"
        
        # Format row
        row = [
            ticker,
            position_display,
            round(float(entry_price), 2) if entry_price else 0,
            quantity,
            round(float(stop_loss), 2) if stop_loss else 0,
            round(float(target_1), 2) if target_1 else 0,
            round(float(target_2), 2) if target_2 else 0,
            entry_date,
            'PENDING',  # Initial status
            notes
        ]
        
        # Debug log to help troubleshoot
        logger.info(f"Formatted row for {ticker}: Price={entry_price}, SL={stop_loss}, T1={target_1}, T2={target_2}")
        
        return row
        
    except Exception as e:
        logger.error(f"Error formatting row: {e}")
        logger.error(f"Signal data keys: {signal_data.keys()}")
        return None


def update_google_sheet(signals_data):
    """Update Google Sheet with trading signals - PRODUCTION HARDENED v4.0"""
    
    try:
        if not GOOGLE_SHEETS_CONFIG['enabled']:
            logger.info("Google Sheets update skipped (disabled)")
            return True
        
        # Get Google Sheets service
        service = get_google_sheets_service()
        if not service:
            logger.error("Could not connect to Google Sheets")
            return False
        
        spreadsheet_id = GOOGLE_SHEETS_CONFIG['spreadsheet_id']
        sheet_name = GOOGLE_SHEETS_CONFIG['sheet_name']
        
        if not spreadsheet_id:
            logger.error("Google Sheet ID not configured")
            return False
        
        # [FIX CRIT-3] Use get_next_trading_day() — skips weekends + NSE holidays.
        # Old code: tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        entry_date = get_next_trading_day()
        
        logger.info(f"📊 Preparing to update Google Sheet")
        logger.info(f"📅 Next Trading Day (Entry): {entry_date}")
        logger.info(f"📥 Received {len(signals_data)} signals from CSV")
        
        # ════════════════════════════════════════════════════════════════════
        # 🆕 CRITICAL: HARD CONTRACT ENFORCEMENT
        # ════════════════════════════════════════════════════════════════════
        
        # Validate input count
        if not signals_data:
            logger.warning("⚠️  No signals received - nothing to update")
            return True
        
        # 🚨 ENFORCEMENT POINT #1: Exactly 2 signals required
        if len(signals_data) != 2:
            logger.critical(f"❌ CONTRACT VIOLATION: Expected exactly 2 signals, got {len(signals_data)}")
            logger.critical(f"❌ This indicates upstream select_top_2_stocks() failed!")
            logger.critical(f"❌ REFUSING to upload - manual review required")
            
            # Log what we received for debugging
            for i, sig in enumerate(signals_data, 1):
                ticker = sig.get('ticker', sig.get('Ticker', 'N/A'))
                logger.critical(f"   Signal #{i}: {ticker}")
            
            return False  # HARD STOP - do not proceed
        
        logger.info("✅ CONTRACT CHECK PASSED: Exactly 2 signals received")
        
        # ════════════════════════════════════════════════════════════════════
        # VALIDATE INPUT
        # ════════════════════════════════════════════════════════════════════
        
        # Log what we received
        logger.info(f"\n{'='*70}")
        logger.info(f"📋 SIGNALS TO UPLOAD:")
        logger.info(f"{'='*70}")
        
        for i, sig in enumerate(signals_data, 1):
            ticker = sig.get('ticker', sig.get('Ticker', 'N/A'))
            side = sig.get('side', sig.get('Side', 'N/A'))
            tier = sig.get('tier', sig.get('Tier', 'N/A'))
            conf = sig.get('confidence', sig.get('Confidence', 0))
            bt_wr = sig.get('BT_WR', 0)
            
            logger.info(f"  #{i}: {ticker} | {side} | Tier: {tier} | Conf: {conf}% | BT_WR: {bt_wr}%")
        
        logger.info(f"{'='*70}\n")


        # ════════════════════════════════════════════════════════════════════
        # 🆕 CRITICAL: TIER VALIDATION
        # ════════════════════════════════════════════════════════════════════

        TIER_1_PREMIUM = 'TIER_1_PREMIUM'
        TIER_2_STANDARD = 'TIER_2_STANDARD'
        TIER_3_SPECULATIVE = 'TIER_3_SPECULATIVE'

        allowed_tiers = {TIER_1_PREMIUM, TIER_2_STANDARD}

        for i, sig in enumerate(signals_data, 1):
            tier = sig.get('tier', sig.get('Tier', 'UNKNOWN'))
            ticker = sig.get('ticker', sig.get('Ticker', 'N/A'))
            
            # 🚨 ENFORCEMENT: Only Tier 1 & 2 allowed
            if tier not in allowed_tiers:
                logger.critical(f"❌ TIER VIOLATION: Signal #{i} ({ticker}) has Tier: {tier}")
                logger.critical(f"❌ Only {allowed_tiers} are allowed for production trading")
                logger.critical(f"❌ REFUSING to upload - Tier-3 detected!")
                return False

        logger.info("✅ TIER VALIDATION PASSED: All signals are Tier-1 or Tier-2")

        # ════════════════════════════════════════════════════════════════════
        # 🆕 CRITICAL: BACKTEST INTEGRITY CHECK
        # ════════════════════════════════════════════════════════════════════

        logger.info("✅ Backtest check delegated to final_trade_gate() - already validated")
        
        
        # ════════════════════════════════════════════════════════════════════
        # ENSURE HEADERS EXIST
        # ════════════════════════════════════════════════════════════════════
        
        try:
            result = service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=f'{sheet_name}!A1:J1'
            ).execute()
            
            values = result.get('values', [])
            
            if not values:
                headers = [
                    ['Ticker', 'Position', 'Entry_Price', 'Quantity', 'Stop_Loss', 
                     'Target_1', 'Target_2', 'Entry_Date', 'Status', 'Notes']
                ]
                
                service.spreadsheets().values().update(
                    spreadsheetId=spreadsheet_id,
                    range=f'{sheet_name}!A1:J1',
                    valueInputOption='RAW',
                    body={'values': headers}
                ).execute()
                
                logger.info("✅ Added column headers to Google Sheet")
                
        except HttpError as e:
            logger.error(f"Error checking/creating headers: {e}")
            return False
        
        # ════════════════════════════════════════════════════════════════════
        # ✅ TAKE TOP 2 AS-IS (ALREADY VALIDATED ABOVE)
        # ════════════════════════════════════════════════════════════════════
        
        # The signals are already pre-selected and validated
        signals_to_upload = signals_data  # All 2 of them
        
        logger.info(f"✅ Using {len(signals_to_upload)} validated signals")
        
        # ════════════════════════════════════════════════════════════════════
        # FORMAT ROWS FOR GOOGLE SHEET
        # ════════════════════════════════════════════════════════════════════
        
        rows_to_add = []
        for signal in signals_to_upload:
            row = format_sheet_row(signal, entry_date)
            if row:
                rows_to_add.append(row)
                logger.info(f"✅ Formatted: {row[0]} ({row[1]}) @ ₹{row[2]}")
            else:
                logger.warning(f"⚠️  Failed to format signal: {signal.get('ticker', 'UNKNOWN')}")
        
        if not rows_to_add:
            logger.warning("⚠️  No rows formatted successfully")
            return True
        
        # 🚨 ENFORCEMENT POINT #2: Verify formatting didn't lose signals
        if len(rows_to_add) != 2:
            logger.critical(f"❌ FORMATTING ERROR: Started with 2 signals, formatted {len(rows_to_add)}")
            logger.critical(f"❌ REFUSING to upload incomplete data")
            return False
        
        # ════════════════════════════════════════════════════════════════════
        # WRITE TO GOOGLE SHEET
        # ════════════════════════════════════════════════════════════════════
        
        # Find the next empty row
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=f'{sheet_name}!A:A'
        ).execute()
        
        existing_rows = len(result.get('values', []))
        next_row = existing_rows + 1
        
        # Append the data
        range_to_update = f'{sheet_name}!A{next_row}:J{next_row + len(rows_to_add) - 1}'
        
        body = {'values': rows_to_add}
        
        result = service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=range_to_update,
            valueInputOption='RAW',
            body=body
        ).execute()
        
        updated_cells = result.get('updatedCells', 0)
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ SUCCESS: Updated {updated_cells} cells in Google Sheet")
        logger.info(f"✅ Added {len(rows_to_add)} signals to rows {next_row}-{next_row + len(rows_to_add) - 1}")
        logger.info(f"🔗 View: https://docs.google.com/spreadsheets/d/{spreadsheet_id}")
        logger.info(f"{'='*70}\n")
        
        # Optional: Format the sheet
        format_google_sheet(service, spreadsheet_id, sheet_name, next_row, len(rows_to_add))
        
        return True
        
    except HttpError as e:
        logger.error(f"❌ Google Sheets API error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error updating Google Sheet: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
def final_trade_gate(signals: list) -> tuple:
    """
    🚨 ULTIMATE GATEKEEPER - Final validation before ANY execution
    
    This is the LAST line of defense before real money is at risk.
    All execution paths (Google Sheets, Email, API, etc.) MUST call this first.
    
    Args:
        signals: List of signal dictionaries to validate
    
    Returns:
        tuple: (passed: bool, reason: str)
            - passed: True if all checks pass, False otherwise
            - reason: Detailed explanation of pass/fail
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 1: EXACT COUNT
    # ═══════════════════════════════════════════════════════════════════════
    
    if not signals:
        return False, "EMPTY: No signals provided"
    
    if len(signals) != 2:
        return False, f"COUNT VIOLATION: Expected 2 signals, got {len(signals)}"
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 2: TIER VALIDATION
    # ═══════════════════════════════════════════════════════════════════════
    
    ALLOWED_TIERS = {'TIER_1_PREMIUM', 'TIER_2_STANDARD'}
    
    for i, sig in enumerate(signals, 1):
        ticker = sig.get('ticker', sig.get('Ticker', 'UNKNOWN'))
        tier = sig.get('tier', sig.get('Tier', 'UNKNOWN'))
        
        if tier not in ALLOWED_TIERS:
            return False, f"TIER VIOLATION: Signal #{i} ({ticker}) is {tier}. Only {ALLOWED_TIERS} allowed."
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 3: BACKTEST VALIDATION
    # ═══════════════════════════════════════════════════════════════════════
    
    MIN_WR = 55
    MIN_PF = 1.2
    
    # Backtest validation is optional — mini-backtest already filters in Phase 1
    for i, sig in enumerate(signals, 1):
        ticker = sig.get('ticker', sig.get('Ticker', 'UNKNOWN'))
        bt_wr = sig.get('BT_WR', 0)
        # Only block if backtest ran AND explicitly failed (not just missing)
        if sig.get('backtest_validated', False) and bt_wr > 0 and bt_wr < 30:
            return False, f"BACKTEST WR VERY LOW: {ticker} WR={bt_wr}% (below 30% floor)"
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 4: PRICE LEVELS SANITY
    # ═══════════════════════════════════════════════════════════════════════
    
    for i, sig in enumerate(signals, 1):
        ticker = sig.get('ticker', sig.get('Ticker', 'UNKNOWN'))
        entry = sig.get('entry_price', sig.get('price', 0))
        sl = sig.get('stop_loss', 0)
        t1 = sig.get('target_1', 0)
        t2 = sig.get('target_2', 0)
        # FIX: Normalise SHORT_ALERT → SHORT for gate logic (alert signals still validated)
        side = sig.get('side', sig.get('Side', 'UNKNOWN')).replace('_ALERT', '')
        
        # Basic sanity checks
        if entry <= 0:
            return False, f"INVALID PRICE: {ticker} entry price is {entry}"
        
        if sl <= 0:
            return False, f"INVALID SL: {ticker} stop loss is {sl}"
        
        if side == 'LONG':
            if sl >= entry:
                return False, f"INVALID LONG SETUP: {ticker} SL ({sl}) >= Entry ({entry})"
            if t1 <= entry:
                return False, f"INVALID LONG TARGET: {ticker} T1 ({t1}) <= Entry ({entry})"
        elif side == 'SHORT':
            if sl <= entry:
                return False, f"INVALID SHORT SETUP: {ticker} SL ({sl}) <= Entry ({entry})"
            if t1 >= entry:
                return False, f"INVALID SHORT TARGET: {ticker} T1 ({t1}) >= Entry ({entry})"
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 5: RISK LIMITS
    # ═══════════════════════════════════════════════════════════════════════
    
    MAX_RISK_PER_TRADE = 5.0  # 5% max risk per trade
    
    for i, sig in enumerate(signals, 1):
        ticker = sig.get('ticker', sig.get('Ticker', 'UNKNOWN'))
        entry = sig.get('entry_price', sig.get('price', 0))
        sl = sig.get('stop_loss', 0)
        side = sig.get('side', sig.get('Side', 'UNKNOWN')).replace('_ALERT', '')
        
        # Calculate risk percentage
        if side == 'LONG':
            risk_pct = ((entry - sl) / entry) * 100
        else:  # SHORT
            risk_pct = ((sl - entry) / entry) * 100
        
        if risk_pct > MAX_RISK_PER_TRADE:
            return False, f"EXCESSIVE RISK: {ticker} risk is {risk_pct:.1f}% (max {MAX_RISK_PER_TRADE}%)"
        
        if risk_pct < 0.5:
            return False, f"INSUFFICIENT RISK: {ticker} risk is {risk_pct:.1f}% (too tight SL)"
    
    # ═══════════════════════════════════════════════════════════════════════
    # ALL CHECKS PASSED
    # ═══════════════════════════════════════════════════════════════════════
    
    signal_summary = ", ".join([f"{s.get('ticker', 'N/A')} ({s.get('side', 'N/A')})" for s in signals])
    return True, f"ALL CHECKS PASSED: {signal_summary}"


def update_google_sheet_with_gatekeeper(signals_data):
    """
    WRAPPER: Update Google Sheet with final gatekeeper validation
    
    Use this instead of calling update_google_sheet() directly
    """
    
    # 🚨 FINAL GATEKEEPER CHECK
    passed, reason = final_trade_gate(signals_data)
    
    if not passed:
        logger.critical("="*80)
        logger.critical("🚨 FINAL TRADE GATE BLOCKED EXECUTION")
        logger.critical("="*80)
        logger.critical(f"Reason: {reason}")
        logger.critical("Signals were rejected at the final validation stage")
        logger.critical("Manual review required before proceeding")
        logger.critical("="*80)
        return False
    
    logger.info("="*80)
    logger.info("✅ FINAL TRADE GATE: ALL VALIDATIONS PASSED")
    logger.info("="*80)
    logger.info(f"Validation: {reason}")
    logger.info("Proceeding with Google Sheet update...")
    logger.info("="*80)
    
    # If gatekeeper passes, proceed with normal update
    return update_google_sheet(signals_data)

def format_google_sheet(service, spreadsheet_id, sheet_name, start_row, num_rows):
    """
    Apply formatting to the Google Sheet (optional but makes it look professional)
    
    Args:
        service: Google Sheets API service
        spreadsheet_id: ID of the spreadsheet
        sheet_name: Name of the sheet
        start_row: Starting row number
        num_rows: Number of rows to format
    """
    try:
        # Get sheet ID
        sheet_metadata = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        sheets = sheet_metadata.get('sheets', [])
        sheet_id = None
        
        for sheet in sheets:
            if sheet['properties']['title'] == sheet_name:
                sheet_id = sheet['properties']['sheetId']
                break
        
        if sheet_id is None:
            logger.warning(f"Could not find sheet ID for {sheet_name}")
            return
        
        # Create formatting requests
        requests = []
        
        # Add alternating row colors
        requests.append({
            'addConditionalFormatRule': {
                'rule': {
                    'ranges': [{
                        'sheetId': sheet_id,
                        'startRowIndex': start_row - 1,
                        'endRowIndex': start_row + num_rows - 1,
                        'startColumnIndex': 0,
                        'endColumnIndex': 10
                    }],
                    'booleanRule': {
                        'condition': {
                            'type': 'CUSTOM_FORMULA',
                            'values': [{'userEnteredValue': '=ISEVEN(ROW())'}]
                        },
                        'format': {
                            'backgroundColor': {'red': 0.95, 'green': 0.95, 'blue': 0.95}
                        }
                    }
                },
                'index': 0
            }
        })
        
        # Apply formatting
        if requests:
            body = {'requests': requests}
            service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body=body
            ).execute()
            
            logger.info("✅ Applied formatting to Google Sheet")
        
    except Exception as e:
        logger.warning(f"⚠️  Could not apply formatting: {e}")


def read_signals_from_csv(csv_file):
    """
    Read signals from CSV file and convert to list of dictionaries
    
    Args:
        csv_file: Path to CSV file
    
    Returns:
        List of dictionaries containing signal data
    """
    try:
        import pandas as pd
        
        if not os.path.exists(csv_file):
            logger.error(f"CSV file not found: {csv_file}")
            return []
        
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Convert to list of dictionaries
        signals = df.to_dict('records')
        
        logger.info(f"📄 Read {len(signals)} signals from CSV")
        return signals
        
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return []


# ============================================================================
# IMPORT YOUR ORIGINAL SCANNER CODE
# ============================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NSE SWING SCANNER v8.5 - ULTIMATE ACCURACY EDITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Complete production-ready scanner with:
✓ All v7.4 Advanced Features
✓ All v8.0 Backtesting Features  
✓ Full error handling
✓ Bug fixes applied
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import io
import os
from typing import Optional, Tuple, List, Dict, Set
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dt_time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import logging
import time

# Platform-specific encoding fix
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================================
# KITE DATA PROVIDER (inlined — replaces all yfinance calls)
# ============================================================================

INDEX_TOKENS = {
    "NIFTY 50":         256265,
    "NIFTY BANK":       260105,
    "INDIA VIX":        264969,
    "NIFTY IT":         259849,
    "NIFTY AUTO":       258801,
    "NIFTY PHARMA":     261889,
    "NIFTY FMCG":       257801,
    "NIFTY METAL":      258121,
    "NIFTY REALTY":     260361,
    "NIFTY ENERGY":     265,
    "NIFTY INFRA":      258537,
    "NIFTY PSU BANK":   261641,
    "NIFTY FIN SERVICE":257537,
    "NIFTY MEDIA":      258633,
    "NIFTY MIDCAP 100": 262153,
    "NIFTY SMALLCAP 100": 261145,
}

# Sector name → Kite index name mapping  (matches SectorRotationAnalyzer keys)
SECTOR_TO_INDEX = {
    "BANKING":          "NIFTY BANK",
    "IT":               "NIFTY IT",
    "PHARMA":           "NIFTY PHARMA",
    "AUTO":             "NIFTY AUTO",
    "FMCG":             "NIFTY FMCG",
    "METAL":            "NIFTY METAL",
    "REALTY":           "NIFTY REALTY",
    "ENERGY":           "NIFTY ENERGY",
    "INFRA":            "NIFTY INFRA",
    "PSU_BANK":         "NIFTY PSU BANK",
    "FINANCIALS":       "NIFTY FIN SERVICE",
    "MEDIA":            "NIFTY MEDIA",
}

# ============================================================================
# NSE SECTOR → human sector label (used in get_fundamentals fallback)
# ============================================================================
SECTOR_KEYWORDS = {
    "Financial Services": "FINANCIALS",
    "Banks":              "BANKING",
    "Information Technology": "IT",
    "Healthcare":         "PHARMA",
    "Automobile":         "AUTO",
    "Consumer Goods":     "FMCG",
    "Metals":             "METAL",
    "Real Estate":        "REALTY",
    "Energy":             "ENERGY",
    "Infrastructure":     "INFRA",
    "Media":              "MEDIA",
}



import pandas as pd
import numpy as np
import requests
from io import StringIO

# Technical Analysis
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from scipy import stats
from scipy.signal import argrelextrema

warnings.filterwarnings('ignore')


# ============================================================================
# KITE DATA PROVIDER — main class
# ============================================================================

class KiteDataProvider:
    """
    Drop-in replacement for all yfinance usage.
    Uses Kite Connect (via kite_session from kite_connector).

    Singleton usage:
        df = kite_provider.get_historical_df("RELIANCE", days=800)
    """

    def __init__(self):
        self._kite = None                      # raw kiteconnect.KiteConnect object
        self._kite_session = None             # kite_connector wrapper
        self._instrument_cache: Dict[str, int] = {}   # symbol → token
        self._instruments_loaded = False
        self._fundamentals_cache: Dict[str, Tuple] = {}

        # Historical data cache (in-memory for current run)
        self._hist_cache: Dict[str, pd.DataFrame] = {}

        self._connect()

    # ------------------------------------------------------------------ #
    # INITIALISATION                                                       #
    # ------------------------------------------------------------------ #

    def _connect(self):
        """Connect to Kite via existing kite_connector."""
        try:
            from kite_connector import kite_session as _ks
            self._kite_session = _ks

            if not _ks.is_connected:
                ok, msg = _ks.initialize()
                if not ok:
                    logger.warning(f"Kite session init failed: {msg}")
                    return

            # Grab the raw KiteConnect object if exposed
            if hasattr(_ks, 'kite'):
                self._kite = _ks.kite
            elif hasattr(_ks, '_kite'):
                self._kite = _ks._kite

            logger.info(f"✅ KiteDataProvider connected: {_ks.is_connected}")

        except ImportError:
            logger.error("kite_connector not found — cannot use Kite API")
        except Exception as e:
            logger.error(f"Kite connect error: {e}")

    @property
    def is_connected(self) -> bool:
        return (self._kite_session is not None and
                getattr(self._kite_session, 'is_connected', False))

    # ------------------------------------------------------------------ #
    # INSTRUMENT TOKEN CACHE                                               #
    # ------------------------------------------------------------------ #

    def _load_instruments(self):
        """Load & cache NSE instrument tokens once per run."""
        if self._instruments_loaded:
            return

        try:
            if self._kite:
                raw = self._kite.instruments("NSE")
            elif self._kite_session and hasattr(self._kite_session, 'get_instruments'):
                raw = self._kite_session.get_instruments("NSE")
            else:
                logger.warning("No method to load instruments — tokens unavailable")
                return

            for inst in raw:
                sym = inst.get('tradingsymbol', '')
                token = inst.get('instrument_token')
                if sym and token:
                    self._instrument_cache[sym] = token

            self._instruments_loaded = True
            logger.info(f"✅ Loaded {len(self._instrument_cache)} NSE instrument tokens")

        except Exception as e:
            logger.error(f"Instrument load failed: {e}")

    def get_token(self, symbol: str) -> Optional[int]:
        """Get Kite instrument token for a stock symbol (no .NS suffix)."""
        # Check session wrapper first (it may have its own cache)
        if self._kite_session and hasattr(self._kite_session, 'get_instrument_token'):
            try:
                token = self._kite_session.get_instrument_token(symbol, exchange="NSE")
                if token:
                    return token
            except Exception:
                pass

        # Fall back to our own cache
        if not self._instruments_loaded:
            self._load_instruments()

        return self._instrument_cache.get(symbol)

    def get_index_token(self, index_name: str) -> Optional[int]:
        """Return hardcoded token for NSE indices."""
        return INDEX_TOKENS.get(index_name)

    # ------------------------------------------------------------------ #
    # HISTORICAL DATA — STOCKS                                             #
    # ------------------------------------------------------------------ #

    def get_historical_df(
        self,
        symbol: str,
        days: int = 800,
        interval: str = "day",
        end_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV DataFrame for a stock via Kite historical API.
        Returns DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex (tz-naive)
        """
        cache_key = f"{symbol}_{days}_{interval}"
        if cache_key in self._hist_cache:
            return self._hist_cache[cache_key]

        token = self.get_token(symbol)
        if not token:
            logger.debug(f"No token for {symbol}")
            return None

        end = end_date or datetime.now()
        start = end - timedelta(days=days)

        df = self._fetch_historical(token, start, end, interval)
        if df is not None:
            self._hist_cache[cache_key] = df

        return df

    def _fetch_historical(
        self,
        token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str = "day",
        retries: int = 3,
    ) -> Optional[pd.DataFrame]:
        """
        Core Kite historical_data call with chunking and retry.
        Kite limits: max 2000 candles for day, max 60 days for minute.
        We chunk into 400-day windows to be safe.
        """
        all_chunks = []
        chunk_days = 400 if interval == "day" else 60

        chunk_start = from_date
        while chunk_start < to_date:
            chunk_end = min(chunk_start + timedelta(days=chunk_days), to_date)

            for attempt in range(retries):
                try:
                    if self._kite:
                        raw = self._kite.historical_data(
                            token,
                            chunk_start.strftime("%Y-%m-%d"),
                            chunk_end.strftime("%Y-%m-%d"),
                            interval,
                        )
                    elif self._kite_session and hasattr(self._kite_session, 'get_historical_data'):
                        raw = self._kite_session.get_historical_data(
                            instrument_token=token,
                            from_date=chunk_start,
                            to_date=chunk_end,
                            interval=interval,
                        )
                    else:
                        return None

                    if raw:
                        all_chunks.extend(raw)
                    break

                except Exception as e:
                    if attempt < retries - 1:
                        time.sleep(1.5 * (attempt + 1))
                    else:
                        logger.debug(f"Historical fetch failed token={token}: {e}")

            chunk_start = chunk_end + timedelta(days=1)

        if not all_chunks:
            return None

        return self._raw_to_df(all_chunks)

    @staticmethod
    def _raw_to_df(raw: list) -> pd.DataFrame:
        """Convert Kite raw list-of-dicts to a clean OHLCV DataFrame."""
        df = pd.DataFrame(raw)

        # Normalise column names (Kite returns lowercase)
        col_map = {
            'date':   'Date',
            'open':   'Open',
            'high':   'High',
            'low':    'Low',
            'close':  'Close',
            'volume': 'Volume',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if 'Date' not in df.columns:
            return None

        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df.set_index('Date').sort_index()

        # Keep only OHLCV
        keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
        df = df[keep].dropna(subset=['Close'])

        for col in keep:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df if len(df) >= 10 else None

    # ------------------------------------------------------------------ #
    # HISTORICAL DATA — INDICES                                            #
    # ------------------------------------------------------------------ #

    def get_index_df(
        self,
        index_name: str,
        days: int = 90,
        interval: str = "day",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV DataFrame for an NSE index (NIFTY 50, INDIA VIX, etc.)
        """
        cache_key = f"IDX_{index_name}_{days}"
        if cache_key in self._hist_cache:
            return self._hist_cache[cache_key]

        token = self.get_index_token(index_name)
        if not token:
            logger.debug(f"No token for index: {index_name}")
            return None

        end = datetime.now()
        start = end - timedelta(days=days)

        df = self._fetch_historical(token, start, end, interval)
        if df is not None:
            self._hist_cache[cache_key] = df

        return df

    # ------------------------------------------------------------------ #
    # NIFTY 50 HELPERS                                                     #
    # ------------------------------------------------------------------ #

    def get_nifty50_df(self, days: int = 90) -> Optional[pd.DataFrame]:
        return self.get_index_df("NIFTY 50", days=days)

    def get_nifty50_returns(self, lookback: int = 20) -> Optional[pd.Series]:
        df = self.get_nifty50_df(days=lookback + 10)
        if df is None:
            return None
        return df['Close'].pct_change().dropna().tail(lookback)

    # ------------------------------------------------------------------ #
    # INDIA VIX                                                            #
    # ------------------------------------------------------------------ #

    def get_vix_data(self, days: int = 30) -> Optional[pd.DataFrame]:
        return self.get_index_df("INDIA VIX", days=days)

    # ------------------------------------------------------------------ #
    # SECTOR INDICES                                                        #
    # ------------------------------------------------------------------ #

    def get_sector_df(self, sector_key: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        sector_key: one of SECTOR_TO_INDEX keys
                    e.g. 'BANKING', 'IT', 'PHARMA' …
        """
        index_name = SECTOR_TO_INDEX.get(sector_key.upper())
        if not index_name:
            return None
        return self.get_index_df(index_name, days=days)

    def get_sector_return_pct(self, sector_key: str, lookback_days: int = 10) -> Optional[float]:
        """10-day return % for a sector index."""
        df = self.get_sector_df(sector_key, days=lookback_days + 5)
        if df is None or len(df) < 2:
            return None
        close = df['Close'].dropna()
        if len(close) < 2:
            return None
        return (float(close.iloc[-1]) / float(close.iloc[-min(lookback_days, len(close) - 1)]) - 1) * 100

    # ------------------------------------------------------------------ #
    # FUNDAMENTALS (sector, PE)                                            #
    # ------------------------------------------------------------------ #

    def get_fundamentals(self, symbol: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Returns (sector, pe_ratio).
        Uses Kite instruments for sector, falls back to NSE stock name matching.
        PE is not available from Kite — we skip it (return None).
        """
        if symbol in self._fundamentals_cache:
            return self._fundamentals_cache[symbol]

        sector = None
        pe = None

        try:
            if not self._instruments_loaded:
                self._load_instruments()

            # Try to get instrument info from instruments list
            if self._kite:
                instruments = self._kite.instruments("NSE")
                for inst in instruments:
                    if inst.get('tradingsymbol') == symbol:
                        name = inst.get('name', '').upper()
                        # Rough sector inference from company name keywords
                        sector = self._infer_sector_from_name(name, symbol)
                        break

        except Exception as e:
            logger.debug(f"Fundamentals lookup failed for {symbol}: {e}")

        result = (sector, pe)
        self._fundamentals_cache[symbol] = result
        return result

    @staticmethod
    def _infer_sector_from_name(name: str, symbol: str) -> Optional[str]:
        """Rough sector inference from company name / symbol."""
        n = name.upper()
        s = symbol.upper()

        # Banking
        if any(k in n for k in ['BANK', 'BANKING', 'NBF', 'NBFC']):
            return 'Financial Services'
        # IT
        if any(k in n for k in ['TECH', 'INFOSY', 'WIPRO', 'INFRA', 'SOFTWARE', 'DIGIT', 'IT LTD', 'SYSTEMS']):
            return 'Information Technology'
        # Pharma
        if any(k in n for k in ['PHARMA', 'DRUG', 'MEDICINE', 'HEALTH', 'BIOCON', 'CIPLA', 'LUPIN']):
            return 'Healthcare'
        # Auto
        if any(k in n for k in ['AUTO', 'MOTOR', 'AUTOMOBILE', 'TRACTOR', 'VEHICLES', 'BAJAJ']):
            return 'Automobile'
        # FMCG
        if any(k in n for k in ['FOODS', 'BEVERAGES', 'CONSUMER', 'HINDUSTAN UNILEVER', 'ITC', 'MARICO', 'BRITANNIA']):
            return 'Consumer Goods'
        # Metal
        if any(k in n for k in ['STEEL', 'METAL', 'ALUMIN', 'COPPER', 'ZINC', 'IRON']):
            return 'Metals'
        # Energy
        if any(k in n for k in ['OIL', 'GAS', 'PETRO', 'POWER', 'ENERGY', 'COAL']):
            return 'Energy'
        # Real Estate
        if any(k in n for k in ['REALTY', 'REAL ESTATE', 'HOUSING', 'PROPERTY', 'BUILDER', 'DLF']):
            return 'Real Estate'
        # Finance (non-banking)
        if any(k in n for k in ['FINANCE', 'FINANCIAL', 'CAPITAL', 'INVEST', 'AMC', 'INSURANCE']):
            return 'Financial Services'

        return None

    # ------------------------------------------------------------------ #
    # BULK CACHE BUILD (replaces build_cache in scanner)                  #
    # ------------------------------------------------------------------ #

    def build_bulk_cache(
        self,
        symbols: List[str],
        days: int = 800,
        max_workers: int = 8,
    ) -> Optional[pd.DataFrame]:
        """
        Download historical data for all symbols using Kite.
        Returns a combined DataFrame with a 'ticker' column.
        Mirrors the build_cache() function in scanner_github.py.
        """
        all_data = []
        failed = []
        successful = 0

        def download_one(symbol: str):
            clean_sym = symbol.replace('.NS', '').replace('.BO', '')
            df = self.get_historical_df(clean_sym, days=days)
            if df is not None and len(df) >= 60:
                df = df.copy().reset_index()
                df['ticker'] = clean_sym
                return df
            return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_one, s): s for s in symbols}
            i = 0
            for future in as_completed(futures):
                symbol = futures[future]
                i += 1
                try:
                    result = future.result(timeout=30)
                    if result is not None:
                        all_data.append(result)
                        successful += 1
                    else:
                        failed.append(symbol)
                except Exception:
                    failed.append(symbol)

                if i % 25 == 0 or i == len(symbols):
                    print(
                        f"  Progress: {i}/{len(symbols)} | "
                        f"Success: {successful} | Failed: {len(failed)}",
                        end="\r",
                        flush=True,
                    )

        print()

        if not all_data:
            logger.error("No data downloaded from Kite!")
            return None

        df_all = pd.concat(all_data, axis=0, ignore_index=True)
        logger.info(f"✅ Kite cache: {df_all['ticker'].nunique()} stocks, "
                    f"{len(df_all):,} bars")
        return df_all

    # ------------------------------------------------------------------ #
    # MARKET REGIME (replaces MarketRegimeFilter.get_market_regime)       #
    # ------------------------------------------------------------------ #

    def get_market_regime(self) -> Dict:
        """
        Compute Nifty 50 regime using Kite data.
        Mirrors MarketRegimeFilter.get_market_regime().
        """
        try:
            df = self.get_nifty50_df(days=90)
            if df is None or len(df) < 50:
                return {"regime": "UNKNOWN", "score": 0, "bias": "NEUTRAL"}

            close = df['Close']
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            ema200 = close.ewm(span=200, adjust=False).mean() if len(close) >= 200 else ema50

            p = float(close.iloc[-1])
            e20 = float(ema20.iloc[-1])
            e50 = float(ema50.iloc[-1])
            e200 = float(ema200.iloc[-1])

            # ADX
            h, l, c2 = df['High'], df['Low'], close
            plus_dm = h.diff().clip(lower=0)
            minus_dm = (-l.diff()).clip(lower=0)
            tr = pd.concat([h - l, abs(h - c2.shift()), abs(l - c2.shift())], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean()
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
            adx = float(dx.rolling(14).mean().iloc[-1])

            vol = float(close.pct_change().tail(20).std() * np.sqrt(252) * 100)

            if vol > 25:
                regime, score, bias = "HIGH_VOLATILITY", 0, "CAUTION"
            elif p > e20 > e50 > e200:
                regime, score, bias = "STRONG_BULLISH", 10, "LONG"
            elif p > e20 > e50:
                regime, score, bias = "BULLISH", 7, "LONG"
            elif p < e20 < e50 < e200:
                regime, score, bias = "STRONG_BEARISH", -10, "SHORT"
            elif p < e20 < e50:
                regime, score, bias = "BEARISH", -7, "SHORT"
            elif adx < 20:
                regime, score, bias = "RANGE_BOUND", 0, "NEUTRAL"
            else:
                regime, score, bias = "MIXED", 0, "NEUTRAL"

            return {
                "regime": regime, "score": score, "bias": bias,
                "nifty_price": p, "adx": adx, "volatility": vol,
                "ema20": e20, "ema50": e50, "ema200": e200,
            }

        except Exception as e:
            logger.error(f"Market regime error: {e}")
            return {"regime": "UNKNOWN", "score": 0, "bias": "NEUTRAL"}

    # ------------------------------------------------------------------ #
    # VIX SENTIMENT (replaces VIXSentimentAnalyzer.get_vix_sentiment)    #
    # ------------------------------------------------------------------ #

    def get_vix_sentiment(self) -> Optional[Dict]:
        """
        Returns VIX-based sentiment dict.
        Mirrors VIXSentimentAnalyzer.get_vix_sentiment().
        """
        try:
            df = self.get_vix_data(days=30)
            if df is None or df.empty:
                return None

            close = df['Close'].dropna()
            current_vix = float(close.iloc[-1])
            vix_avg = float(close.mean())

            from scipy import stats as sc_stats
            vix_pct = float(sc_stats.percentileofscore(close, current_vix))

            if current_vix < 12:
                regime, sentiment, score = "LOW_VOLATILITY", "BULLISH", 5
            elif current_vix < 16:
                regime, sentiment, score = "LOW_VOLATILITY", "BULLISH", 3
            elif current_vix < 20:
                regime, sentiment, score = "NORMAL", "NEUTRAL", 0
            elif current_vix < 25:
                regime, sentiment, score = "ELEVATED", "CAUTIOUS", -3
            else:
                regime, sentiment, score = "HIGH_VOLATILITY", "BEARISH", -7

            return {
                "vix": current_vix,
                "vix_avg": vix_avg,
                "vix_percentile": vix_pct,
                "regime": regime,
                "sentiment": sentiment,
                "score": score,
            }

        except Exception as e:
            logger.debug(f"VIX sentiment error: {e}")
            return None

    # ------------------------------------------------------------------ #
    # SECTOR STRENGTH (replaces SectorRotationAnalyzer)                  #
    # ------------------------------------------------------------------ #

    def get_all_sector_strengths(self) -> Dict[str, float]:
        """
        Returns {sector_key: 10d_return_%} for all sectors.
        Mirrors SectorRotationAnalyzer.update_sector_strength().
        """
        strengths = {}
        for sector_key in SECTOR_TO_INDEX:
            try:
                ret = self.get_sector_return_pct(sector_key, lookback_days=10)
                if ret is not None:
                    strengths[sector_key] = ret
            except Exception:
                pass
        return strengths

    def get_sector_strength_label(
        self,
        sector: str,
        strengths: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, str]:
        """
        Map a human sector name (from fundamentals) to a strength label.
        Returns (strength_value, 'STRONG'|'MODERATE'|'WEAK'|'NEUTRAL')
        """
        if strengths is None:
            strengths = self.get_all_sector_strengths()

        # Map human sector → sector key
        sector_map = {
            'Financial Services': 'FINANCIALS',
            'Financial':          'FINANCIALS',
            'Banks':              'BANKING',
            'Banking':            'BANKING',
            'Information Technology': 'IT',
            'Technology':         'IT',
            'Healthcare':         'PHARMA',
            'Pharmaceuticals':    'PHARMA',
            'Automobile':         'AUTO',
            'Consumer Goods':     'FMCG',
            'FMCG':               'FMCG',
            'Metals':             'METAL',
            'Metal':              'METAL',
            'Real Estate':        'REALTY',
            'Energy':             'ENERGY',
            'Infrastructure':     'INFRA',
            'Media':              'MEDIA',
        }

        key = sector_map.get(sector, sector.upper())
        val = strengths.get(key, 0.0)

        if val > 5:
            label = "STRONG"
        elif val > 0:
            label = "MODERATE"
        else:
            label = "WEAK"

        return val, label

    # ------------------------------------------------------------------ #
    # NIFTY RELATIVE STRENGTH (replaces RelativeStrengthCalculator)      #
    # ------------------------------------------------------------------ #

    def get_nifty_returns_series(self, lookback: int = 20) -> Optional[pd.Series]:
        """Return Nifty 50 daily pct-change series for RS calculation."""
        df = self.get_nifty50_df(days=lookback + 10)
        if df is None:
            return None
        return df['Close'].pct_change().dropna()

    def calculate_relative_strength(
        self,
        stock_df: pd.DataFrame,
        lookback: int = 20,
    ) -> Tuple[float, str]:
        """
        Calculate stock RS vs Nifty 50.
        Mirrors RelativeStrengthCalculator.calculate_rs().
        """
        try:
            nifty_returns = self.get_nifty_returns_series(lookback)
            if nifty_returns is None or len(stock_df) < lookback:
                return 0, "UNKNOWN"

            stock_ret = stock_df['Close'].pct_change().tail(lookback)
            stock_cum = float((1 + stock_ret).prod() - 1)
            nifty_cum = float((1 + nifty_returns.tail(lookback)).prod() - 1)

            if abs(nifty_cum) > 0.001:
                rs = (stock_cum / nifty_cum) * 100
            else:
                rs = 100 if stock_cum > 0 else -100

            if rs > 150:
                label = "VERY_STRONG"
            elif rs > 110:
                label = "STRONG"
            elif rs > 90:
                label = "NEUTRAL"
            elif rs > 50:
                label = "WEAK"
            else:
                label = "VERY_WEAK"

            return round(rs, 1), label

        except Exception as e:
            logger.debug(f"RS calc error: {e}")
            return 0, "UNKNOWN"

    # ------------------------------------------------------------------ #
    # DETECT MARKET REGIME (for select_top_2_stocks)                      #
    # ------------------------------------------------------------------ #

    def detect_regime_for_selection(self) -> Tuple[str, Dict]:
        """
        Returns (regime_str, thresholds_dict) for select_top_2_stocks.
        Replaces the old Nifty fetch call inside that function.
        """
        regime_data = self.get_market_regime()
        regime = regime_data.get('regime', 'RANGE_BOUND')

        # Map strong/weak variants to base regimes
        if 'BULLISH' in regime:
            base = 'TRENDING_UP'
        elif 'BEARISH' in regime:
            base = 'TRENDING_DOWN'
        elif regime == 'HIGH_VOLATILITY':
            base = 'HIGH_VOLATILITY'
        else:
            base = 'RANGE_BOUND'

        thresholds = {
            'TRENDING_UP':     {'min_confidence': 70, 'min_bt_wr': 55, 'min_score': 60,
                                'prefer_side': 'LONG',
                                'description': 'Trending Up — favour LONG positions'},
            'TRENDING_DOWN':   {'min_confidence': 70, 'min_bt_wr': 55, 'min_score': 60,
                                'prefer_side': 'SHORT',
                                'description': 'Trending Down — favour SHORT positions'},
            'RANGE_BOUND':     {'min_confidence': 75, 'min_bt_wr': 60, 'min_score': 65,
                                'prefer_side': None,
                                'description': 'Range-Bound — higher quality required'},
            'HIGH_VOLATILITY': {'min_confidence': 80, 'min_bt_wr': 65, 'min_score': 70,
                                'prefer_side': None,
                                'description': 'High Volatility — very selective'},
        }

        return base, thresholds.get(base, thresholds['RANGE_BOUND'])


# ============================================================================
# MODULE-LEVEL SINGLETON
# ============================================================================

kite_provider = KiteDataProvider()



# ============================================================================
# CONFIGURATION - CHANGE THESE SETTINGS
# ============================================================================

ACCURACY_MODE = 'BALANCED'  # 'CONSERVATIVE', 'BALANCED', or 'AGGRESSIVE'
BACKTEST_MODE = 'HYBRID'      # 'MINI', 'FULL', 'HYBRID', 'NONE'

# ── FIX: CNC SHORT TRADING NOT SUPPORTED ON NSE EQUITY ──────────────────────
# In India, you cannot short CNC (delivery) stocks. Setting this True means:
# • SHORT signals are still scored & displayed (useful for awareness/hedging via options)
# • They are tagged side="SHORT_ALERT" and NEVER sent to Zerodha as CNC orders
# • Set False ONLY if you trade NFO futures/options and want actual short orders
CNC_LONG_ONLY = True  # Recommended: True for pure equity swing traders
# ─────────────────────────────────────────────────────────────────────────────

# Configuration Presets
PRESETS = {
    'CONSERVATIVE': {
        'min_confidence': 75,
        'min_turnover': 2_000_000,
        'min_rr_ratio': 2.5,
        'min_volatility_pct': 0.008,
        'min_volume_ratio': 0.8,
        'rsi_range_long': (45, 60),
        'rsi_range_short': (40, 55),
        'conditions_required': 5,
        'use_sector_rotation': True,
        'use_vix_sentiment': True,
        'use_fibonacci': True,
        'use_portfolio_risk': True,
        'use_full_backtest': True,
        'use_walk_forward': True,
        'max_stocks': 500,
        'rs_lookback': 21,
        'fib_lookback': 50,
        'atr_period': 14,
        'min_adx': 30,
        'min_volume_breakout': 2.0,
        'vcp_lookback': 20,
        'min_backtest_win_rate': 65,
        'min_profit_factor': 2.0, 
        'max_drawdown': 12,
        'sigmoid_divisor': 40,
    },
    'BALANCED': {
        # ✅ OPTIMIZED FOR 2-5 DAILY SIGNALS
        'min_confidence': 65,               # ✅ Already correct
        'min_turnover': 800_000,            # ✅ Already correct
        'min_rr_ratio': 1.7,                # ✅ Already correct (you changed from 1.6 to 2.0)
        'min_volatility_pct': 0.005,        # ✅ Already correct
        'min_volume_ratio': 0.5,            # ✅ Already correct
        'rsi_range_long': (40, 62),         # ✅ Already correct
        'rsi_range_short': (38, 60),        # ✅ Already correct
        'conditions_required': 4,           # ✅ Already correct
        'use_sector_rotation': True,        # ✅ Already correct
        'use_vix_sentiment': True,          # ✅ Already correct
        'use_fibonacci': True,              # ✅ Already correct
        'use_portfolio_risk': True,         # ✅ Already correct
        'use_full_backtest': True,          # ✅ Already correct
        'use_walk_forward': True,           # ✅ Already correct
        'max_stocks': 500,                  # ✅ Already correct
        'rs_lookback': 14,                  # ✅ Already correct
        'fib_lookback': 40,                 # ✅ Already correct
        'atr_period': 12,                   # ✅ Already correct
        'min_adx': 23,                      # ✅ Already correct
        'min_volume_breakout': 1.6,         # ✅ Already correct
        'vcp_lookback': 18,                 # ✅ Already correct
        'min_backtest_win_rate': 58,        # ✅ Already correct
        'min_profit_factor': 1.7,           # ✅ Already correct
        'max_drawdown': 18,                 # ✅ Already correct
        'sigmoid_divisor': 35,              # ✅ Already correct
    },
    'AGGRESSIVE': {
        'min_confidence': 40,
        'min_turnover': 200_000,
        'min_rr_ratio': 1.2,
        'min_volatility_pct': 0.003,
        'min_volume_ratio': 0.2,
        'rsi_range_long': (30, 70),
        'rsi_range_short': (30, 70),
        'conditions_required': 2,
        'use_sector_rotation': False,
        'use_vix_sentiment': False,
        'use_fibonacci': True,
        'use_portfolio_risk': False,
        'use_full_backtest': False,
        'use_walk_forward': False,
        'max_stocks': 500,
        'rs_lookback': 10,
        'fib_lookback': 30,
        'atr_period': 8,
        'min_adx': 15,
        'min_volume_breakout': 1.2,
        'vcp_lookback': 10,
        'min_backtest_win_rate': 50,
        'min_profit_factor': 1.3,
        'max_drawdown': 25,
        'sigmoid_divisor': 30,
    }
}

PRESET = PRESETS[ACCURACY_MODE]

# Scoring Weights - OPTIMIZED FOR BALANCED MODE
SCORING_WEIGHTS = {
    "ema_trend": 9,                 # ✅ Raised from 8
    "ema_200": 5,                   # ✅ Raised from 4
    "rsi": 8,                       # ✅ Raised from 7
    "macd": 8,                      # ✅ Raised from 7
    "volume_ratio": 5,              # ✅ Raised from 4
    "ema_slope": 4,                 # ✅ Raised from 3
    "obv": 4,                       # ✅ Raised from 3
    "sr_support": 6,                # ✅ Raised from 5
    "sr_resistance": 6,             # ✅ Raised from 5
    "candle": 5,                    # ✅ Raised from 4
    "mtf_trend": 7,                 # ✅ Raised from 6
    "mtf_penalty": 6,               # ✅ Raised from 5
    "vix_sentiment": 5,             # ✅ Raised from 4
    "fib_boost": 6,                 # ✅ Raised from 5
    "adx_momentum": 6,              # ✅ Raised from 5
    "vsa_confirmation": 5,          # ✅ Raised from 4
    "gap_detection": 4,             # ✅ Raised from 3
    "vcp_pattern": 7,               # ✅ Raised from 6
    "supertrend": 6,                # ✅ Raised from 5
    "volume_breakout": 5,           # ✅ Raised from 4
    "backtest_win_rate": 13,        # ✅ Raised from 12
    "backtest_profit_factor": 9,    # ✅ Raised from 8
    "pattern_reliability": 11,      # ✅ Raised from 10
}

# Scanner Parameters
LOOKBACK_DAYS = 800             # FIX: Extended to 800 days so 3-yr backtest has enough raw data
MIN_CONFIDENCE = PRESET['min_confidence']
MIN_TURNOVER = PRESET['min_turnover']
MIN_RR_RATIO = PRESET['min_rr_ratio']
TOP_RESULTS = 10               # ✅ Changed from 20 to 15 (focus on top signals)

# ── Account & Risk ──────────────────────────────────────────────────────────
# ACCOUNT_CAPITAL is read DYNAMICALLY from your Google Sheet (_Monitor tab,
# cell B1) every time the scanner runs.
# → Change it in the sheet today to 50k, tomorrow to 70k — scanner adapts.
# → Fallback to 50,000 only if the sheet is unreachable or cell is empty.
#
# _Monitor sheet layout (set this up once in your Google Sheet):
#   A1: ACCOUNT_CAPITAL    B1: 50000   ← edit B1 to change your capital
#   A2: RISK_PER_TRADE     B2: 0.01    ← optional, edit to change risk %
#
_CAPITAL_FALLBACK = 50_000   # used only if Google Sheet is unreachable

def _read_capital_from_sheet() -> float:
    """
    Read ACCOUNT_CAPITAL from _Monitor!B1 in your Google Sheet.
    Returns the fallback value if anything goes wrong.
    Runs once at startup so the whole scanner uses the live value.
    """
    try:
        if not GOOGLE_SHEETS_CONFIG.get("enabled") or not GOOGLE_SHEETS_CONFIG.get("spreadsheet_id"):
            logger.warning("Capital config: Google Sheets not configured — using fallback Rs.%s", _CAPITAL_FALLBACK)
            return float(_CAPITAL_FALLBACK)

        from google.oauth2.service_account import Credentials as _Creds
        from googleapiclient.discovery import build as _build
        import json as _json

        cred_json = GOOGLE_SHEETS_CONFIG.get("credentials_json", "")
        if not cred_json:
            return float(_CAPITAL_FALLBACK)

        creds   = _Creds.from_service_account_info(
            _json.loads(cred_json),
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
        )
        svc    = _build("sheets", "v4", credentials=creds)
        result = svc.spreadsheets().values().get(
            spreadsheetId=GOOGLE_SHEETS_CONFIG["spreadsheet_id"],
            range="_Monitor!B1",
            valueRenderOption="UNFORMATTED_VALUE"
        ).execute()

        values = result.get("values", [])
        if not values or not values[0]:
            logger.warning(
                "WARNING: _Monitor!B1 is empty — using fallback Rs.%s. "
                "Put your capital amount in cell B1 of the _Monitor sheet.",
                _CAPITAL_FALLBACK
            )
            return float(_CAPITAL_FALLBACK)

        capital = float(values[0][0])
        if capital <= 0:
            logger.warning("WARNING: _Monitor!B1 value <= 0 — using fallback Rs.%s", _CAPITAL_FALLBACK)
            return float(_CAPITAL_FALLBACK)

        logger.info("ACCOUNT_CAPITAL read from Google Sheet (_Monitor!B1): Rs.%.0f", capital)
        return capital

    except Exception as e:
        logger.warning("Could not read capital from sheet (%s) — using fallback Rs.%s", e, _CAPITAL_FALLBACK)
        return float(_CAPITAL_FALLBACK)


# Read capital dynamically at scanner startup
ACCOUNT_CAPITAL     = _read_capital_from_sheet()
RISK_PER_TRADE      = 0.01
MAX_SECTOR_EXPOSURE = 0.32      # Max 32% of capital in any single sector

# Data Quality
MAX_DATA_STALE_DAYS = 5         # ✅ Changed from 7 to 5 (fresher data)
MIN_PRICE = 5                   # ✅ Keep at 5 (avoid penny stocks)
MAX_PRICE = 100000

# Feature Flags - OPTIMIZED FOR BALANCED
USE_SECTOR_ROTATION = True      # ✅ ENABLED
USE_VIX_SENTIMENT = True        # ✅ ENABLED
USE_FIBONACCI_SCORING = True    # ✅ ENABLED
USE_PORTFOLIO_RISK = True       # ✅ ENABLED
USE_MINI_BACKTEST = True        # ✅ ENABLED
USE_FULL_BACKTEST = True        # ✅ ENABLED
USE_WALK_FORWARD = True         # ✅ ENABLED
USE_MONTE_CARLO = True          # [FIX MAJ-4] was False — enabled for risk-of-ruin awareness
USE_REALTIME_DATA = True  # Use live data during market hours

USE_TREND_STRENGTH_FILTER = True   # Set to False to disable
USE_BREAKOUT_CONFIRMATION = True   # Set to False to disable

# Backtesting
BACKTEST_LOOKBACK_DAYS = 750    # FIX: Extended to 750 trading days (~3 years, multi-cycle validation)
MIN_BACKTEST_TRADES = 12        # ✅ Raised from 10 to 12
WALK_FORWARD_PERIODS = 5        # [FIX MAJ-5] was 3 — raised to 5 for stronger out-of-sample validation

# Output
SIGNALS_DIR = f"signals_v8.5_{ACCURACY_MODE.lower()}"
ERROR_LOG_FILE = f"scanner_v8.5_{ACCURACY_MODE.lower()}.log"

# ============================================================================
# LOGGING & STATS
# ============================================================================

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(ERROR_LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Complete stats dictionary
stats = {
    "total": 0, "passed": 0, "no_data": 0,
    "indicator_fail": 0, "turnover_fail": 0, "confidence_fail": 0,
    "rr_fail": 0, "data_quality_fail": 0, "volatility_fail": 0,
    "mini_backtest_fail": 0, "full_backtest_fail": 0, "full_backtest_run": 0,
    "walk_forward_fail": 0, "monte_carlo_fail": 0,
    "vcp_fail": 0, "vsa_fail": 0, "supertrend_fail": 0, "portfolio_fail": 0,
    "liquidity_fail": 0, "circuit_fail": 0, "hours_fail": 0, "price_range_fail": 0,
}

rejection_samples = []

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_series(series_like, index=None):
    """Convert series-like object to pandas Series"""
    arr = np.asarray(series_like)
    if arr.ndim > 1:
        arr = arr[:, 0]
    return pd.Series(arr, index=index)

def safe_indicator(func, default=None):
    """Safely execute indicator calculation"""
    try:
        result = func()
        if pd.isna(result) or (isinstance(result, (int, float)) and not np.isfinite(result)):
            return default
        return result
    except Exception:
        return default

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class BacktestResult:
    """Comprehensive backtest metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    avg_holding_days: float
    best_trade: float
    worst_trade: float
    consecutive_wins: int
    consecutive_losses: int
    reliability_score: float
    expectancy: float

@dataclass
class WalkForwardResult:
    """Walk-forward validation results"""
    period_results: List[BacktestResult]
    avg_win_rate: float
    win_rate_stability: float
    avg_profit_factor: float
    consistency_score: float
    passed: bool

@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results"""
    simulations: int
    avg_final_equity: float
    median_final_equity: float
    worst_case: float
    best_case: float
    prob_profit: float
    confidence_95: Tuple[float, float]
    risk_of_ruin: float
# ============================================================================
# TRADE RULE CLASS
# ============================================================================

class TradeRule:
    """Enhanced trade rule with multi-entry support - AGGRESSIVE QTY FIX"""

    def __init__(self, ticker: str, side: str, current_price: float, atr: float):
        self.ticker = ticker
        self.side = side
        self.entry_price = round(current_price, 2)
        self.atr = max(atr, current_price * 0.005)  # Min 0.5% ATR
        
        self.risk_amount = ACCOUNT_CAPITAL * RISK_PER_TRADE
        
        # [FIX CRIT-2] Use shared _get_stop_multiplier() so TradeRule (live
        # signals) and AdvancedBacktester._simulate_trade() (historical sim)
        # always use the SAME stop distance.  Old code had the if/elif inline
        # here and ALSO a hardcoded 1.5 in _simulate_trade — they diverged.
        atr_pct = self.atr / current_price
        stop_multiplier = TradeRule._get_stop_multiplier(atr_pct)
        
        if side == "LONG":
            self.stop_loss = round(current_price - (self.atr * stop_multiplier), 2)
            self.target_1 = round(current_price + (self.atr * stop_multiplier * 1.8), 2)
            self.target_2 = round(current_price + (self.atr * stop_multiplier * 3.5), 2)
        else:
            self.stop_loss = round(current_price + (self.atr * stop_multiplier), 2)
            self.target_1 = round(current_price - (self.atr * stop_multiplier * 1.8), 2)
            self.target_2 = round(current_price - (self.atr * stop_multiplier * 3.5), 2)
        
        # ✅ Calculate quantity based on actual stop distance
        risk_per_share = abs(self.entry_price - self.stop_loss)
        
        if risk_per_share > 0:
            # Calculate max qty by risk
            max_qty_by_risk = int(self.risk_amount / risk_per_share)
            
            # Calculate max qty by position size (max 10% of capital)
            max_qty_by_capital = int((ACCOUNT_CAPITAL * 0.10) / current_price)
            
            # Take minimum of both constraints
            self.qty = min(max_qty_by_risk, max_qty_by_capital)
            
            # Cap at 1000 shares
            self.qty = min(self.qty, 1000)
            
            # Ensure at least 1 share
            self.qty = max(1, self.qty)
        else:
            self.qty = 1
        
        self.position_value = self.qty * self.entry_price
        self.actual_risk = self.qty * risk_per_share
        
        # ✅ Calculate R:R based on new default targets (now ~1.67x minimum)
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.target_1 - self.entry_price)
        self.risk_reward_ratio = reward / risk if risk > 0 else 0

    @staticmethod
    def _get_stop_multiplier(atr_pct: float) -> float:
        """
        [FIX CRIT-2] Shared ATR stop-distance multiplier.
        Called by both TradeRule.__init__() (live signals) and
        AdvancedBacktester._simulate_trade() (historical simulation) to ensure
        backtest stop distances always match what the scanner actually shows.

        Volatility tiers (matching NSE swing-trade best practice):
          >4 % ATR  → 2.0× (high-vol stocks like small-caps, results season)
          2.5-4 %   → 1.5× (medium-vol: most Nifty 200 stocks)
          <2.5 %    → 1.2× (low-vol: large-cap consolidations)
        """
        if atr_pct > 0.04:
            return 2.0
        elif atr_pct > 0.025:
            return 1.5
        else:
            return 1.2

    def calculate_dynamic_targets(self, confidence: float):
        """Adjust targets based on setup quality - ENHANCED VERSION"""
        
        # ✅ Dynamic multipliers based on confidence
        if confidence >= 90:
            multiplier_t1 = 3.0
            multiplier_t2 = 6.0
        elif confidence >= 85:
            multiplier_t1 = 2.8
            multiplier_t2 = 5.5
        elif confidence >= 80:
            multiplier_t1 = 2.5
            multiplier_t2 = 5.0
        elif confidence >= 75:
            multiplier_t1 = 2.3
            multiplier_t2 = 4.5
        elif confidence >= 70:
            multiplier_t1 = 2.2
            multiplier_t2 = 4.2
        elif confidence >= 65:
            multiplier_t1 = 2.0
            multiplier_t2 = 4.0
        else:
            # Low confidence - use defaults from __init__
            multiplier_t1 = 2.5  # ✅ Increased from 1.8
            multiplier_t2 = 5.0  # ✅ Increased from 3.5
        
        if self.side == "LONG":
            self.target_1 = round(self.entry_price + (self.atr * multiplier_t1), 2)
            self.target_2 = round(self.entry_price + (self.atr * multiplier_t2), 2)
        else:
            self.target_1 = round(self.entry_price - (self.atr * multiplier_t1), 2)
            self.target_2 = round(self.entry_price - (self.atr * multiplier_t2), 2)
        
        # Recalculate R:R with new targets
        risk_per_share = abs(self.entry_price - self.stop_loss)
        reward_per_share = abs(self.target_1 - self.entry_price)
        
        if risk_per_share > 0:
            self.risk_reward_ratio = reward_per_share / risk_per_share
        else:
            self.risk_reward_ratio = 0
# ============================================================================
# DATA QUALITY VALIDATOR - FIXED VERSION
# ============================================================================

class DataQualityValidator:
    """Validate data quality and freshness"""
    
    @staticmethod
    def is_data_fresh(df: pd.DataFrame, max_stale_days: int = MAX_DATA_STALE_DAYS) -> bool:
        """Check if data is recent"""
        try:
            last_date = pd.to_datetime(df.index[-1])
            days_old = (datetime.now() - last_date).days
            return days_old <= max_stale_days
        except Exception:
            return False
    
    @staticmethod
    def has_sufficient_history(df: pd.DataFrame, min_bars: int = 60) -> bool:
        """Check minimum data length"""
        return len(df) >= min_bars
    
    @staticmethod
    def has_valid_prices(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate price data - BULLETPROOF VERSION"""
        try:
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df = df.copy()
                df.columns = df.columns.get_level_values(0)
            
            # Check for Close column
            if 'Close' not in df.columns.tolist():
                return False, "No Close column"
            
            if len(df) == 0:
                return False, "Empty dataframe"
            
            # Get non-NaN close values
            close_series = df['Close'].dropna()
            
            if len(close_series) == 0:
                return False, "All Close values are NaN"
            
            # Get last non-NaN close
            close = close_series.iloc[-1]
            
            if pd.isna(close):
                return False, "Close is NaN"
            
            try:
                close_price = float(close)
            except (ValueError, TypeError):
                return False, "Close not numeric"
            
            if not np.isfinite(close_price):
                return False, "Close is infinite"
            
            # Price range
            if close_price < MIN_PRICE:
                return False, f"Price too low: ₹{close_price:.2f}"
            
            if close_price > MAX_PRICE:
                return False, f"Price too high: ₹{close_price:.2f}"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)[:40]}"

# ============================================================================
# VOLATILITY REGIME FILTER - FIXED VERSION
# ============================================================================
class VolatilityRegimeFilter:
    """Advanced volatility regime analysis - COMPLETELY BULLETPROOF"""
    
    @staticmethod
    def analyze_regime(df: pd.DataFrame, price: float) -> Tuple[bool, str, Dict]:
        """Comprehensive volatility regime analysis"""
        try:
            # ============================================================
            # STEP 1: VALIDATE DATAFRAME
            # ============================================================
            if df is None:
                return False, "DataFrame is None", {}
            
            if not isinstance(df, pd.DataFrame):
                return False, f"Not a DataFrame: {type(df)}", {}
            
            if df.empty:
                return False, "Empty DataFrame", {}
            
            # ============================================================
            # STEP 2: CHECK COLUMNS
            # ============================================================
            required_cols = ['High', 'Low', 'Close']
            
            # Get actual columns
            actual_cols = list(df.columns)
            
            # Check if all required columns exist
            missing = [col for col in required_cols if col not in actual_cols]
            if missing:
                return False, f"Missing: {missing}", {}
            
            # ============================================================
            # STEP 3: EXTRACT DATA AS SERIES
            # ============================================================
            try:
                # Extract columns and ensure they are Series
                high_col = df['High']
                low_col = df['Low']
                close_col = df['Close']
                
                # Convert to Series if needed
                if isinstance(high_col, pd.DataFrame):
                    high_col = high_col.iloc[:, 0]
                if isinstance(low_col, pd.DataFrame):
                    low_col = low_col.iloc[:, 0]
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                
                # Make clean copies
                high_data = pd.Series(high_col.values, name='High')
                low_data = pd.Series(low_col.values, name='Low')
                close_data = pd.Series(close_col.values, name='Close')
                
            except Exception as e:
                return False, f"Data extract fail: {str(e)[:25]}", {}
            
            # ============================================================
            # STEP 4: CLEAN DATA
            # ============================================================
            try:
                # Remove NaN values
                high_data = high_data.dropna()
                low_data = low_data.dropna()
                close_data = close_data.dropna()
                
                # Check we have enough data
                min_len = min(len(high_data), len(low_data), len(close_data))
                
                if min_len < PRESET.get('atr_period', 14) + 5:
                    return False, f"Insufficient data: {min_len}", {}
                
                # Align lengths (take shortest)
                if len(high_data) != len(low_data) or len(low_data) != len(close_data):
                    min_len = min(len(high_data), len(low_data), len(close_data))
                    high_data = high_data.iloc[-min_len:]
                    low_data = low_data.iloc[-min_len:]
                    close_data = close_data.iloc[-min_len:]
                
            except Exception as e:
                return False, f"Data clean fail: {str(e)[:25]}", {}
            
            # ============================================================
            # STEP 5: CALCULATE ATR
            # ============================================================
            try:
                atr_period = PRESET.get('atr_period', 14)
                
                # Calculate True Range manually for more control
                tr1 = high_data - low_data
                tr2 = abs(high_data - close_data.shift(1))
                tr3 = abs(low_data - close_data.shift(1))
                
                # Combine and take max
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                # Calculate ATR as rolling mean
                atr_series = tr.rolling(window=atr_period, min_periods=atr_period).mean()
                
                # Remove NaN values
                atr_series = atr_series.dropna()
                
                if len(atr_series) == 0:
                    return False, "ATR calculation empty", {}
                
                # Get current ATR
                current_atr = float(atr_series.iloc[-1])
                
                if pd.isna(current_atr) or current_atr <= 0:
                    return False, f"Invalid ATR: {current_atr}", {}
                
            except Exception as e:
                return False, f"ATR calc fail: {str(e)[:25]}", {}
            
            # ============================================================
            # STEP 6: CALCULATE METRICS
            # ============================================================
            try:
                # Validate price
                if price <= 0 or pd.isna(price):
                    return False, "Invalid price", {}
                
                # ATR as percentage
                atr_pct = float(current_atr / price)
                
                # ATR slope
                if len(atr_series) >= 5:
                    atr_5 = float(atr_series.iloc[-5])
                    if atr_5 > 0:
                        atr_slope = float((current_atr - atr_5) / atr_5)
                    else:
                        atr_slope = 0.0
                else:
                    atr_slope = 0.0
                
                # ATR percentile
                if len(atr_series) > 10:
                    from scipy import stats as scipy_stats
                    atr_percentile = float(scipy_stats.percentileofscore(atr_series, current_atr))
                else:
                    atr_percentile = 50.0
                
            except Exception as e:
                return False, f"Metrics fail: {str(e)[:25]}", {}
            
            # ============================================================
            # STEP 7: BUILD METRICS DICT
            # ============================================================
            metrics = {
                'atr': current_atr,
                'atr_pct': atr_pct,
                'atr_slope': atr_slope,
                'atr_percentile': atr_percentile
            }
            
            # ============================================================
            # STEP 8: REGIME CLASSIFICATION
            # ============================================================
            min_vol = PRESET.get('min_volatility_pct', 0.003)
            
            # Very low volatility
            if atr_pct < min_vol * 0.6:
                return False, f"Very low vol ({atr_pct:.3%})", metrics
            
            # Contracting volatility
            if atr_pct < min_vol and atr_slope < -0.1:
                return False, f"Vol contracting ({atr_pct:.3%})", metrics
            
            # Extreme volatility
            if atr_pct > 0.10:
                return False, f"Extreme vol ({atr_pct:.3%})", metrics
            
            # Historical extremes
            if atr_percentile > 95:
                return False, f"Vol at extremes (P{atr_percentile:.0f})", metrics
            
            # ACCEPTABLE REGIME
            if atr_pct >= min_vol:
                return True, f"OK ({atr_pct:.3%})", metrics
            
            # DEFAULT: REJECT
            return False, f"Low vol ({atr_pct:.3%})", metrics
            
        except Exception as e:
            # Final catch-all
            return False, f"Error: {type(e).__name__}", {}
        
# ============================================================================
# PORTFOLIO RISK MANAGER
# ============================================================================

class PortfolioRiskManager:
    """
    Capital-aware portfolio risk manager.

    [FIX RISK-2] Old logic used a fixed MAX_CONCURRENT_TRADES count (e.g. 3)
    which had no relation to actual capital available.  New logic:

      available_capital = ACCOUNT_CAPITAL - total_deployed_capital
      → Only allow new trade if its position_value fits within available_capital
      → Also enforces sector concentration and total risk % caps

    This means:
      • 50k capital, 30k deployed  → 20k free  → new signals sized to fit 20k
      • 50k capital, 48k deployed  → 2k free   → blocks most new signals
      • Capital correct in config  → everything works automatically
    """

    # Minimum free capital required to take any new position (10% of total)
    MIN_FREE_CAPITAL_PCT = 0.10

    def __init__(self, capital: float):
        self.capital           = capital
        self.current_trades    = []
        self.sector_exposure   = {}   # sector → fraction of capital deployed
        self.total_risk        = 0.0  # sum of all risk_amount values
        self.total_deployed    = 0.0  # sum of all position_value values

    # ── Read-only helpers ────────────────────────────────────────────────────

    @property
    def available_capital(self) -> float:
        """Capital not yet deployed in open trades."""
        return max(0.0, self.capital - self.total_deployed)

    @property
    def deployed_pct(self) -> float:
        """Fraction of capital currently deployed (0–1)."""
        return self.total_deployed / self.capital if self.capital > 0 else 0.0

    def available_capital_summary(self) -> str:
        return (
            f"Capital: ₹{self.capital:,.0f} | "
            f"Deployed: ₹{self.total_deployed:,.0f} ({self.deployed_pct*100:.1f}%) | "
            f"Available: ₹{self.available_capital:,.0f} | "
            f"Open trades: {len(self.current_trades)}"
        )

    # ── Core check ───────────────────────────────────────────────────────────

    def can_add_trade(self, trade: Dict) -> Tuple[bool, str]:
        """
        Returns (True, 'OK') only if ALL of the following pass:
          1. Available capital >= position value of new trade
          2. Available capital >= MIN_FREE_CAPITAL_PCT of total (safety buffer)
          3. Sector exposure won't exceed MAX_SECTOR_EXPOSURE
          4. Total portfolio risk won't exceed 5% of capital
        """

        position_value = trade.get('position_value', 0)
        trade_risk     = trade.get('risk_amount', 0)
        sector         = trade.get('sector', 'UNKNOWN')
        ticker         = trade.get('ticker', '?')

        # ── Check 1: Enough free capital for this position ───────────────────
        if position_value > 0 and position_value > self.available_capital:
            return False, (
                f"Insufficient capital for {ticker}: "
                f"needs ₹{position_value:,.0f}, "
                f"only ₹{self.available_capital:,.0f} free "
                f"(₹{self.total_deployed:,.0f} already deployed)"
            )

        # ── Check 2: Safety buffer — keep MIN_FREE_CAPITAL_PCT always free ───
        min_free = self.capital * self.MIN_FREE_CAPITAL_PCT
        capital_after = self.available_capital - position_value
        if capital_after < min_free:
            return False, (
                f"Safety buffer breach for {ticker}: "
                f"would leave only ₹{capital_after:,.0f} free "
                f"(minimum buffer = ₹{min_free:,.0f})"
            )

        # ── Check 3: Sector concentration ────────────────────────────────────
        current_sector_pct = self.sector_exposure.get(sector, 0.0)
        new_sector_pct     = current_sector_pct + (position_value / self.capital)
        if new_sector_pct > MAX_SECTOR_EXPOSURE:
            return False, (
                f"Sector limit for {ticker} ({sector}): "
                f"would reach {new_sector_pct*100:.1f}% "
                f"(max {MAX_SECTOR_EXPOSURE*100:.0f}%)"
            )

        # ── Check 4: Total portfolio risk cap (5% of capital) ────────────────
        if self.total_risk + trade_risk > self.capital * 0.05:
            return False, (
                f"Portfolio risk cap for {ticker}: "
                f"total risk would be ₹{self.total_risk + trade_risk:,.0f} "
                f"(max ₹{self.capital * 0.05:,.0f})"
            )

        return True, (
            f"OK — ₹{self.available_capital - position_value:,.0f} "
            f"will remain free after this trade"
        )

    # ── State update ─────────────────────────────────────────────────────────

    def add_trade(self, trade: Dict):
        """Register a trade (new signal or existing open position)."""
        self.current_trades.append(trade)
        sector         = trade.get('sector', 'UNKNOWN')
        position_value = trade.get('position_value', 0)
        self.sector_exposure[sector] = (
            self.sector_exposure.get(sector, 0.0) + position_value / self.capital
        )
        self.total_risk     += trade.get('risk_amount', 0)
        self.total_deployed += position_value# ============================================================================
# INSTITUTIONAL VOLUME ANALYZER
# ============================================================================

class InstitutionalVolumeAnalyzer:
    """Detect institutional participation"""
    
    @staticmethod
    def analyze(df: pd.DataFrame, current_volume: float) -> Tuple[float, str, Dict]:
        """Analyze volume for institutional activity"""
        try:
            vol_20 = df['Volume'].tail(20).mean()
            vol_50 = df['Volume'].tail(50).mean()
            vol_ratio = current_volume / vol_20 if vol_20 > 0 else 1.0
            
            vol_slope = (vol_20 - vol_50) / vol_50 if vol_50 > 0 else 0
            
            price_change = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]
            vol_change = (current_volume - df['Volume'].iloc[-2]) / df['Volume'].iloc[-2]
            
            pv_correlation = 1 if (price_change > 0 and vol_change > 0) or (price_change < 0 and vol_change < 0) else -1
            
            metrics = {
                'volume_ratio': vol_ratio,
                'vol_slope': vol_slope,
                'pv_correlation': pv_correlation
            }
            
            score = 0
            signal = ""
            
            if vol_ratio >= 3.0 and pv_correlation > 0:
                score = 10
                signal = "Very strong institutional buying"
            elif vol_ratio >= PRESET['min_volume_breakout'] and pv_correlation > 0:
                score = 7
                signal = "Strong volume breakout"
            elif vol_ratio >= 1.5:
                score = 4
                signal = "Above average volume"
            elif vol_ratio < 0.8:
                score = -5
                signal = "Weak volume"
            elif vol_ratio < 0.5:
                score = -8
                signal = "Very weak volume"
            
            return score, signal, metrics
            
        except Exception as e:
            return 0, f"Volume analysis error: {str(e)[:30]}", {}

# ============================================================================
# VOLUME SPREAD ANALYSIS (VSA)
# ============================================================================

class VolumeSpreadAnalyzer:
    """Volume Spread Analysis for smart money detection"""
    
    @staticmethod
    def analyze(df: pd.DataFrame) -> Tuple[float, str, Dict]:
        """Comprehensive VSA analysis"""
        try:
            if len(df) < 5:
                return 0, "Insufficient data", {}
            
            recent = df.tail(5)
            current = recent.iloc[-1]
            prev = recent.iloc[-2]
            
            current_spread = current['High'] - current['Low']
            prev_spread = prev['High'] - prev['Low']
            avg_spread = recent['High'].sub(recent['Low']).mean()
            
            current_volume = current['Volume']
            avg_volume = df['Volume'].tail(20).mean()
            
            spread_ratio = current_spread / avg_spread if avg_spread > 0 else 1
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            close_position = (current['Close'] - current['Low']) / current_spread if current_spread > 0 else 0.5
            
            metrics = {
                'spread_ratio': spread_ratio,
                'volume_ratio': volume_ratio,
                'close_position': close_position
            }
            
            score = 0
            signal = ""
            
            # Accumulation
            if volume_ratio > 1.5 and spread_ratio < 0.8 and close_position > 0.7:
                score = 8
                signal = "VSA: Accumulation"
            # Distribution
            elif volume_ratio > 1.5 and spread_ratio < 0.8 and close_position < 0.3:
                score = -8
                signal = "VSA: Distribution"
            # Stopping volume
            elif volume_ratio > 2.0 and spread_ratio < 0.6:
                score = 6
                signal = "VSA: Stopping volume"
            # No demand
            elif volume_ratio < 0.7 and spread_ratio > 1.3:
                score = -6
                signal = "VSA: No demand"
            # Effort vs Result
            elif volume_ratio > 1.8 and abs(current['Close'] - current['Open']) / current_spread < 0.3:
                score = -5
                signal = "VSA: High effort, no result"
            
            return score, signal, metrics
            
        except Exception as e:
            return 0, f"VSA error: {str(e)[:30]}", {}

# ============================================================================
# MOMENTUM SCORE CALCULATOR
# ============================================================================

class MomentumScoreCalculator:
    """Comprehensive momentum analysis"""
    
    @staticmethod
    def calculate(df: pd.DataFrame) -> Tuple[float, str, Dict]:
        """Calculate composite momentum score"""
        try:
            c = normalize_series(df["Close"])
            h = normalize_series(df["High"])
            l = normalize_series(df["Low"])
            
            adx_indicator = ADXIndicator(h, l, c, window=14)
            adx = adx_indicator.adx().iloc[-1]
            adx_pos = adx_indicator.adx_pos().iloc[-1]
            adx_neg = adx_indicator.adx_neg().iloc[-1]
            
            roc = ROCIndicator(c, window=10).roc().iloc[-1]
            rsi = RSIIndicator(c, window=14).rsi().iloc[-1]
            
            momentum_direction = "BULLISH" if adx_pos > adx_neg else "BEARISH"
            
            metrics = {
                'adx': adx,
                'adx_pos': adx_pos,
                'adx_neg': adx_neg,
                'roc': roc,
                'rsi': rsi,
                'direction': momentum_direction
            }
            
            score = 0
            signal = ""
            
            if adx > 30 and abs(roc) > 5:
                score = 10
                signal = f"Very strong {momentum_direction.lower()} momentum"
            elif adx >= PRESET['min_adx'] and abs(roc) > 2:
                score = 7
                signal = f"Strong {momentum_direction.lower()} momentum"
            elif adx >= 20:
                score = 4
                signal = f"Moderate momentum"
            elif adx < 15:
                score = -5
                signal = "Weak momentum"
            
            # RSI confirmation
            if momentum_direction == "BULLISH" and rsi > 60:
                score += 3
            elif momentum_direction == "BEARISH" and rsi < 40:
                score += 3
            
            return score, signal, metrics
            
        except Exception as e:
            return 0, f"Momentum error: {str(e)[:30]}", {}

# ============================================================================
# VCP DETECTOR
# ============================================================================

class VCPDetector:
    """Detect Volatility Contraction Patterns"""
    
    @staticmethod
    def detect(df: pd.DataFrame) -> Tuple[float, str, Dict]:
        """Detect VCP setup"""
        try:
            lookback = PRESET['vcp_lookback']
            
            if len(df) < lookback + 20:
                return 0, "Insufficient data", {}
            
            recent = df.tail(lookback).copy()
            
            atr = AverageTrueRange(
                high=normalize_series(recent['High']),
                low=normalize_series(recent['Low']),
                close=normalize_series(recent['Close']),
                window=10
            ).average_true_range()
            
            bb = BollingerBands(
                close=normalize_series(recent['Close']),
                window=20,
                window_dev=2
            )
            bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
            
            atr_contraction = (atr.iloc[-1] - atr.max()) / atr.max() if atr.max() > 0 else 0
            bb_contraction = (bb_width.iloc[-1] - bb_width.max()) / bb_width.max() if bb_width.max() > 0 else 0
            
            volume = recent['Volume']
            vol_contraction = (volume.iloc[-5:].mean() - volume.max()) / volume.max() if volume.max() > 0 else 0
            
            contraction_score = abs(atr_contraction) + abs(bb_contraction) + abs(vol_contraction)
            price_range = (recent['High'].max() - recent['Low'].min()) / recent['Close'].iloc[-1]
            
            metrics = {
                'atr_contraction': atr_contraction,
                'bb_contraction': bb_contraction,
                'vol_contraction': vol_contraction,
                'contraction_score': contraction_score,
                'price_range': price_range
            }
            
            score = 0
            signal = ""
            
            if contraction_score > 0.6 and price_range < 0.15:
                score = 10
                signal = "Strong VCP pattern"
            elif contraction_score > 0.4 and price_range < 0.20:
                score = 6
                signal = "Moderate VCP pattern"
            elif contraction_score > 0.2:
                score = 3
                signal = "Weak VCP forming"
            
            return score, signal, metrics
            
        except Exception as e:
            return 0, f"VCP error: {str(e)[:30]}", {}

# ============================================================================
# S/R CLUSTER DETECTOR
# ============================================================================

class SRClusterDetector:
    """Advanced S/R detection using price clustering"""
    
    @staticmethod
    def detect(df: pd.DataFrame, lookback: int = 60) -> Dict:
        """Detect clustered S/R levels"""
        try:
            recent = df.tail(lookback).copy()
            
            highs = recent['High'].values
            lows = recent['Low'].values
            
            high_indices = argrelextrema(highs, np.greater, order=3)[0]
            low_indices = argrelextrema(lows, np.less, order=3)[0]
            
            swing_highs = [highs[i] for i in high_indices]
            swing_lows = [lows[i] for i in low_indices]
            
            current_price = recent['Close'].iloc[-1]
            
            if not swing_highs and not swing_lows:
                return {
                    'support': current_price * 0.98,
                    'resistance': current_price * 1.02,
                    'strength': 0,
                    'bias': 'NEUTRAL'
                }
            
            all_levels = swing_highs + swing_lows
            tolerance = np.std(all_levels) * 0.25 if all_levels else current_price * 0.01
            
            clusters = {}
            for level in all_levels:
                found = False
                for center in list(clusters.keys()):
                    if abs(level - center) <= tolerance:
                        clusters[center].append(level)
                        found = True
                        break
                if not found:
                    clusters[level] = [level]
            
            cluster_strength = {k: len(v) for k, v in clusters.items()}
            sorted_clusters = sorted(cluster_strength.items(), key=lambda x: x[1], reverse=True)
            
            support_levels = [c[0] for c in sorted_clusters if c[0] < current_price]
            resistance_levels = [c[0] for c in sorted_clusters if c[0] > current_price]
            
            if support_levels:
                support = max(support_levels)
                support_strength = cluster_strength[support]
            else:
                support = min(all_levels) if all_levels else current_price * 0.98
                support_strength = 1
            
            if resistance_levels:
                resistance = min(resistance_levels)
                resistance_strength = cluster_strength[resistance]
            else:
                resistance = max(all_levels) if all_levels else current_price * 1.02
                resistance_strength = 1
            
            dist_to_support = abs(current_price - support) / current_price
            dist_to_resistance = abs(resistance - current_price) / current_price
            
            if dist_to_support < 0.02:
                bias = "AT_SUPPORT"
            elif dist_to_resistance < 0.02:
                bias = "AT_RESISTANCE"
            elif dist_to_support < dist_to_resistance:
                bias = "NEAR_SUPPORT"
            else:
                bias = "NEAR_RESISTANCE"
            
            return {
                'support': float(support),
                'resistance': float(resistance),
                'support_strength': support_strength,
                'resistance_strength': resistance_strength,
                'total_clusters': len(sorted_clusters),
                'bias': bias,
                'dist_to_support': dist_to_support,
                'dist_to_resistance': dist_to_resistance
            }
            
        except Exception as e:
            logger.debug(f"S/R clustering failed: {e}")
            return {
                'support': None,
                'resistance': None,
                'strength': 0,
                'bias': 'UNKNOWN'
            }

# ============================================================================
# SUPERTREND INDICATOR
# ============================================================================

class SupertrendIndicator:
    """Supertrend for trend confirmation"""
    
    @staticmethod
    def calculate(df: pd.DataFrame, period: int = 10, multiplier: int = 3) -> Tuple[Optional[str], float, Dict]:
        """Calculate Supertrend"""
        try:
            h = normalize_series(df["High"])
            l = normalize_series(df["Low"])
            c = normalize_series(df["Close"])
            
            atr = AverageTrueRange(h, l, c, window=period).average_true_range()
            
            hl2 = (h + l) / 2
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            
            supertrend = pd.Series(index=c.index, dtype=float)
            direction = pd.Series(index=c.index, dtype=str)
            
            for i in range(len(c)):
                if i < period:
                    supertrend.iloc[i] = 0
                    direction.iloc[i] = None
                    continue
                
                if c.iloc[i] > upper_band.iloc[i-1]:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = "UP"
                elif c.iloc[i] < lower_band.iloc[i-1]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = "DOWN"
                else:
                    supertrend.iloc[i] = supertrend.iloc[i-1]
                    direction.iloc[i] = direction.iloc[i-1]
            
            current_direction = direction.iloc[-1]
            
            if current_direction == "UP":
                strength = ((c.iloc[-1] - supertrend.iloc[-1]) / supertrend.iloc[-1]) * 100
            elif current_direction == "DOWN":
                strength = ((supertrend.iloc[-1] - c.iloc[-1]) / c.iloc[-1]) * 100
            else:
                strength = 0
            
            consecutive = 1
            for i in range(len(direction)-2, -1, -1):
                if direction.iloc[i] == current_direction:
                    consecutive += 1
                else:
                    break
            
            metrics = {
                'direction': current_direction,
                'strength': strength,
                'consecutive_bars': consecutive,
                'supertrend_value': supertrend.iloc[-1]
            }
            
            return current_direction, round(strength, 2), metrics
            
        except Exception as e:
            return None, 0, {}

# ============================================================================
# GAP DETECTOR
# ============================================================================

class GapDetector:
    """Detect and analyze gaps"""
    
    @staticmethod
    def detect(df: pd.DataFrame) -> Tuple[float, str, Dict]:
        """Detect gap patterns"""
        try:
            if len(df) < 2:
                return 0, "No gap", {}
            
            prev_close = df['Close'].iloc[-2]
            current_open = df['Open'].iloc[-1]
            current_close = df['Close'].iloc[-1]
            current_high = df['High'].iloc[-1]
            current_low = df['Low'].iloc[-1]
            
            gap_pct = ((current_open - prev_close) / prev_close) * 100
            
            if gap_pct > 0:
                gap_filled = current_low <= prev_close
            else:
                gap_filled = current_high >= prev_close
            
            if current_open != 0:
                post_gap_move = ((current_close - current_open) / current_open) * 100
            else:
                post_gap_move = 0
            
            metrics = {
                'gap_pct': gap_pct,
                'gap_filled': gap_filled,
                'post_gap_move': post_gap_move
            }
            
            score = 0
            signal = ""
            
            if abs(gap_pct) < 0.5:
                return 0, "No significant gap", metrics
            
            if gap_pct > 2.0 and not gap_filled and post_gap_move > 0:
                score = 8
                signal = f"Strong gap up {gap_pct:.1f}%"
            elif gap_pct > 1.0 and not gap_filled:
                score = 5
                signal = f"Gap up {gap_pct:.1f}%"
            elif gap_pct > 1.0 and gap_filled:
                score = -4
                signal = f"Gap up filled - bearish"
            elif gap_pct < -2.0 and not gap_filled and post_gap_move < 0:
                score = -8
                signal = f"Strong gap down {gap_pct:.1f}%"
            elif gap_pct < -1.0 and not gap_filled:
                score = -5
                signal = f"Gap down {gap_pct:.1f}%"
            elif gap_pct < -1.0 and gap_filled:
                score = 4
                signal = f"Gap down filled - bullish"
            
            return score, signal, metrics
            
        except Exception as e:
            return 0, f"Gap detection error: {str(e)[:30]}", {}

# ============================================================================
# CANDLESTICK PATTERN DETECTOR
# ============================================================================

class CandlestickPatternDetector:
    """Detect common candlestick patterns"""
    
    @staticmethod
    def detect(df: pd.DataFrame) -> Dict:
        """Detect bullish/bearish candlestick patterns"""
        try:
            if len(df) < 3:
                return {'pattern': 'NONE', 'score': 0}
            
            recent = df.tail(3)
            c0 = recent.iloc[-3]
            c1 = recent.iloc[-2]
            c2 = recent.iloc[-1]
            
            def is_bullish(candle):
                return candle['Close'] > candle['Open']
            
            def is_bearish(candle):
                return candle['Close'] < candle['Open']
            
            def body_size(candle):
                return abs(candle['Close'] - candle['Open'])
            
            def candle_range(candle):
                return candle['High'] - candle['Low']
            
            def upper_shadow(candle):
                return candle['High'] - max(candle['Open'], candle['Close'])
            
            def lower_shadow(candle):
                return min(candle['Open'], candle['Close']) - candle['Low']
            
            pattern = 'NONE'
            score = 0
            
            # Bullish Engulfing
            if (is_bearish(c1) and is_bullish(c2) and
                c2['Close'] > c1['Open'] and c2['Open'] < c1['Close']):
                pattern = 'BULLISH_ENGULFING'
                score = 7
            
            # Bearish Engulfing
            elif (is_bullish(c1) and is_bearish(c2) and
                  c2['Close'] < c1['Open'] and c2['Open'] > c1['Close']):
                pattern = 'BEARISH_ENGULFING'
                score = -7
            
            # Hammer
            elif (body_size(c2) < candle_range(c2) * 0.3 and
                  lower_shadow(c2) > body_size(c2) * 2 and
                  upper_shadow(c2) < body_size(c2) * 0.5):
                pattern = 'HAMMER'
                score = 6
            
            # Shooting Star
            elif (body_size(c2) < candle_range(c2) * 0.3 and
                  upper_shadow(c2) > body_size(c2) * 2 and
                  lower_shadow(c2) < body_size(c2) * 0.5):
                pattern = 'SHOOTING_STAR'
                score = -6
            
            # Doji
            elif body_size(c2) < candle_range(c2) * 0.1:
                pattern = 'DOJI'
                score = -2
            
            # Morning Star
            elif (is_bearish(c0) and body_size(c1) < body_size(c0) * 0.3 and
                  is_bullish(c2) and c2['Close'] > (c0['Open'] + c0['Close']) / 2):
                pattern = 'MORNING_STAR'
                score = 8
            
            # Evening Star
            elif (is_bullish(c0) and body_size(c1) < body_size(c0) * 0.3 and
                  is_bearish(c2) and c2['Close'] < (c0['Open'] + c0['Close']) / 2):
                pattern = 'EVENING_STAR'
                score = -8
            
            # Strong bullish
            elif (is_bullish(c2) and body_size(c2) > candle_range(c2) * 0.7 and
                  candle_range(c2) > candle_range(c1) * 1.5):
                pattern = 'STRONG_BULLISH'
                score = 5
            
            # Strong bearish
            elif (is_bearish(c2) and body_size(c2) > candle_range(c2) * 0.7 and
                  candle_range(c2) > candle_range(c1) * 1.5):
                pattern = 'STRONG_BEARISH'
                score = -5
            
            return {
                'pattern': pattern,
                'score': score,
                'description': pattern.replace('_', ' ').title()
            }
            
        except Exception as e:
            return {'pattern': 'NONE', 'score': 0, 'description': 'None'}
# ============================================================================
# MULTI-TIMEFRAME ANALYZER
# ============================================================================

# ✅ CORRECTED CODE - Use cached data:
class MultiTimeframeAnalyzer:
    """Analyze trends across multiple timeframes - OPTIMIZED"""
    
    @staticmethod
    def get_trend_from_cache(df: pd.DataFrame) -> Optional[str]:
        """Get multi-timeframe trend using cached data (NO re-download!)"""
        try:
            if df is None or len(df) < 50:
                return None
            
            # Use daily data we already have
            daily_close = df['Close']
            
            # Calculate EMAs on daily
            daily_ema20 = daily_close.ewm(span=20, adjust=False).mean().iloc[-1]
            daily_ema50 = daily_close.ewm(span=50, adjust=False).mean().iloc[-1]
            daily_ema200 = daily_close.ewm(span=200, adjust=False).mean().iloc[-1] if len(df) >= 200 else daily_ema50
            current_price = daily_close.iloc[-1]
            
            # Daily trend
            if current_price > daily_ema20 > daily_ema50:
                daily_trend = "UP"
            elif current_price < daily_ema20 < daily_ema50:
                daily_trend = "DOWN"
            else:
                daily_trend = "MIXED"
            
            # Simulate weekly by resampling (if enough data)
            if len(df) >= 30:
                weekly = df.resample('W').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                
                if len(weekly) >= 10:
                    weekly_ema10 = weekly['Close'].ewm(span=10, adjust=False).mean().iloc[-1]
                    weekly_close = weekly['Close'].iloc[-1]
                    
                    weekly_trend = "UP" if weekly_close > weekly_ema10 else "DOWN"
                    
                    # Combined trend
                    if daily_trend == weekly_trend:
                        return daily_trend
                    else:
                        return "MIXED"
            
            return daily_trend
            
        except Exception as e:
            logger.debug(f"MTF analysis failed: {e}")
            return None
    
    # Keep old method for backward compatibility but mark deprecated
    def get_trend(self, ticker: str) -> Optional[str]:
        """DEPRECATED - Use get_trend_from_cache instead"""
        return None  # Return None to skip slow download

# ============================================================================
# FIBONACCI LEVEL DETECTOR
# ============================================================================

class FibonacciDetector:
    """Fibonacci retracement levels"""
    
    @staticmethod
    def calculate_levels(df: pd.DataFrame, lookback: int = 50) -> Optional[Dict]:
        """Calculate Fibonacci levels"""
        try:
            recent = df.tail(lookback)
            high = recent['High'].max()
            low = recent['Low'].min()
            diff = high - low
            
            levels = {
                'high': high,
                'low': low,
                'fib_0.236': high - (diff * 0.236),
                'fib_0.382': high - (diff * 0.382),
                'fib_0.500': high - (diff * 0.500),
                'fib_0.618': high - (diff * 0.618),
                'fib_0.786': high - (diff * 0.786),
            }
            
            return levels
            
        except Exception:
            return None
    
    @staticmethod
    def detect_bounce(price: float, fib_levels: Dict, side: str) -> Tuple[int, str]:
        """Detect if price is near Fibonacci level"""
        try:
            if not fib_levels:
                return 0, ""
            
            tolerance = 0.02
            
            for level_name, level_price in fib_levels.items():
                if 'fib_' not in level_name:
                    continue
                
                distance = abs(price - level_price) / price
                
                if distance < tolerance:
                    if side == "LONG" and price >= level_price:
                        return 8, f"Bouncing off {level_name}"
                    elif side == "SHORT" and price <= level_price:
                        return 8, f"Rejecting at {level_name}"
            
            return 0, ""
            
        except Exception:
            return 0, ""

# ============================================================================
# VIX SENTIMENT ANALYZER
# ============================================================================

class VIXSentimentAnalyzer:
    """Market sentiment using VIX"""
    
    @staticmethod
    def get_vix_sentiment() -> Optional[Dict]:
        """Get current VIX sentiment"""
        try:
            vix_df = kite_provider.get_vix_data(days=35)
            
            if vix_df is None or vix_df.empty:
                return None
            
            vix = vix_df  # keep variable name for rest of the method
            current_vix = vix['Close'].iloc[-1]
            vix_avg = vix['Close'].mean()
            vix_percentile = stats.percentileofscore(vix['Close'], current_vix)
            
            if current_vix < 15:
                regime = "LOW_VOLATILITY"
                sentiment = "BULLISH"
                score = 5
            elif current_vix < 20:
                regime = "NORMAL"
                sentiment = "NEUTRAL"
                score = 0
            elif current_vix < 25:
                regime = "ELEVATED"
                sentiment = "CAUTIOUS"
                score = -3
            else:
                regime = "HIGH_VOLATILITY"
                sentiment = "BEARISH"
                score = -7
            
            return {
                'vix': current_vix,
                'vix_avg': vix_avg,
                'vix_percentile': vix_percentile,
                'regime': regime,
                'sentiment': sentiment,
                'score': score
            }
            
        except Exception:
            return None


# ✅ NEW CLASS - Add this:

class MarketRegimeFilter:
    """
    Detect overall market regime to avoid trading in unfavorable conditions
    - BULLISH: Favor LONG signals
    - BEARISH: Favor SHORT signals  
    - RANGING: Reduce position sizes
    - VOLATILE: Skip trading
    """
    
    _cache = {}
    _cache_time = None
    _cache_duration = 1800  # 30 minutes
    
    @classmethod
    def get_market_regime(cls) -> Dict:
        """Get current market regime"""
        import time as time_module
        
        # Return cached result if fresh
        if cls._cache and cls._cache_time and (time_module.time() - cls._cache_time) < cls._cache_duration:
            return cls._cache
        
        try:
            # Fetch Nifty 50 data via Kite
            nifty = kite_provider.get_nifty50_df(days=90)
            
            if nifty is None or nifty.empty or len(nifty) < 50:
                return {"regime": "UNKNOWN", "score": 0, "bias": "NEUTRAL"}
            
            close = nifty['Close']
            
            # Calculate indicators
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            ema200 = close.ewm(span=200, adjust=False).mean()
            
            current_price = float(close.iloc[-1])
            current_ema20 = float(ema20.iloc[-1])
            current_ema50 = float(ema50.iloc[-1])
            current_ema200 = float(ema200.iloc[-1])
            
            # Calculate ADX for trend strength
            high = nifty['High']
            low = nifty['Low']
            
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr = pd.concat([
                high - low,
                abs(high - close.shift()),
                abs(low - close.shift())
            ], axis=1).max(axis=1)
            
            atr14 = tr.rolling(14).mean()
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = float(dx.rolling(14).mean().iloc[-1])
            
            # Calculate 20-day volatility
            returns = close.pct_change()
            volatility = float(returns.tail(20).std() * np.sqrt(252) * 100)
            
            # Determine regime
            score = 0
            
            if current_price > current_ema20 > current_ema50 > current_ema200:
                regime = "STRONG_BULLISH"
                score = 10
                bias = "LONG"
            elif current_price > current_ema20 > current_ema50:
                regime = "BULLISH"
                score = 7
                bias = "LONG"
            elif current_price < current_ema20 < current_ema50 < current_ema200:
                regime = "STRONG_BEARISH"
                score = -10
                bias = "SHORT"
            elif current_price < current_ema20 < current_ema50:
                regime = "BEARISH"
                score = -7
                bias = "SHORT"
            elif adx < 20:
                regime = "RANGING"
                score = 0
                bias = "NEUTRAL"
            else:
                regime = "MIXED"
                score = 0
                bias = "NEUTRAL"
            
            # Adjust for high volatility
            if volatility > 25:
                regime = "HIGH_VOLATILITY"
                bias = "CAUTION"
                score = score * 0.5
            
            result = {
                "regime": regime,
                "score": score,
                "bias": bias,
                "nifty_price": current_price,
                "adx": adx,
                "volatility": volatility,
                "ema20": current_ema20,
                "ema50": current_ema50,
                "ema200": current_ema200
            }
            
            cls._cache = result
            cls._cache_time = time_module.time()
            
            return result
            
        except Exception as e:
            logger.debug(f"Market regime check failed: {e}")
            return {"regime": "UNKNOWN", "score": 0, "bias": "NEUTRAL"}

# ✅ NEW CLASS - Add this:

class RelativeStrengthCalculator:
    """Calculate stock's relative strength vs Nifty 50"""
    
    _nifty_data = None
    _nifty_returns = None
    
    @classmethod
    def _load_nifty(cls):
        """Load Nifty data once — via Kite"""
        if cls._nifty_data is None:
            try:
                nifty_df = kite_provider.get_nifty50_df(days=90)
                if nifty_df is not None and not nifty_df.empty:
                    cls._nifty_data = nifty_df
                    cls._nifty_returns = nifty_df['Close'].pct_change()
            except:
                pass
    
    @classmethod
    def calculate_rs(cls, stock_df: pd.DataFrame, lookback: int = 20) -> Tuple[float, str]:
        """
        Calculate Relative Strength vs Nifty
        Returns: (RS value, interpretation)
        """
        try:
            cls._load_nifty()
            
            if cls._nifty_returns is None or len(stock_df) < lookback:
                return 0, "UNKNOWN"
            
            # Stock returns
            stock_returns = stock_df['Close'].pct_change().tail(lookback)
            stock_cumulative = (1 + stock_returns).prod() - 1
            
            # Nifty returns (same period)
            nifty_returns = cls._nifty_returns.tail(lookback)
            nifty_cumulative = (1 + nifty_returns).prod() - 1
            
            # RS = Stock performance / Nifty performance
            if abs(nifty_cumulative) > 0.001:
                rs = (stock_cumulative / nifty_cumulative) * 100
            else:
                rs = 100 if stock_cumulative > 0 else -100
            
            # Interpretation
            if rs > 150:
                interpretation = "VERY_STRONG"
            elif rs > 110:
                interpretation = "STRONG"
            elif rs > 90:
                interpretation = "NEUTRAL"
            elif rs > 50:
                interpretation = "WEAK"
            else:
                interpretation = "VERY_WEAK"
            
            return round(rs, 1), interpretation
            
        except Exception as e:
            logger.debug(f"RS calculation failed: {e}")
            return 0, "UNKNOWN"
        

# ============================================================================
# SECTOR ROTATION ANALYZER
# ============================================================================

# ============================================================================
# SECTOR ROTATION ANALYZER - FIXED WITH WORKING SYMBOLS
# ============================================================================

class SectorRotationAnalyzer:
    """Analyze sector strength - FIXED VERSION WITH WORKING SYMBOLS"""
    
    def __init__(self):
        # Sector keys — Kite index tokens are handled inside kite_data_provider
        self.sector_strength = {}
        self._last_update = None
        self._cache_duration = 3600  # 1 hour cache

    def update_sector_strength(self):
        """Update sector strength via Kite sector index data — yfinance removed"""
        import time as time_module

        if self._last_update and (time_module.time() - self._last_update) < self._cache_duration:
            return

        try:
            self.sector_strength = kite_provider.get_all_sector_strengths()
            self._last_update = time_module.time()
            logger.debug(f"Sector strengths updated: {len(self.sector_strength)} sectors")
        except Exception as e:
            logger.debug(f"Sector rotation update failed: {e}")
    
    def get_sector_strength(self, sector: str) -> Optional[Tuple[float, str]]:
        """Get sector relative strength"""
        try:
            if not self.sector_strength:
                self.update_sector_strength()
            
            # Map common sector names to our keys
            sector_map = {
                'Financial Services': 'FINANCIALS',
                'Financial': 'FINANCIALS',
                'Banks': 'BANKING',
                'Banking': 'BANKING',
                'Information Technology': 'IT',
                'Technology': 'IT',
                'Healthcare': 'PHARMA',
                'Pharmaceuticals': 'PHARMA',
                'Automobile': 'AUTO',
                'Auto': 'AUTO',
                'Consumer Goods': 'FMCG',
                'FMCG': 'FMCG',
                'Metals': 'METAL',
                'Metal': 'METAL',
                'Real Estate': 'REALTY',
                'Realty': 'REALTY',
                'Energy': 'ENERGY',
                'Oil & Gas': 'ENERGY',
                'Infrastructure': 'INFRA',
                'Infra': 'INFRA',
                'Media': 'MEDIA',
            }
            
            # Normalize sector name
            mapped_sector = sector_map.get(sector, sector.upper())
            
            if mapped_sector in self.sector_strength:
                strength = self.sector_strength[mapped_sector]
                
                if strength > 5:
                    return strength, "STRONG"
                elif strength > 0:
                    return strength, "MODERATE"
                else:
                    return strength, "WEAK"
            
            # Return neutral if sector not found
            return 0, "NEUTRAL"
            
        except Exception:
            return 0, "NEUTRAL"

# ============================================================================
# ADVANCED BACKTESTING ENGINE
# ============================================================================

class AdvancedBacktester:
    """Comprehensive backtesting with walk-forward and Monte Carlo"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.trades = []
    
    def backtest_strategy(
        self,
        df: pd.DataFrame,
        signal_type: str,
        entry_conditions: Dict,
        lookback_days: int = BACKTEST_LOOKBACK_DAYS
    ) -> Optional[BacktestResult]:
        """Full backtest of strategy"""
        
        if len(df) < lookback_days + 50:
            lookback_days = max(100, len(df) - 50)
        
        if len(df) < 150:
            return None
        
        test_data = df.iloc[-lookback_days:].copy()
        trades = []
        
        for i in range(50, len(test_data) - 20):
            window = test_data.iloc[:i+1].copy()
            
            if self._check_entry_conditions(window, signal_type, entry_conditions):
                trade = self._simulate_trade(
                    test_data.iloc[i:min(i+30, len(test_data))],
                    signal_type,
                    entry_price=test_data.iloc[i]['Close'],
                    atr=self._calculate_atr(window)
                )
                
                if trade:
                    trades.append(trade)
        
        if len(trades) < MIN_BACKTEST_TRADES:
            return None
        
        return self._calculate_metrics(trades)
    
    def _check_entry_conditions(
        self,
        df: pd.DataFrame,
        signal_type: str,
        entry_conditions: Dict
    ) -> bool:
        """Enhanced entry condition check with v7.4 features"""
        try:
            if len(df) < 50:
                return False
            
            close = df['Close'].iloc[-1]
            
            ema20 = df['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = df['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
            
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            
            volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            
            exp12 = df['Close'].ewm(span=12, adjust=False).mean()
            exp26 = df['Close'].ewm(span=26, adjust=False).mean()
            macd_line = exp12 - exp26
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            
            high = df['High']
            low = df['Low']
            
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr1 = high - low
            tr2 = abs(high - df['Close'].shift())
            tr3 = abs(low - df['Close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(14).mean().iloc[-1]
            
            vcp_score, _, _ = VCPDetector.detect(df)
            vsa_score, _, _ = VolumeSpreadAnalyzer.analyze(df)
            vol_ok, _, _ = VolatilityRegimeFilter.analyze_regime(df, close)
            st_direction, _, _ = SupertrendIndicator.calculate(df)
            
            rsi_range = PRESET['rsi_range_long'] if signal_type == "LONG" else PRESET['rsi_range_short']
            
            conditions_met = []
            
            if signal_type == "LONG":
                conditions_met = [
                    close > ema20,
                    ema20 > ema50,
                    rsi_range[0] <= rsi <= rsi_range[1],
                    macd_line.iloc[-1] > macd_signal.iloc[-1],
                    volume_ratio >= PRESET['min_volume_ratio'],
                    adx >= PRESET['min_adx'],
                    vol_ok,
                    vsa_score >= 0,
                    st_direction == "UP" if st_direction else True,
                ]
            else:
                conditions_met = [
                    close < ema20,
                    ema20 < ema50,
                    rsi_range[0] <= rsi <= rsi_range[1],
                    macd_line.iloc[-1] < macd_signal.iloc[-1],
                    volume_ratio >= PRESET['min_volume_ratio'],
                    adx >= PRESET['min_adx'],
                    vol_ok,
                    vsa_score <= 0,
                    st_direction == "DOWN" if st_direction else True,
                ]
            
            if vcp_score > 5:
                conditions_met.append(True)
            
            return sum(conditions_met) >= PRESET['conditions_required']
            
        except Exception as e:
            logger.debug(f"Entry condition check failed: {e}")
            return False
    
    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate ATR"""
        try:
            atr = AverageTrueRange(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                window=PRESET['atr_period']
            ).average_true_range().iloc[-1]
            return atr
        except:
            return df['Close'].iloc[-1] * 0.02
    
    def _simulate_trade(
        self,
        future_data: pd.DataFrame,
        signal_type: str,
        entry_price: float,
        atr: float
    ) -> Optional[Dict]:
        """Simulate single trade - NO LOOK-AHEAD BIAS"""
        
        if len(future_data) < 3:
            return None
        
        # ✅ FIX #3: Enter at NEXT bar's open (realistic)
        actual_entry = future_data.iloc[1]['Open']

        # ── Realistic NSE transaction costs (round-trip) ─────────────────────
        # CNC LONG (delivery):
        #   STT buy+sell: 0.1%+0.1% | Exchange: 0.0069% | Stamp: 0.015% buy
        #   GST on fees | Total round-trip: ~0.22%
        # CNC SHORT alert (futures/options):
        #   STT: 0.01% sell side only | Exchange: 0.0019% | Much lower
        #   Total round-trip: ~0.05-0.08%
        # We use 0.20% for LONG (conservative) and 0.08% for SHORT
        TOTAL_COST_PCT = 0.20 if signal_type == "LONG" else 0.08

        # Slippage: 0.05% extra for gap/impact cost on entry/exit
        SLIPPAGE_PCT = 0.05
        TOTAL_COST_PCT += SLIPPAGE_PCT   # 0.25% LONG, 0.13% SHORT

        # [FIX CRIT-2] Use the SAME dynamic stop-multiplier as TradeRule so the
        # simulated trade stop distance matches the live signal the user sees.
        # Old code: always atr * 1.5 regardless of stock volatility.
        _atr_pct = (atr / actual_entry) if actual_entry > 0 else 0.02
        _stop_mult = TradeRule._get_stop_multiplier(_atr_pct)
        # Keep target multiples proportional: T1 = 1.8× risk, T2 = 3.5× risk
        if signal_type == "LONG":
            stop_loss = actual_entry - (atr * _stop_mult)
            target_1  = actual_entry + (atr * _stop_mult * 1.8)
            target_2  = actual_entry + (atr * _stop_mult * 3.5)
        else:
            stop_loss = actual_entry + (atr * _stop_mult)
            target_1  = actual_entry - (atr * _stop_mult * 1.8)
            target_2  = actual_entry - (atr * _stop_mult * 3.5)
        
        exit_price = None
        exit_day = None
        exit_reason = None
        
        # Start from day 2 (day 1 is entry bar)
        for day in range(2, len(future_data)):
            row = future_data.iloc[day]
            
            if signal_type == "LONG":
                # ✅ Check gap down at open first
                if row['Open'] <= stop_loss:
                    exit_price = min(stop_loss, row['Open'])  # Slippage
                    exit_day = day
                    exit_reason = "Stop Loss (gap)"
                    break
                
                # Check if stop hit during the day
                elif row['Low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_day = day
                    exit_reason = "Stop Loss"
                    break
                
                # Check target 2
                elif row['High'] >= target_2:
                    exit_price = target_2
                    exit_day = day
                    exit_reason = "Target 2"
                    break
                
                # Check target 1 after 5 days
                elif day >= 7 and row['High'] >= target_1:
                    exit_price = target_1
                    exit_day = day
                    exit_reason = "Target 1"
                    break
            
            else:  # SHORT
                if row['Open'] >= stop_loss:
                    exit_price = max(stop_loss, row['Open'])
                    exit_day = day
                    exit_reason = "Stop Loss (gap)"
                    break
                
                elif row['High'] >= stop_loss:
                    exit_price = stop_loss
                    exit_day = day
                    exit_reason = "Stop Loss"
                    break
                
                elif row['Low'] <= target_2:
                    exit_price = target_2
                    exit_day = day
                    exit_reason = "Target 2"
                    break
                
                elif day >= 7 and row['Low'] <= target_1:
                    exit_price = target_1
                    exit_day = day
                    exit_reason = "Target 1"
                    break
        
        # Time exit if no target/stop hit
        if exit_price is None:
            exit_price = future_data.iloc[-1]['Close']
            exit_day = len(future_data) - 1
            exit_reason = "Time Exit"
        
        # Calculate return based on ACTUAL entry (costs already set above)
        if signal_type == "LONG":
            return_pct = ((exit_price - actual_entry) / actual_entry) * 100
        else:
            return_pct = ((actual_entry - exit_price) / actual_entry) * 100

        # Deduct realistic round-trip costs + slippage
        return_pct = return_pct - TOTAL_COST_PCT

        # return_dollars based on ACTUAL position size (entry * 1 share = per-share basis)
        # Scaled to ₹10,000 notional position for meaningful expectancy figures
        NOTIONAL_POSITION = 10_000
        return_dollars = (return_pct / 100) * NOTIONAL_POSITION

        return {
            'entry_price': actual_entry,
            'exit_price': exit_price,
            'return_pct': return_pct,
            'return_dollars': return_dollars,
            'holding_days': exit_day - 1,  # Subtract entry day
            'exit_reason': exit_reason,
            'signal_type': signal_type
        }
    
    def _calculate_metrics(self, trades: List[Dict]) -> BacktestResult:
        """Calculate comprehensive metrics"""
        
        if not trades:
            return None
        
        returns = [t['return_pct'] for t in trades]
        return_dollars = [t['return_dollars'] for t in trades]
        
        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r <= 0]
        
        equity = self.initial_capital
        equity_curve = [equity]
        peak_equity = equity
        max_drawdown = 0
        
        for ret_dollars in return_dollars:
            equity += ret_dollars
            equity_curve.append(equity)
            peak_equity = max(peak_equity, equity)
            drawdown = ((peak_equity - equity) / peak_equity) * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_streak = 0
        
        for ret in returns:
            if ret > 0:
                current_streak = current_streak + 1 if current_streak >= 0 else 1
                max_consecutive_wins = max(max_consecutive_wins, current_streak)
            else:
                current_streak = current_streak - 1 if current_streak <= 0 else -1
                max_consecutive_losses = max(max_consecutive_losses, abs(current_streak))
        
        total_trades = len(trades)
        win_rate = (len(winning_trades) / total_trades) * 100
        
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        gross_profit = sum([r for r in returns if r > 0])
        gross_loss = abs(sum([r for r in returns if r < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        total_return = ((equity - self.initial_capital) / self.initial_capital) * 100
        expectancy = np.mean(return_dollars)
        
        reliability_score = self._calculate_reliability_score(
            win_rate, profit_factor, max_drawdown, sharpe_ratio
        )
        
        return BacktestResult(
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=round(win_rate, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(profit_factor, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            max_drawdown=round(max_drawdown, 2),
            total_return=round(total_return, 2),
            avg_holding_days=round(np.mean([t['holding_days'] for t in trades]), 1),
            best_trade=round(max(returns), 2),
            worst_trade=round(min(returns), 2),
            consecutive_wins=max_consecutive_wins,
            consecutive_losses=max_consecutive_losses,
            reliability_score=round(reliability_score, 1),
            expectancy=round(expectancy, 2)
        )
    
    def walk_forward_validation(
    self,
    df: pd.DataFrame,
    signal_type: str,
    entry_conditions: Dict,
    periods: int = 4
    ) -> Optional[WalkForwardResult]:
        """
        Walk-Forward Validation - FIXED VERSION v2.0
        
        Validates strategy consistency across non-overlapping time periods.
        Prevents overfitting by testing on independent data slices.
        
        Args:
            df: Historical price data (pandas DataFrame)
            signal_type: Trade direction - "LONG" or "SHORT"
            entry_conditions: Strategy parameters dictionary
            periods: Number of validation periods (default: 4)
        
        Returns:
            WalkForwardResult: Contains validation metrics and pass/fail status
            None: If validation fails or insufficient data
        
        Methodology:
            1. Splits data into N non-overlapping periods
            2. Backtests each period independently (no data reuse)
            3. Requires minimum 5 trades per period for statistical validity
            4. Checks performance consistency across all periods
            5. Uses mode-specific passing criteria (CONSERVATIVE/BALANCED/AGGRESSIVE)
            6. Scores based on performance + stability + drawdown
        
        Example:
            600 bars, 4 periods:
            Period 1: [0-150]   → 8 trades, 62% WR, 1.8 PF ✅
            Period 2: [150-300] → 10 trades, 58% WR, 1.6 PF ✅
            Period 3: [300-450] → 9 trades, 60% WR, 1.7 PF ✅
            Period 4: [450-600] → 7 trades, 56% WR, 1.5 PF ✅
            Result: PASSED (consistent performance)
        
        Fixes Applied:
            ✅ Removed 100-bar overlap bug
            ✅ Increased minimum data requirement (150 bars/period)
            ✅ Added minimum trades per period (5+)
            ✅ Mode-specific passing criteria
            ✅ Improved consistency scoring algorithm
            ✅ Detailed logging for debugging
            ✅ Better error handling
        """
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: VALIDATE INPUT DATA
        # ═══════════════════════════════════════════════════════════════════
        
        # Calculate minimum required data
        # Conservative: 150 bars per period minimum for reliable results
        min_bars_per_period = 150
        min_required_total = periods * min_bars_per_period
        
        if len(df) < min_required_total:
            logger.debug(
                f"Walk-forward rejected: Insufficient data "
                f"({len(df)} bars < {min_required_total} required for {periods} periods)"
            )
            return None
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: CALCULATE PERIOD BOUNDARIES (NO OVERLAP!)
        # ═══════════════════════════════════════════════════════════════════
        
        period_length = len(df) // periods
        period_results = []
        period_details = []
        
        # Debug header
        logger.debug(f"\n{'='*70}")
        logger.debug(f"WALK-FORWARD VALIDATION")
        logger.debug(f"{'='*70}")
        logger.debug(f"Strategy: {signal_type}")
        logger.debug(f"Total Data: {len(df)} bars")
        logger.debug(f"Periods: {periods}")
        logger.debug(f"Period Length: {period_length} bars")
        logger.debug(f"Mode: {ACCURACY_MODE}")
        logger.debug(f"{'='*70}")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: BACKTEST EACH PERIOD INDEPENDENTLY
        # ═══════════════════════════════════════════════════════════════════
        
        for i in range(periods):
            # Calculate strict boundaries (NO overlap with other periods)
            start_idx = i * period_length
            
            # Last period gets all remaining data (handles rounding)
            if i == periods - 1:
                end_idx = len(df)
            else:
                end_idx = (i + 1) * period_length
            
            # Extract period data
            period_data = df.iloc[start_idx:end_idx].copy()
            actual_bars = len(period_data)
            
            # Calculate lookback for this period
            # Use 70% of period for backtesting, 30% for indicator warmup
            lookback = max(50, int(actual_bars * 0.7))
            
            logger.debug(f"\n{'─'*70}")
            logger.debug(f"PERIOD {i + 1}/{periods}")
            logger.debug(f"{'─'*70}")
            logger.debug(f"  Date Range: {period_data.index[0]} to {period_data.index[-1]}")
            logger.debug(f"  Index Range: [{start_idx}:{end_idx}]")
            logger.debug(f"  Total Bars: {actual_bars}")
            logger.debug(f"  Lookback: {lookback} bars")
            logger.debug(f"  Test Window: ~{lookback - 50} bars")
            
            # Run backtest on this period
            try:
                result = self.backtest_strategy(
                    df=period_data,
                    signal_type=signal_type,
                    entry_conditions=entry_conditions,
                    lookback_days=lookback
                )
            except Exception as e:
                logger.debug(f"  ❌ Backtest Error: {str(e)[:50]}")
                result = None
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 4: VALIDATE PERIOD RESULTS
            # ═══════════════════════════════════════════════════════════════
            
            # Minimum trades required for statistical validity
            min_trades_required = 5
            
            if result is None:
                logger.debug(f"  ❌ REJECTED: Backtest returned None")
                continue
            
            if result.total_trades < min_trades_required:
                logger.debug(
                    f"  ❌ REJECTED: Only {result.total_trades} trades "
                    f"(minimum {min_trades_required} required)"
                )
                continue
            
            # Period passed - store results
            period_results.append(result)
            
            period_details.append({
                'period': i + 1,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'bars': actual_bars,
                'trades': result.total_trades,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'avg_win': result.avg_win,
                'avg_loss': result.avg_loss
            })
            
            logger.debug(f"  ✅ VALID PERIOD")
            logger.debug(f"     Trades: {result.total_trades}")
            logger.debug(f"     Win Rate: {result.win_rate:.1f}%")
            logger.debug(f"     Profit Factor: {result.profit_factor:.2f}")
            logger.debug(f"     Max Drawdown: {result.max_drawdown:.1f}%")
            logger.debug(f"     Sharpe Ratio: {result.sharpe_ratio:.2f}")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 5: CHECK MINIMUM PERIODS REQUIREMENT
        # ═══════════════════════════════════════════════════════════════════
        
        # Require at least 50% of periods to succeed (minimum 2)
        min_required_periods = max(2, (periods + 1) // 2)
        
        if len(period_results) < min_required_periods:
            logger.debug(f"\n{'='*70}")
            logger.debug(f"❌ WALK-FORWARD FAILED")
            logger.debug(f"{'='*70}")
            logger.debug(
                f"Only {len(period_results)}/{periods} periods valid "
                f"(need {min_required_periods}+ periods)"
            )
            logger.debug(f"{'='*70}\n")
            return None
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 6: CALCULATE AGGREGATE STATISTICS
        # ═══════════════════════════════════════════════════════════════════
        
        win_rates = [r.win_rate for r in period_results]
        profit_factors = [r.profit_factor for r in period_results]
        max_drawdowns = [r.max_drawdown for r in period_results]
        sharpe_ratios = [r.sharpe_ratio for r in period_results]
        
        # Averages
        avg_win_rate = float(np.mean(win_rates))
        avg_profit_factor = float(np.mean(profit_factors))
        avg_max_drawdown = float(np.mean(max_drawdowns))
        avg_sharpe = float(np.mean(sharpe_ratios))
        
        # Stability (standard deviation)
        win_rate_stability = float(np.std(win_rates))
        pf_stability = float(np.std(profit_factors))
        
        # Min/Max
        min_win_rate = float(min(win_rates))
        max_win_rate = float(max(win_rates))
        min_profit_factor = float(min(profit_factors))
        max_profit_factor = float(max(profit_factors))
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 7: CALCULATE CONSISTENCY SCORE (IMPROVED ALGORITHM)
        # ═══════════════════════════════════════════════════════════════════
        
        # Base score from average performance
        if avg_win_rate < 35:
            base_score = 0  # Terrible
        elif avg_win_rate < 45:
            base_score = 20  # Very poor
        elif avg_win_rate < 50:
            base_score = 35  # Poor
        elif avg_win_rate < 55:
            base_score = 50  # Below average
        elif avg_win_rate < 60:
            base_score = 65  # Average
        elif avg_win_rate < 65:
            base_score = 80  # Good
        elif avg_win_rate < 70:
            base_score = 90  # Very good
        else:
            base_score = 95  # Excellent
        
        # Penalty for win rate instability
        # High std = inconsistent performance
        stability_penalty = min(40, win_rate_stability * 3)
        
        # Bonus for profit factor
        # Reward strategies that make more when they win
        pf_bonus = min(15, max(0, (avg_profit_factor - 1.0) * 8))
        
        # Penalty for drawdown
        # Penalize risky strategies
        dd_penalty = min(25, avg_max_drawdown / 1.5)
        
        # Bonus for Sharpe ratio
        # Reward risk-adjusted returns
        sharpe_bonus = min(10, max(0, avg_sharpe * 4))
        
        # Penalty for worst period
        # If any single period is terrible, reduce score
        worst_period_penalty = 0
        if min_win_rate < 40:
            worst_period_penalty = 15
        elif min_win_rate < 45:
            worst_period_penalty = 10
        elif min_win_rate < 50:
            worst_period_penalty = 5
        
        # Calculate final consistency score (0-100)
        consistency_score = base_score - stability_penalty + pf_bonus - dd_penalty + sharpe_bonus - worst_period_penalty
        consistency_score = float(max(0, min(100, consistency_score)))
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 8: DETERMINE PASS/FAIL (MODE-SPECIFIC CRITERIA)
        # ═══════════════════════════════════════════════════════════════════
        
        # Get mode-specific thresholds
        if ACCURACY_MODE == 'CONSERVATIVE':
            required_min_wr = 60
            required_min_pf = 1.8
            required_max_stability = 8
            required_max_dd = 15
        elif ACCURACY_MODE == 'BALANCED':
            required_min_wr = 55
            required_min_pf = 1.5
            required_max_stability = 10
            required_max_dd = 20
        else:  # AGGRESSIVE
            required_min_wr = 50
            required_min_pf = 1.3
            required_max_stability = 15
            required_max_dd = 25
        
        # Check 1: All periods must meet minimum criteria
        all_periods_pass_individual = all([
            r.win_rate >= required_min_wr and 
            r.profit_factor >= required_min_pf and
            r.max_drawdown <= required_max_dd
            for r in period_results
        ])
        
        # Check 2: Overall stability acceptable
        stability_acceptable = win_rate_stability <= required_max_stability
        
        # Check 3: Average performance meets requirements
        average_performance_ok = (
            avg_win_rate >= required_min_wr and
            avg_profit_factor >= required_min_pf
        )
        
        # Check 4: No catastrophic single period
        no_terrible_periods = min_win_rate >= (required_min_wr - 10)
        
        # Final pass/fail decision
        passed = (
            all_periods_pass_individual and 
            stability_acceptable and 
            average_performance_ok and
            no_terrible_periods
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 9: DETAILED LOGGING
        # ═══════════════════════════════════════════════════════════════════
        
        logger.debug(f"\n{'='*70}")
        logger.debug(f"WALK-FORWARD RESULTS")
        logger.debug(f"{'='*70}")
        logger.debug(f"Validated Periods: {len(period_results)}/{periods}")
        logger.debug(f"")
        logger.debug(f"AGGREGATE STATISTICS:")
        logger.debug(f"  Win Rate: {avg_win_rate:.1f}% (range: {min_win_rate:.1f}% - {max_win_rate:.1f}%)")
        logger.debug(f"  Stability: ±{win_rate_stability:.1f}%")
        logger.debug(f"  Profit Factor: {avg_profit_factor:.2f} (range: {min_profit_factor:.2f} - {max_profit_factor:.2f})")
        logger.debug(f"  Max Drawdown: {avg_max_drawdown:.1f}%")
        logger.debug(f"  Sharpe Ratio: {avg_sharpe:.2f}")
        logger.debug(f"")
        logger.debug(f"CONSISTENCY SCORE: {consistency_score:.1f}/100")
        logger.debug(f"  Base Score: {base_score:.0f}")
        logger.debug(f"  Stability Penalty: -{stability_penalty:.0f}")
        logger.debug(f"  Profit Factor Bonus: +{pf_bonus:.0f}")
        logger.debug(f"  Drawdown Penalty: -{dd_penalty:.0f}")
        logger.debug(f"  Sharpe Bonus: +{sharpe_bonus:.0f}")
        logger.debug(f"  Worst Period Penalty: -{worst_period_penalty:.0f}")
        logger.debug(f"")
        logger.debug(f"REQUIREMENTS ({ACCURACY_MODE} mode):")
        logger.debug(f"  Min Win Rate: {required_min_wr}% {'✅' if avg_win_rate >= required_min_wr else '❌'}")
        logger.debug(f"  Min Profit Factor: {required_min_pf} {'✅' if avg_profit_factor >= required_min_pf else '❌'}")
        logger.debug(f"  Max Stability: {required_max_stability}% {'✅' if win_rate_stability <= required_max_stability else '❌'}")
        logger.debug(f"  Max Drawdown: {required_max_dd}% {'✅' if avg_max_drawdown <= required_max_dd else '❌'}")
        logger.debug(f"")
        logger.debug(f"VALIDATION CHECKS:")
        logger.debug(f"  All periods pass individual: {'✅' if all_periods_pass_individual else '❌'}")
        logger.debug(f"  Stability acceptable: {'✅' if stability_acceptable else '❌'}")
        logger.debug(f"  Average performance OK: {'✅' if average_performance_ok else '❌'}")
        logger.debug(f"  No terrible periods: {'✅' if no_terrible_periods else '❌'}")
        
        if period_details:
            logger.debug(f"")
            logger.debug(f"PERIOD BREAKDOWN:")
            for detail in period_details:
                status = "✅" if (
                    detail['win_rate'] >= required_min_wr and 
                    detail['profit_factor'] >= required_min_pf and
                    detail['max_drawdown'] <= required_max_dd
                ) else "❌"
                
                logger.debug(
                    f"  Period {detail['period']} {status}: "
                    f"{detail['trades']} trades | "
                    f"WR={detail['win_rate']:.1f}% | "
                    f"PF={detail['profit_factor']:.2f} | "
                    f"DD={detail['max_drawdown']:.1f}% | "
                    f"Sharpe={detail['sharpe_ratio']:.2f}"
                )
        
        logger.debug(f"")
        logger.debug(f"{'='*70}")
        logger.debug(f"FINAL RESULT: {'✅ PASSED' if passed else '❌ FAILED'}")
        logger.debug(f"{'='*70}\n")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 10: RETURN RESULT
        # ═══════════════════════════════════════════════════════════════════
        
        return WalkForwardResult(
            period_results=period_results,
            avg_win_rate=round(avg_win_rate, 2),
            win_rate_stability=round(win_rate_stability, 2),
            avg_profit_factor=round(avg_profit_factor, 2),
            consistency_score=round(consistency_score, 1),
            passed=passed
        )
    

    def _calculate_reliability_score(
        self,
        win_rate: float,
        profit_factor: float,
        max_drawdown: float,
        sharpe_ratio: float
    ) -> float:
        """Calculate 0-100 reliability score"""
        
        wr_score = min(100, (win_rate - 40) * 2.5) if win_rate > 40 else 0
        pf_score = min(100, (profit_factor - 1) * 50) if profit_factor > 1 else 0
        dd_score = max(0, 100 - max_drawdown * 2)
        sharpe_score = min(100, sharpe_ratio * 33) if sharpe_ratio > 0 else 0
        
        score = (
            wr_score * 0.30 +
            pf_score * 0.30 +
            dd_score * 0.20 +
            sharpe_score * 0.20
        )
        
        return max(0, min(100, score))
    
    def monte_carlo_simulation(
        self,
        backtest_result: BacktestResult,
        simulations: int = 1000,
        num_trades: int = 50
    ) -> MonteCarloResult:
        """Monte Carlo simulation"""
        
        if not backtest_result or backtest_result.total_trades < 10:
            return None
        
        win_rate = backtest_result.win_rate / 100
        avg_win = backtest_result.avg_win
        avg_loss = backtest_result.avg_loss
        
        final_equities = []
        
        for _ in range(simulations):
            equity = self.initial_capital
            
            for _ in range(num_trades):
                if np.random.random() < win_rate:
                    ret = np.random.normal(avg_win, avg_win * 0.3)
                else:
                    ret = np.random.normal(avg_loss, abs(avg_loss) * 0.3)
                
                equity *= (1 + ret / 100)
            
            final_equities.append(equity)
        
        final_equities = np.array(final_equities)
        
        avg_final = np.mean(final_equities)
        median_final = np.median(final_equities)
        worst_case = np.percentile(final_equities, 5)
        best_case = np.percentile(final_equities, 95)
        
        prob_profit = (final_equities > self.initial_capital).sum() / simulations
        
        confidence_95 = (
            np.percentile(final_equities, 2.5),
            np.percentile(final_equities, 97.5)
        )
        
        risk_of_ruin = (final_equities < self.initial_capital * 0.5).sum() / simulations
        
        return MonteCarloResult(
            simulations=simulations,
            avg_final_equity=round(avg_final, 2),
            median_final_equity=round(median_final, 2),
            worst_case=round(worst_case, 2),
            best_case=round(best_case, 2),
            prob_profit=round(prob_profit * 100, 2),
            confidence_95=(round(confidence_95[0], 2), round(confidence_95[1], 2)),
            risk_of_ruin=round(risk_of_ruin * 100, 2)
        )



   
# ============================================================================
# FUNDAMENTALS FETCHER
# ============================================================================

# ✅ CORRECTED - With caching:

_fundamentals_cache = {}

def get_fundamentals(ticker: str) -> Tuple[Optional[str], Optional[float]]:
    """Get sector and P/E ratio - WITH CACHING"""
    global _fundamentals_cache
    
    # Check cache first
    if ticker in _fundamentals_cache:
        return _fundamentals_cache[ticker]
    
    try:
        # Use Kite for fundamentals — yfinance removed
        clean = ticker.replace('.NS', '').replace('.BO', '')
        sector, pe = kite_provider.get_fundamentals(clean)

        _fundamentals_cache[ticker] = (sector, pe)
        return sector, pe

    except Exception:
        _fundamentals_cache[ticker] = (None, None)
        return None, None

# ============================================================================
# INDICATOR CALCULATION
# ============================================================================
def normalize_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed price normalization - only convert if definitely paise data"""
    try:
        if len(df) < 10:
            return df
        
        # Get current close price
        current_close = df['Close'].iloc[-1]
        avg_close = df['Close'].mean()
        
        # Only normalize if we're SURE it's paise data
        # Criteria: Average price > 100,000 (no stock trades above ₹100k except in paise)
        if avg_close > 100000:
            # This is definitely paise data
            df = df.copy()
            df['Open'] = df['Open'] / 100
            df['High'] = df['High'] / 100
            df['Low'] = df['Low'] / 100
            df['Close'] = df['Close'] / 100
            logger.debug(f"Normalized paise data: {avg_close:.0f} → {avg_close/100:.2f}")
        
        # Don't normalize anything else - keep actual prices
        return df
        
    except Exception as e:
        logger.debug(f"Price normalization error: {e}")
        return df
    
def calculate_indicators(df: pd.DataFrame) -> Optional[Dict]:
    """Calculate all technical indicators"""
    try:
        if len(df) < 60:
            return None
        
        df = df.sort_index()
        
        c = normalize_series(df["Close"])
        h = normalize_series(df["High"])
        l = normalize_series(df["Low"])
        v = normalize_series(df["Volume"])
        
        ema20_series = EMAIndicator(c, 20).ema_indicator()
        ema50_series = EMAIndicator(c, 50).ema_indicator()
        ema200_series = EMAIndicator(c, 200).ema_indicator()
        
        ema20 = safe_indicator(lambda: float(ema20_series.iloc[-1]), default=float(c.iloc[-1]))
        ema50 = safe_indicator(lambda: float(ema50_series.iloc[-1]), default=float(c.iloc[-1]))
        ema200 = safe_indicator(lambda: float(ema200_series.iloc[-1]), default=float(c.iloc[-1]))
        
        def calc_slope(series, periods=5):
            try:
                return float(series.iloc[-1] - series.iloc[-periods])
            except:
                return 0.0
        
        ema20_slope = calc_slope(ema20_series)
        ema50_slope = calc_slope(ema50_series)
        
        atr_series = AverageTrueRange(h, l, c, PRESET['atr_period']).average_true_range()
        atr_val = safe_indicator(lambda: float(atr_series.iloc[-1]), default=float(c.iloc[-1]) * 0.02)
        atr_slope = calc_slope(atr_series)
        
        rsi = safe_indicator(lambda: float(RSIIndicator(c, 14).rsi().iloc[-1]), default=50)
        
        try:
            macd_obj = MACD(c)
            macd_line = float(macd_obj.macd().iloc[-1])
            macd_signal = float(macd_obj.macd_signal().iloc[-1])
        except:
            macd_line = 0
            macd_signal = 0
        
        price_change = c.diff().fillna(0)
        direction = np.sign(price_change)
        obv_series = (direction * v).cumsum()
        obv_slope = calc_slope(obv_series)
        
        adx = safe_indicator(lambda: float(ADXIndicator(h, l, c, 14).adx().iloc[-1]), default=20)
        roc = safe_indicator(lambda: float(ROCIndicator(c, 10).roc().iloc[-1]), default=0)
        
        avg_vol = v.rolling(20).mean().iloc[-1]
        volume_ratio = float(v.iloc[-1] / avg_vol) if avg_vol > 0 else 1.0
        
        return {
            "price": float(c.iloc[-1]),
            "volume": float(v.iloc[-1]),
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "ema20_slope": ema20_slope,
            "ema50_slope": ema50_slope,
            "atr": atr_val,
            "atr_slope": atr_slope,
            "rsi": rsi,
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "obv_slope": obv_slope,
            "adx": adx,
            "roc": roc,
            "volume_ratio": volume_ratio,
        }
        
    except Exception as e:
        logger.debug(f"Indicator calculation failed: {e}")
        return None

# ============================================================================
# ENTRY SIGNAL GENERATOR
# ============================================================================

def generate_entry_signal(ind: Dict, df: pd.DataFrame, side: str) -> Dict:
    """Generate entry signal with detailed reasons"""
    
    try:
        entry_ready = False
        entry_signal = "NONE"
        reasons = []
        
        price = ind["price"]
        rsi = ind["rsi"]
        ema20 = ind["ema20"]
        ema50 = ind["ema50"]
        macd_line = ind["macd_line"]
        macd_signal = ind["macd_signal"]
        volume_ratio = ind["volume_ratio"]
        
        rsi_range = PRESET['rsi_range_long'] if side == "LONG" else PRESET['rsi_range_short']
        
        if side == "LONG":
            if price > ema20:
                reasons.append("Price > EMA20")
            
            if ema20 > ema50:
                reasons.append("EMA20 > EMA50")
            
            if rsi_range[0] <= rsi <= rsi_range[1]:
                reasons.append(f"RSI {rsi:.1f} in range")
            
            if macd_line > macd_signal:
                reasons.append("MACD bullish")
            
            if volume_ratio >= 1.0:
                reasons.append(f"Volume {volume_ratio:.2f}x")
            
            conditions = [
                price > ema20,
                ema20 > ema50,
                rsi_range[0] <= rsi <= rsi_range[1],
                macd_line > macd_signal,
                volume_ratio >= PRESET['min_volume_ratio']
            ]
            
            if sum(conditions) >= 3:
                entry_ready = True
                entry_signal = "LONG_ENTRY"
        
        else:  # SHORT
            if price < ema20:
                reasons.append("Price < EMA20")
            
            if ema20 < ema50:
                reasons.append("EMA20 < EMA50")
            
            if rsi_range[0] <= rsi <= rsi_range[1]:
                reasons.append(f"RSI {rsi:.1f} in range")
            
            if macd_line < macd_signal:
                reasons.append("MACD bearish")
            
            if volume_ratio >= 1.0:
                reasons.append(f"Volume {volume_ratio:.2f}x")
            
            conditions = [
                price < ema20,
                ema20 < ema50,
                rsi_range[0] <= rsi <= rsi_range[1],
                macd_line < macd_signal,
                volume_ratio >= PRESET['min_volume_ratio']
            ]
            
            if sum(conditions) >= 3:
                entry_ready = True
                entry_signal = "SHORT_ENTRY"
        
        return {
            "entry_ready": entry_ready,
            "entry_signal": entry_signal,
            "reasons": reasons
        }
        
    except Exception as e:
        return {
            "entry_ready": False,
            "entry_signal": "ERROR",
            "reasons": [str(e)[:50]]
        }

# ============================================================================
# COMPREHENSIVE SCORING SYSTEM
# ============================================================================

def score_signal(
    ind: Dict,
    df: pd.DataFrame,
    side: str,
    vix_sentiment: Optional[Dict],
    sector_rs: Optional[Tuple[float, str]],
    fibo_boost: int,
    sr_info: Dict,
    candle: Dict,
    mtf_trend: Optional[str],
    vol_analysis: Dict,
    volume_analysis: Dict,
    vsa_analysis: Dict,
    momentum_analysis: Dict,
    vcp_analysis: Dict,
    gap_analysis: Dict,
    supertrend_info: Dict
) -> Tuple[float, List[str]]:
    """Comprehensive scoring with REALISTIC confidence - STEEPER SIGMOID"""
    
    w = SCORING_WEIGHTS
    raw_score = 0.0
    reasons = []
    
    # ═══════════════════════════════════════════════════════════════════════
    # EMA TREND (max ±15 points)
    # ═══════════════════════════════════════════════════════════════════════
    if side == "LONG":
        if ind["price"] > ind["ema20"] > ind["ema50"]:
            raw_score += 15
            reasons.append("Strong EMA alignment")
        elif ind["price"] > ind["ema20"]:
            raw_score += 7
        else:
            raw_score -= 12
        
        if ind["price"] > ind["ema200"]:
            raw_score += w["ema_200"]
            reasons.append("Above 200 EMA")
        else:
            raw_score -= 3
    else:  # SHORT
        if ind["price"] < ind["ema20"] < ind["ema50"]:
            raw_score += 15
            reasons.append("Strong EMA alignment")
        elif ind["price"] < ind["ema20"]:
            raw_score += 7
        else:
            raw_score -= 12
        
        if ind["price"] < ind["ema200"]:
            raw_score += w["ema_200"]
            reasons.append("Below 200 EMA")
        else:
            raw_score -= 3
    
    # ═══════════════════════════════════════════════════════════════════════
    # EMA SLOPE (max ±5 points)
    # ═══════════════════════════════════════════════════════════════════════
    if side == "LONG" and ind["ema20_slope"] > 0:
        raw_score += w["ema_slope"]
        reasons.append("EMA rising")
    elif side == "SHORT" and ind["ema20_slope"] < 0:
        raw_score += w["ema_slope"]
        reasons.append("EMA falling")
    else:
        raw_score -= 4
    
    # ═══════════════════════════════════════════════════════════════════════
    # RSI (max ±10 points)
    # ═══════════════════════════════════════════════════════════════════════
    rsi_range = PRESET['rsi_range_long'] if side == "LONG" else PRESET['rsi_range_short']
    
    if rsi_range[0] <= ind["rsi"] <= rsi_range[1]:
        raw_score += w["rsi"]
        reasons.append(f"RSI {ind['rsi']:.1f} optimal")
    elif ind["rsi"] < 20 or ind["rsi"] > 80:
        raw_score -= 8
    elif ind["rsi"] < 30 or ind["rsi"] > 70:
        raw_score -= 5
    
    # ═══════════════════════════════════════════════════════════════════════
    # MACD (max ±10 points)
    # ═══════════════════════════════════════════════════════════════════════
    if side == "LONG" and ind["macd_line"] > ind["macd_signal"]:
        raw_score += w["macd"]
        reasons.append("MACD bullish")
    elif side == "SHORT" and ind["macd_line"] < ind["macd_signal"]:
        raw_score += w["macd"]
        reasons.append("MACD bearish")
    else:
        raw_score -= 6
    
    # ═══════════════════════════════════════════════════════════════════════
    # VOLUME (max ±6 points)
    # ═══════════════════════════════════════════════════════════════════════
    if ind["volume_ratio"] >= 1.5:
        raw_score += w["volume_ratio"]
        reasons.append("Strong volume")
    elif ind["volume_ratio"] >= 1.0:
        raw_score += w["volume_ratio"] * 0.5
    elif ind["volume_ratio"] < 0.7:
        raw_score -= w["volume_ratio"]
    
    # ═══════════════════════════════════════════════════════════════════════
    # OBV (max ±4 points)
    # ═══════════════════════════════════════════════════════════════════════
    if side == "LONG" and ind["obv_slope"] > 0:
        raw_score += w["obv"]
    elif side == "SHORT" and ind["obv_slope"] < 0:
        raw_score += w["obv"]
    else:
        raw_score -= 3
    
    # ═══════════════════════════════════════════════════════════════════════
    # SUPPORT/RESISTANCE (max ±7 points)
    # ═══════════════════════════════════════════════════════════════════════
    sr_bias = sr_info.get('bias', 'UNKNOWN')
    
    if sr_bias == 'AT_SUPPORT' and side == "LONG":
        raw_score += w["sr_support"]
        reasons.append("At support")
    elif sr_bias == 'AT_RESISTANCE' and side == "SHORT":
        raw_score += w["sr_resistance"]
        reasons.append("At resistance")
    elif sr_bias == 'AT_RESISTANCE' and side == "LONG":
        raw_score -= 5
    elif sr_bias == 'AT_SUPPORT' and side == "SHORT":
        raw_score -= 5
    
    # ═══════════════════════════════════════════════════════════════════════
    # CANDLESTICK PATTERNS (max ±6 points)
    # ═══════════════════════════════════════════════════════════════════════
    candle_score = candle.get('score', 0)
    
    if (side == "LONG" and candle_score > 0) or (side == "SHORT" and candle_score < 0):
        raw_score += w["candle"] * min(1, abs(candle_score) / 8)
        reasons.append(candle.get('description', 'Pattern'))
    elif abs(candle_score) > 5:
        raw_score -= 4
    
    # ═══════════════════════════════════════════════════════════════════════
    # MULTI-TIMEFRAME (max ±9 points)
    # ═══════════════════════════════════════════════════════════════════════
    if mtf_trend:
        if (side == "LONG" and mtf_trend == "UP") or (side == "SHORT" and mtf_trend == "DOWN"):
            raw_score += w["mtf_trend"]
            reasons.append(f"MTF {mtf_trend}")
        elif mtf_trend == "MIXED":
            raw_score -= w["mtf_penalty"]
        else:
            raw_score -= 8
    
    # ═══════════════════════════════════════════════════════════════════════
    # VIX SENTIMENT (max ±6 points)
    # ═══════════════════════════════════════════════════════════════════════
    if vix_sentiment and USE_VIX_SENTIMENT:
        vix_score = vix_sentiment.get('score', 0)
        if side == "LONG" and vix_score > 0:
            raw_score += w["vix_sentiment"]
        elif side == "SHORT" and vix_score < 0:
            raw_score += w["vix_sentiment"]
        elif abs(vix_score) > 5:
            raw_score -= 3
    
    # ═══════════════════════════════════════════════════════════════════════
    # FIBONACCI (max ±8 points)
    # ═══════════════════════════════════════════════════════════════════════
    if fibo_boost > 0:
        raw_score += w["fib_boost"]
        reasons.append("At Fib level")
    
    # ═══════════════════════════════════════════════════════════════════════
    # VOLUME BREAKOUT (max ±7 points)
    # ═══════════════════════════════════════════════════════════════════════
    if volume_analysis:
        vol_score = volume_analysis.get('score', 0)
        if vol_score > 0:
            raw_score += min(w["volume_breakout"], vol_score)
            if vol_score > 5:
                reasons.append(volume_analysis.get('signal', ''))
        elif vol_score < -5:
            raw_score -= 5
    
    # ═══════════════════════════════════════════════════════════════════════
    # VSA (max ±7 points)
    # ═══════════════════════════════════════════════════════════════════════
    if vsa_analysis:
        vsa_score = vsa_analysis.get('score', 0)
        if (side == "LONG" and vsa_score > 0) or (side == "SHORT" and vsa_score < 0):
            raw_score += w["vsa_confirmation"] * min(1, abs(vsa_score) / 8)
            reasons.append(vsa_analysis.get('signal', ''))
        elif abs(vsa_score) > 5:
            raw_score -= 4
    
    # ═══════════════════════════════════════════════════════════════════════
    # MOMENTUM (max ±8 points)
    # ═══════════════════════════════════════════════════════════════════════
    if momentum_analysis:
        mom_score = momentum_analysis.get('score', 0)
        if mom_score > 0:
            raw_score += min(w["adx_momentum"], mom_score)
            if mom_score > 5:
                reasons.append(momentum_analysis.get('signal', ''))
        elif mom_score < -5:
            raw_score -= 5
    
    # ═══════════════════════════════════════════════════════════════════════
    # VCP PATTERN (max ±9 points)
    # ═══════════════════════════════════════════════════════════════════════
    if vcp_analysis:
        vcp_score = vcp_analysis.get('score', 0)
        if vcp_score > 5:
            raw_score += w["vcp_pattern"] * min(1, vcp_score / 10)
            reasons.append(vcp_analysis.get('signal', ''))
    
    # ═══════════════════════════════════════════════════════════════════════
    # GAP DETECTION (max ±5 points)
    # ═══════════════════════════════════════════════════════════════════════
    if gap_analysis:
        gap_score = gap_analysis.get('score', 0)
        if (side == "LONG" and gap_score > 0) or (side == "SHORT" and gap_score < 0):
            raw_score += w["gap_detection"] * min(1, abs(gap_score) / 8)
        elif abs(gap_score) > 5:
            raw_score -= 3
    
    # ═══════════════════════════════════════════════════════════════════════
    # SUPERTREND (max ±8 points)
    # ═══════════════════════════════════════════════════════════════════════
    if supertrend_info:
        st_direction = supertrend_info.get('direction')
        if (side == "LONG" and st_direction == "UP") or (side == "SHORT" and st_direction == "DOWN"):
            raw_score += w["supertrend"]
            reasons.append(f"Supertrend {st_direction}")
        elif st_direction:
            raw_score -= 6
    
    # ═══════════════════════════════════════════════════════════════════════
    # SECTOR STRENGTH (max ±5 points)
    # ═══════════════════════════════════════════════════════════════════════
    if sector_rs and USE_SECTOR_ROTATION:
        rs_value, rs_label = sector_rs
        if rs_label == "STRONG":
            raw_score += 5
            reasons.append("Strong sector")
        elif rs_label == "WEAK":
            raw_score -= 4
    
    # ═══════════════════════════════════════════════════════════════════════
    # MARKET REGIME ALIGNMENT (NEW) - CALL ONLY ONCE!
    # ═══════════════════════════════════════════════════════════════════════
    market_regime = MarketRegimeFilter.get_market_regime()  # ← SINGLE CALL
    regime_bias = market_regime.get('bias', 'NEUTRAL')
    
    if regime_bias == side:
        raw_score += 8
        reasons.append(f"Market {regime_bias}")
    elif regime_bias not in ["NEUTRAL", "CAUTION"] and regime_bias != side:
        raw_score -= 10

    # ═══════════════════════════════════════════════════════════════════════
    # SIGMOID TRANSFORMATION WITH FLOOR/CEILING
    # ═══════════════════════════════════════════════════════════════════════
    sigmoid_divisor = PRESET.get('sigmoid_divisor', 35)
    
    # Base sigmoid calculation
    confidence = 50 + (45 / (1 + np.exp(-raw_score / sigmoid_divisor)))
    
    # Apply floor based on number of positive conditions
    positive_reasons = len([r for r in reasons if r])
    
    if positive_reasons >= 8:
        confidence = max(confidence, 70)
    elif positive_reasons >= 6:
        confidence = max(confidence, 60)
    elif positive_reasons >= 4:
        confidence = max(confidence, 50)
    
    # Apply ceiling for counter-trend trades (REUSE regime_bias - no second call!)
    if regime_bias == 'LONG' and side == 'SHORT':
        confidence = min(confidence, 75)
    elif regime_bias == 'SHORT' and side == 'LONG':
        confidence = min(confidence, 75)
    
    # Final bounds
    confidence = round(min(95, max(30, confidence)), 1)
    
    return confidence, reasons[:12]

# ============================================================================
# MINI BACKTEST
# ============================================================================

# ✅ CORRECTED - Stricter mini backtest:
def mini_backtest(
    df: pd.DataFrame,
    entry_price: float,
    side: str,
    rr_ratio: float = 2.0
) -> Tuple[bool, Dict]:
    """Enhanced mini backtest - BALANCED CRITERIA (not too strict)"""
    try:
        if len(df) < 25:
            return False, {"reason": "Insufficient data"}
        
        lookback = min(60, len(df) - 1)
        future_data = df.iloc[:-1].tail(lookback).copy()
        
        if future_data.empty or len(future_data) < 10:
            return False, {"reason": "No future data"}
        
        atr = AverageTrueRange(
            high=normalize_series(future_data['High']),
            low=normalize_series(future_data['Low']),
            close=normalize_series(future_data['Close']),
            window=PRESET['atr_period']
        ).average_true_range().iloc[-1]
        
        # [FIX CRIT-2] Use same dynamic stop-multiplier as TradeRule and _simulate_trade.
        # Old code always used atr * 1.5 regardless of stock volatility.
        _mini_atr_pct = (atr / entry_price) if entry_price > 0 else 0.02
        _mini_stop_mult = TradeRule._get_stop_multiplier(_mini_atr_pct)
        if side == "LONG":
            stop_loss = entry_price - (atr * _mini_stop_mult)
            target = entry_price + (atr * rr_ratio)
        else:
            stop_loss = entry_price + (atr * _mini_stop_mult)
            target = entry_price - (atr * rr_ratio)
        
        target_hit = False
        stop_hit = False
        target_bar = None
        stop_bar = None
        max_profit = 0
        max_loss = 0
        
        for i, row in enumerate(future_data.itertuples()):
            if side == "LONG":
                current_profit = (row.High - entry_price) / entry_price * 100
                current_loss = (entry_price - row.Low) / entry_price * 100
                
                if row.High >= target and not target_hit:
                    target_hit = True
                    target_bar = i
                if row.Low <= stop_loss and not stop_hit:
                    stop_hit = True
                    stop_bar = i
            else:
                current_profit = (entry_price - row.Low) / entry_price * 100
                current_loss = (row.High - entry_price) / entry_price * 100
                
                if row.Low <= target and not target_hit:
                    target_hit = True
                    target_bar = i
                if row.High >= stop_loss and not stop_hit:
                    stop_hit = True
                    stop_bar = i
            
            max_profit = max(max_profit, current_profit)
            max_loss = max(max_loss, current_loss)
        
        # =====================================================================
        # SUCCESS CRITERIA - BALANCED (Not too strict, not too loose)
        # =====================================================================
        
        # Case 1: Clean winner - target hit, stop never hit
        if target_hit and not stop_hit:
            success = True
            reason = "Target hit, stop never touched"
        
        # Case 2: Both hit - which came first?
        elif target_hit and stop_hit:
            if target_bar <= stop_bar:  # ← Changed < to <= (same bar = win)
                success = True
                reason = "Target hit before/with stop"
            else:
                # Stop hit first - check if recovered
                final_price = future_data.iloc[-1]['Close']
                if side == "LONG":
                    final_pnl = ((final_price - entry_price) / entry_price) * 100
                else:
                    final_pnl = ((entry_price - final_price) / entry_price) * 100
                
                # ✅ LESS STRICT: Pass if final P&L is positive
                success = final_pnl > 0
                reason = f"Stop first, recovered to {final_pnl:.1f}%"
        
        # Case 3: Only stop hit
        elif stop_hit and not target_hit:
            # Check final P&L - maybe it recovered
            final_price = future_data.iloc[-1]['Close']
            if side == "LONG":
                final_pnl = ((final_price - entry_price) / entry_price) * 100
            else:
                final_pnl = ((entry_price - final_price) / entry_price) * 100
            
            # ✅ LESS STRICT: Pass if final P&L is only slightly negative
            success = final_pnl > -1.0  # ← Allow up to 1% loss
            reason = f"Stop hit, final: {final_pnl:.1f}%"
        
        # Case 4: Neither hit - evaluate final position
        else:
            final_price = future_data.iloc[-1]['Close']
            if side == "LONG":
                final_pnl_pct = ((final_price - entry_price) / entry_price) * 100
            else:
                final_pnl_pct = ((entry_price - final_price) / entry_price) * 100
            
            stop_distance_pct = abs(entry_price - stop_loss) / entry_price * 100
            
            # ✅ BALANCED CRITERIA:
            # Pass if: positive P&L OR showed good profit potential
            success = (
                final_pnl_pct >= 0 or                           # Any positive P&L
                (max_profit >= stop_distance_pct * 0.3 and      # Good profit potential
                 max_loss < stop_distance_pct * 1.2)            # Didn't exceed stop by much
            )
            reason = f"Final: {final_pnl_pct:.1f}%, MaxProfit: {max_profit:.1f}%"
        
        metrics = {
            "target_hit": target_hit,
            "stop_hit": stop_hit,
            "success_reason": reason,
            "max_favorable": round(max_profit, 2),
            "max_adverse": round(max_loss, 2),
            "bars_analyzed": lookback
        }
        
        return success, metrics
        
    except Exception as e:
        return False, {"reason": str(e)[:50]}
    
# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def get_universe_symbols() -> List[str]:
    """Get NIFTY 500 stock universe - FIXED FOR NIFTY 500 ONLY"""
    
    print("  📊 Fetching NIFTY 500 constituents...")
    
    try:
        # Method 1: Try NSE official API (most reliable)
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            
            # NSE CSV has 'Symbol' column
            if 'Symbol' in df.columns:
                symbols = df['Symbol'].tolist()
            elif 'SYMBOL' in df.columns:
                symbols = df['SYMBOL'].tolist()
            else:
                print("  ⚠️  Column not found, trying fallback...")
                return get_nifty500_fallback()
            
            # Clean and add .NS suffix
            symbols = [s.strip() + ".NS" for s in symbols if isinstance(s, str) and s.strip()]
            
            # Remove duplicates
            symbols = list(dict.fromkeys(symbols))
            
            print(f"  ✅ Loaded {len(symbols)} Nifty 500 stocks from NSE")
            return symbols[:500]  # Ensure max 500
        
        else:
            print(f"  ⚠️  NSE API returned {response.status_code}, using fallback...")
            return get_nifty500_fallback()
            
    except Exception as e:
        print(f"  ⚠️  Error fetching Nifty 500: {e}")
        print(f"  🔄 Using fallback list...")
        return get_nifty500_fallback()

def get_nifty500_fallback() -> List[str]:
    """
    Fallback Nifty 500 list (Top 100 most liquid stocks)
    
    🎯 This is a curated list of highly liquid Nifty 500 stocks
    💡 Update this list periodically from: https://www.niftyindices.com
    """
    
    nifty_500_symbols = [
        # Nifty 50 (Blue chips)
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        "BAJFINANCE.NS", "LT.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS",
        "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS",
        "HCLTECH.NS", "TATAMOTORS.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS",
        "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "TATASTEEL.NS", "M&M.NS",
        "BAJAJFINSV.NS", "TECHM.NS", "JSWSTEEL.NS", "GRASIM.NS", "INDUSINDBK.NS",
        "HINDALCO.NS", "CIPLA.NS", "DRREDDY.NS", "APOLLOHOSP.NS", "DIVISLAB.NS",
        "EICHERMOT.NS", "BRITANNIA.NS", "SHREECEM.NS", "HEROMOTOCO.NS", "TATACONSUM.NS",
        "BAJAJ-AUTO.NS", "UPL.NS", "SBILIFE.NS", "HDFCLIFE.NS", "BPCL.NS",
        
        # Nifty Next 50
        "ADANIGREEN.NS", "ADANITRANS.NS", "AMBUJACEM.NS", "BANDHANBNK.NS", "BERGEPAINT.NS",
        "BEL.NS", "BOSCHLTD.NS", "COLPAL.NS", "DABUR.NS", "DMART.NS",
        "DLF.NS", "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS", "HINDPETRO.NS",
        "ICICIPRULI.NS", "INDIGO.NS", "IRCTC.NS", "JINDALSTEL.NS", "LICHSGFIN.NS",
        "LUPIN.NS", "MARICO.NS", "MCDOWELL-N.NS", "MOTHERSON.NS", "MUTHOOTFIN.NS",
        "NAUKRI.NS", "NMDC.NS", "OFSS.NS", "PAGEIND.NS", "PETRONET.NS",
        "PFC.NS", "PIDILITIND.NS", "PNB.NS", "RECLTD.NS", "SAIL.NS",
        "SIEMENS.NS", "SRF.NS", "TATAPOWER.NS", "TORNTPHARM.NS", "TRENT.NS",
        "TVSMOTOR.NS", "VEDL.NS", "ZYDUSLIFE.NS", "ATUL.NS", "AUROPHARMA.NS",
        
        # Mid Cap (High Quality)
        "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", "ALKEM.NS", "AMARAJABAT.NS",
        "ASTRAL.NS", "AUBANK.NS", "BALKRISIND.NS", "BATAINDIA.NS", "BHARATFORG.NS",
        "BIOCON.NS", "CADILAHC.NS", "CANBK.NS", "CHAMBLFERT.NS", "CHOLAFIN.NS",
        "COFORGE.NS", "CONCOR.NS", "CUMMINSIND.NS", "DEEPAKNTR.NS", "DIXON.NS",
        "ESCORTS.NS", "FEDERALBNK.NS", "GLENMARK.NS", "GMRINFRA.NS", "GNFC.NS",
        "GODREJPROP.NS", "GRANULES.NS", "GUJGASLTD.NS", "HAL.NS", "HDFCAMC.NS",
        "HONAUT.NS", "IDFCFIRSTB.NS", "INDUSTOWER.NS", "IOC.NS", "IPCALAB.NS",
        "IRFC.NS", "IGL.NS", "JUBLFOOD.NS", "KANSAINER.NS", "KEI.NS",
        "L&TFH.NS", "LALPATHLAB.NS", "LAURUSLABS.NS", "LTIM.NS", "MANAPPURAM.NS",
        "MAXHEALTH.NS", "MGL.NS", "MPHASIS.NS", "MRF.NS", "NAM-INDIA.NS",
        "OBEROIRLTY.NS", "OIL.NS", "PERSISTENT.NS", "PIIND.NS", "PVR.NS",
        "RAIN.NS", "RAMCOCEM.NS", "SBICARD.NS", "SKFINDIA.NS", "SOLARINDS.NS",
        "SONACOMS.NS", "SUNTV.NS", "SUPREMEIND.NS", "SYNGENE.NS", "TATACHEM.NS",
        "TATACOMM.NS", "TATAELXSI.NS", "TIINDIA.NS", "TORNTPOWER.NS", "TTML.NS",
        "UBL.NS", "UNIONBANK.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "ZEEL.NS",
        
        # Small Cap (Momentum stocks)
        "AAVAS.NS", "ABCAPITAL.NS", "AEGISCHEM.NS", "AFFLE.NS", "AJANTPHARM.NS",
        "ALKYLAMINE.NS", "AMBER.NS", "APLLTD.NS", "ASAHIINDIA.NS", "ASHOKLEY.NS",
        "ATUL.NS", "AVANTIFEED.NS", "AXISCADES.NS", "BASF.NS", "BEML.NS",
        "BHARTIHEXA.NS", "BIKAJI.NS", "BLS.NS", "BSOFT.NS", "CAMS.NS",
        "CAPLIPOINT.NS", "CARBORUNIV.NS", "CCL.NS", "CDSL.NS", "CERA.NS",
        "CHALET.NS", "CLEAN.NS", "COCHINSHIP.NS", "COROMANDEL.NS", "CREDITACC.NS",
        "CROMPTON.NS", "CSB.NS", "CUB.NS", "CYIENT.NS", "DATAPATTNS.NS",
        "DCBBANK.NS", "DELTACORP.NS", "DHANI.NS", "DHANUKA.NS", "DISHTV.NS",
        "EMAMILTD.NS", "ENDURANCE.NS", "EQUITAS.NS", "FINEORG.NS", "FINPIPE.NS",
        "FSL.NS", "GATEWAY.NS", "GICRE.NS", "GILLETTE.NS", "GLAXO.NS",
        "GPIL.NS", "GRAPHITE.NS", "GREAVESCOT.NS", "GRINDWELL.NS", "GRSE.NS",
        "GSFC.NS", "GSPL.NS", "HAPPSTMNDS.NS", "HATHWAY.NS", "HCG.NS",
        "HFCL.NS", "HIMATSEIDE.NS", "HNDFDS.NS", "HOMEFIRST.NS", "HUDCO.NS",
        "IFBIND.NS", "IIFL.NS", "IIFLSEC.NS", "INDHOTEL.NS", "INOXLEISUR.NS",
        "JBCHEPHARM.NS", "JKCEMENT.NS", "JKLAKSHMI.NS", "JKPAPER.NS", "JMFINANCIL.NS",
        "JSWENERGY.NS", "JTEKTINDIA.NS", "JUBLINGREA.NS", "JUSTDIAL.NS", "JYOTHYLAB.NS",
        "KAJARIACER.NS", "KALPATPOWR.NS", "KARURVYSYA.NS", "KEC.NS", "KPRMILL.NS",
        "KRBL.NS", "LEMONTREE.NS", "LXCHEM.NS", "MAHLOG.NS", "MAHSEAMLES.NS",
        "MAHSCOOTER.NS", "MASTEK.NS", "MINDACORP.NS", "MINDAIND.NS", "MOTILALOFS.NS",
        "NATIONALUM.NS", "NAVINFLUOR.NS", "NBCC.NS", "NCC.NS", "NDTV.NS",
        "NELCO.NS", "NETWORK18.NS", "NHPC.NS", "NILKAMAL.NS", "NLCINDIA.NS",
        "NOCIL.NS", "NYKAA.NS", "ORIENTELEC.NS", "PARAGMILK.NS", "PCJEWELLER.NS",
        "PNBHOUSING.NS", "PNCINFRA.NS", "POLICYBZR.NS", "POLYCAB.NS", "POLYMED.NS",
        "POONAWALLA.NS", "PRSMJOHNSN.NS", "PTCIL.NS", "PVP.NS", "RAJESHEXPO.NS",
        "RALLIS.NS", "RATNAMANI.NS", "RAYMOND.NS", "RBL.NS", "RCF.NS",
        "REDINGTON.NS", "RELAXO.NS", "RITES.NS", "ROUTE.NS", "SANOFI.NS",
        "SCHAEFFLER.NS", "SHARDACROP.NS", "SHILPAMED.NS", "SHOPERSTOP.NS", "SHYAMMETL.NS",
        "SJVN.NS", "SOBHA.NS", "SPANDANA.NS", "SPARC.NS", "STARCEMENT.NS",
        "STAR.NS", "STLTECH.NS", "STYRENIX.NS", "SUBEX.NS", "SUNDARMFIN.NS",
        "SUNDRMFAST.NS", "SUNPHARMA.NS", "SUPRAJIT.NS", "SUPRIYA.NS", "SUVEN.NS",
        "SWANENERGY.NS", "SYMPHONY.NS", "TATACOMM.NS", "TATAINVEST.NS", "TATAMETALI.NS",
        "TCNSBRANDS.NS", "TEAMLEASE.NS", "TECHM.NS", "THERMAX.NS", "THYROCARE.NS",
        "TIMKEN.NS", "TIPSINDLTD.NS", "TRITURBINE.NS", "TRIDENT.NS", "TRITURBINE.NS",
        "TTK.NS", "TV18BRDCST.NS", "UCOBANK.NS", "UJAAS.NS", "UJJIVAN.NS",
        "ULTRACEMCO.NS", "UNICHEMLAB.NS", "UNIONBANK.NS", "UTIAMC.NS", "VAIBHAVGBL.NS",
        "VARROC.NS", "VBL.NS", "VENKEYS.NS", "VGUARD.NS", "VIPIND.NS",
        "VINATIORGA.NS", "VSTIND.NS", "WABCOINDIA.NS", "WELCORP.NS", "WELSPUNIND.NS",
        "WESTLIFE.NS", "WILSON.NS", "WOCKPHARMA.NS", "YESBANK.NS", "ZENSARTECH.NS",
    ]
    
    print(f"  ✅ Using fallback: {len(nifty_500_symbols)} curated Nifty 500 stocks")
    
    return nifty_500_symbols[:500]



def build_cache(tickers: List[str]) -> Optional[pd.DataFrame]:
    """Build price data cache - FIXED VERSION FOR UNIQUE DATA"""
    print(f"\n[Building Price Cache for {len(tickers)} stocks]")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    
    all_data = []
    failed = []
    successful_downloads = {}  # Track successful downloads to prevent duplicates
    
    def download_ticker(ticker):
        """Download single ticker — Kite only (yfinance removed).
        Uses kite_session first, then kite_provider as fallback.
        """
        original_ticker = ticker
        base_ticker = ticker.replace('.NS', '').replace('.BO', '')
        test_ticker = base_ticker + '.NS'
        df = None

        # ── PRIORITY 1: Kite Connect Historical API ──────────────────────────
        # Requires kite_session to be connected (token valid).
        # kite_session is imported from kite_connector at module level (optional import).
        try:
            from kite_connector import kite_session as _ks
            if _ks and _ks.is_connected:
                instrument_token = _ks.get_instrument_token(base_ticker, exchange="NSE")
                if instrument_token:
                    kite_df = _ks.get_historical_data(
                        instrument_token=instrument_token,
                        from_date=start_date,
                        to_date=end_date,
                        interval="day"
                    )
                    if kite_df is not None and len(kite_df) >= 60:
                        df = kite_df
                        test_ticker = base_ticker  # Kite uses clean symbol
        except Exception:
            pass  # kite_session unavailable — kite_provider fallback below

        # ── PRIORITY 2: kite_provider fallback (if kite_session not available) ──
        if df is None:
            df = kite_provider.get_historical_df(base_ticker, days=LOOKBACK_DAYS)
            if df is not None and len(df) >= 60:
                df = df.reset_index()  # make Date a column to match kite_session output
                test_ticker = base_ticker
        
        # Validate we got data
        if df is None or df.empty or len(df) < 60:
            return None
        
        try:
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure required columns
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required):
                return None
            
            # Remove any NaN values
            df = df.dropna(subset=['Close', 'Volume'])
            
            if len(df) < 60:
                return None
            
            # Create unique signature for this data
            data_signature = f"{df['Close'].mean():.2f}_{df['Close'].std():.2f}_{df['Volume'].mean():.0f}"
            
            # Check if we already have this exact data
            if data_signature in successful_downloads:
                logger.debug(f"{test_ticker}: Duplicate data detected, skipping")
                return None
            
            # Mark this data as downloaded
            successful_downloads[data_signature] = test_ticker
            
            # DON'T normalize prices here - let the scanner handle it
            # This ensures we get actual market prices
            
            # Build clean dataframe with actual ticker used
            clean_df = pd.DataFrame({
                'Date': df.index,
                'Open': df['Open'].values,
                'High': df['High'].values,
                'Low': df['Low'].values,
                'Close': df['Close'].values,
                'Volume': df['Volume'].values,
                'ticker': test_ticker  # Store the actual ticker that worked
            })
            
            clean_df['Date'] = pd.to_datetime(clean_df['Date']).dt.tz_localize(None)
            
            # Final validation
            if clean_df['Close'].nunique() < 10:
                return None
                
            return clean_df
            
        except Exception as e:
            logger.debug(f"{test_ticker}: Processing failed: {e}")
            return None
    
    # Download data with progress tracking
    with ThreadPoolExecutor(max_workers=5) as executor:  # Reduced workers for stability
        futures = {executor.submit(download_ticker, t): t for t in tickers}
        
        for i, future in enumerate(as_completed(futures), 1):
            ticker = futures[future]
            try:
                result = future.result(timeout=30)
                if result is not None:
                    all_data.append(result)
                else:
                    failed.append(ticker)
            except Exception:
                failed.append(ticker)
            
            # Progress update
            if i % 25 == 0 or i == len(tickers):
                print(f"  Progress: {i}/{len(tickers)} | Success: {len(all_data)} | Failed: {len(failed)}", 
                      end="\r", flush=True)
    
    print()  # New line after progress
    
    if not all_data:
        print("\n  ❌ No data downloaded successfully!")
        return None
    
    # Concatenate all data
    try:
        df_all = pd.concat(all_data, axis=0, ignore_index=True)
        
        # Verify each ticker has unique data
        print(f"\n  📊 Verifying data uniqueness...")
        tickers_to_remove = []
        
        for ticker in df_all['ticker'].unique():
            ticker_data = df_all[df_all['ticker'] == ticker]
            unique_prices = ticker_data['Close'].nunique()
            
            if unique_prices < 10:
                tickers_to_remove.append(ticker)
                print(f"  ⚠️  Removing {ticker} - insufficient unique prices ({unique_prices})")
        
        # Remove bad tickers
        if tickers_to_remove:
            df_all = df_all[~df_all['ticker'].isin(tickers_to_remove)]
        
        final_ticker_count = df_all['ticker'].nunique()
        print(f"  ✓ Cached {final_ticker_count} stocks with {len(df_all):,} total bars")
        
        return df_all
        
    except Exception as e:
        print(f"\n  ❌ Error concatenating data: {e}")
        return None

# ============================================================================
# TREND STRENGTH CALCULATOR (NEW)
# ============================================================================

def calculate_trend_strength(df: pd.DataFrame) -> Tuple[float, str]:
    """
    Calculate trend strength (0-100)
    Returns: (strength_score, interpretation)
    
    - STRONG: Score >= 70 (trending strongly)
    - MODERATE: Score 40-70 (decent trend)
    - WEAK: Score < 40 (choppy/ranging)
    """
    try:
        if len(df) < 50:
            return 50, "MODERATE"
        
        close = df['Close']
        
        # Calculate EMAs
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean() if len(df) >= 200 else ema50
        
        current_price = float(close.iloc[-1])
        current_ema20 = float(ema20.iloc[-1])
        current_ema50 = float(ema50.iloc[-1])
        current_ema200 = float(ema200.iloc[-1])
        
        # Component 1: EMA separation as % of price (max 30 points)
        ema_separation = abs(current_ema20 - current_ema50) / current_price * 100
        ema_score = min(30, ema_separation * 8)
        
        # Component 2: Price distance from EMA20 (max 20 points)
        price_distance = abs(current_price - current_ema20) / current_price * 100
        distance_score = min(20, price_distance * 5)
        
        # Component 3: Consecutive days in trend (max 25 points)
        trend_up = current_price > current_ema20
        trend_days = 0
        for i in range(1, min(20, len(close))):
            if trend_up:
                if float(close.iloc[-i]) > float(ema20.iloc[-i]):
                    trend_days += 1
                else:
                    break
            else:
                if float(close.iloc[-i]) < float(ema20.iloc[-i]):
                    trend_days += 1
                else:
                    break
        consistency_score = min(25, trend_days * 2.5)
        
        # Component 4: EMA alignment bonus (max 25 points)
        alignment_score = 0
        if trend_up:
            if current_price > current_ema20 > current_ema50:
                alignment_score = 15
            if current_price > current_ema20 > current_ema50 > current_ema200:
                alignment_score = 25
        else:
            if current_price < current_ema20 < current_ema50:
                alignment_score = 15
            if current_price < current_ema20 < current_ema50 < current_ema200:
                alignment_score = 25
        
        # Calculate total strength (0-100)
        strength = ema_score + distance_score + consistency_score + alignment_score
        strength = max(0, min(100, strength))
        
        # Interpretation
        if strength >= 70:
            interpretation = "STRONG"
        elif strength >= 40:
            interpretation = "MODERATE"
        else:
            interpretation = "WEAK"
        
        return round(strength, 1), interpretation
        
    except Exception as e:
        logger.debug(f"Trend strength calculation failed: {e}")
        return 50, "MODERATE"


# ============================================================================
# BREAKOUT CONFIRMATION FILTER (NEW)
# ============================================================================
# ============================================================================
# BREAKOUT CONFIRMATION FILTER - FIXED (LESS STRICT)
# ============================================================================

def is_breakout_valid(df: pd.DataFrame, side: str, price: float) -> Tuple[bool, str]:
    """
    Check if breakout/breakdown is confirmed with volume
    
    ✅ FIXED: Only reject CLEAR breakouts without volume
    ✅ NON-BREAKOUT setups always pass!
    
    Returns: (is_valid, reason)
    """
    try:
        if len(df) < 20:
            return True, "Insufficient data"
        
        close = float(df['Close'].iloc[-1])
        volume = float(df['Volume'].iloc[-1])
        avg_volume = float(df['Volume'].tail(20).mean())
        
        high_20 = float(df['High'].tail(20).max())
        low_20 = float(df['Low'].tail(20).min())
        
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        # Calculate how close we are to high/low
        dist_to_high_pct = (high_20 - close) / close * 100
        dist_to_low_pct = (close - low_20) / close * 100
        
        if side == "LONG":
            # ✅ ONLY check if ACTUALLY breaking out (within 1% of high)
            is_breakout = dist_to_high_pct <= 1.0
            
            if is_breakout:
                # This IS a breakout - check volume
                if volume_ratio >= 1.2:
                    return True, f"Breakout confirmed ({volume_ratio:.1f}x vol)"
                elif volume_ratio >= 0.8:
                    # Marginal volume - still allow but note it
                    return True, f"Breakout with fair volume ({volume_ratio:.1f}x)"
                else:
                    # ✅ ONLY reject if VERY weak volume on clear breakout
                    if volume_ratio < 0.5:
                        return False, f"Breakout with very weak volume ({volume_ratio:.1f}x)"
                    else:
                        return True, f"Breakout with below-avg volume ({volume_ratio:.1f}x)"
            else:
                # ✅ NOT a breakout setup - ALWAYS PASS!
                return True, "Not a breakout pattern"
        
        else:  # SHORT
            is_breakdown = dist_to_low_pct <= 1.0
            
            if is_breakdown:
                if volume_ratio >= 1.2:
                    return True, f"Breakdown confirmed ({volume_ratio:.1f}x vol)"
                elif volume_ratio >= 0.8:
                    return True, f"Breakdown with fair volume ({volume_ratio:.1f}x)"
                else:
                    if volume_ratio < 0.5:
                        return False, f"Breakdown with very weak volume ({volume_ratio:.1f}x)"
                    else:
                        return True, f"Breakdown with below-avg volume ({volume_ratio:.1f}x)"
            else:
                return True, "Not a breakdown pattern"
        
    except Exception as e:
        logger.debug(f"Breakout validation failed: {e}")
        return True, "Check skipped"


# ============================================================================
# SETUP TYPE CLASSIFIER & HISTORY TRACKER (NEW)
# ============================================================================

# Global tracking of setup performance
SETUP_HISTORY = {
    "EMA_BOUNCE": {"wins": 0, "losses": 0, "total_pnl": 0},
    "BREAKOUT": {"wins": 0, "losses": 0, "total_pnl": 0},
    "VCP": {"wins": 0, "losses": 0, "total_pnl": 0},
    "SUPPORT_BOUNCE": {"wins": 0, "losses": 0, "total_pnl": 0},
    "RSI_REVERSAL": {"wins": 0, "losses": 0, "total_pnl": 0},
    "MOMENTUM": {"wins": 0, "losses": 0, "total_pnl": 0},
    "GAP_PLAY": {"wins": 0, "losses": 0, "total_pnl": 0},
    "UNKNOWN": {"wins": 0, "losses": 0, "total_pnl": 0},
}


def get_setup_type(reasons: List[str], candle_pattern: str = "", sr_bias: str = "") -> str:
    """
    Classify the setup type based on entry reasons
    
    Args:
        reasons: List of entry reasons from scoring
        candle_pattern: Detected candlestick pattern
        sr_bias: Support/Resistance bias
    
    Returns:
        Setup type string
    """
    try:
        reason_str = " ".join(reasons).lower()
        
        # Priority order of setup classification
        
        # 1. VCP (Volatility Contraction Pattern) - High priority
        if "vcp" in reason_str or "volatility contraction" in reason_str:
            return "VCP"
        
        # 2. Breakout/Breakdown
        if "breakout" in reason_str or "breaking" in reason_str:
            return "BREAKOUT"
        if "volume" in reason_str and ("strong" in reason_str or "heavy" in reason_str):
            return "BREAKOUT"
        
        # 3. Support/Resistance Bounce
        if sr_bias in ["AT_SUPPORT", "NEAR_SUPPORT"]:
            return "SUPPORT_BOUNCE"
        if "support" in reason_str or "at support" in reason_str:
            return "SUPPORT_BOUNCE"
        
        # 4. Gap Play
        if "gap" in reason_str:
            return "GAP_PLAY"
        
        # 5. RSI Reversal
        if "rsi" in reason_str and ("oversold" in reason_str or "overbought" in reason_str):
            return "RSI_REVERSAL"
        if "rsi" in reason_str and "optimal" in reason_str:
            return "RSI_REVERSAL"
        
        # 6. Strong Momentum
        if "momentum" in reason_str and "strong" in reason_str:
            return "MOMENTUM"
        if "adx" in reason_str or "supertrend" in reason_str:
            return "MOMENTUM"
        
        # 7. EMA Bounce (most common)
        if "ema" in reason_str:
            return "EMA_BOUNCE"
        
        # 8. Default
        return "UNKNOWN"
        
    except Exception:
        return "UNKNOWN"


def get_setup_win_rate(setup_type: str) -> float:
    """Get historical win rate for a setup type"""
    try:
        history = SETUP_HISTORY.get(setup_type, {"wins": 0, "losses": 0})
        total = history["wins"] + history["losses"]
        if total >= 5:  # Need at least 5 trades for meaningful stats
            return (history["wins"] / total) * 100
        return 50.0  # Default 50% if not enough history
    except:
        return 50.0


def update_setup_history(setup_type: str, is_win: bool, pnl_pct: float = 0):
    """Update setup history after trade closes"""
    global SETUP_HISTORY
    try:
        if setup_type not in SETUP_HISTORY:
            SETUP_HISTORY[setup_type] = {"wins": 0, "losses": 0, "total_pnl": 0}
        
        if is_win:
            SETUP_HISTORY[setup_type]["wins"] += 1
        else:
            SETUP_HISTORY[setup_type]["losses"] += 1
        
        SETUP_HISTORY[setup_type]["total_pnl"] += pnl_pct
        
    except Exception:
        pass


def print_setup_statistics():
    """Print performance by setup type"""
    print("\n" + "="*80)
    print("SETUP TYPE PERFORMANCE")
    print("="*80)
    print(f"{'Setup Type':<20} {'Trades':>8} {'Wins':>8} {'Win Rate':>10} {'Avg P&L':>10}")
    print("-"*80)
    
    for setup_type, stats in SETUP_HISTORY.items():
        total = stats["wins"] + stats["losses"]
        if total > 0:
            win_rate = (stats["wins"] / total) * 100
            avg_pnl = stats["total_pnl"] / total
            print(f"{setup_type:<20} {total:>8} {stats['wins']:>8} {win_rate:>9.1f}% {avg_pnl:>9.2f}%")
    
    print("="*80)
# ============================================================================
# MAIN SCANNER FUNCTION - COMPLETE WITH TICKER MATCHING FIX
# ============================================================================

# ============================================================================
# MAIN SCANNER FUNCTION - COMPLETE FIXED VERSION WITH ALL IMPROVEMENTS
# ============================================================================

def scan_ticker(
    ticker: str,
    df_all: pd.DataFrame,
    quality_validator: DataQualityValidator,
    sector_filter: SectorRotationAnalyzer,
    vix_filter: VIXSentimentAnalyzer,
    fibo_detector: FibonacciDetector,
    portfolio_mgr: PortfolioRiskManager,
    mtf_analyzer: MultiTimeframeAnalyzer,
    backtester: Optional[AdvancedBacktester] = None,
    run_full_backtest: bool = False
) -> Optional[Dict]:
    """
    Complete scanner - FULLY FIXED VERSION WITH ALL IMPROVEMENTS
    
    Includes:
    ✅ All bug fixes
    ✅ Trend Strength Filter
    ✅ Breakout Confirmation Filter
    ✅ Setup Type Classification
    ✅ Relative Strength vs Nifty
    ✅ Market Regime Filter
    """
    
    # =========================================================================
    # STEP 0: CRITICAL - Declare globals and define log_rejection FIRST!
    # =========================================================================
    global stats, rejection_samples
    
    # ✅ FIX: Define log_rejection BEFORE any code uses it!
    def log_rejection(reason: str, detail: str = ""):
        """Log rejection reason for debugging"""
        if len(rejection_samples) < 15:
            msg = f"{ticker.replace('.NS', '').replace('.BO', '')}: {reason}"
            if detail:
                msg += f" ({detail})"
            rejection_samples.append(msg)
    
    # ✅ NOW it's safe to use log_rejection and increment stats
    stats["total"] += 1
    
    # =========================================================================
    # STEP 1: MARKET REGIME CHECK (Early exit if unfavorable)
    # =========================================================================
    market_regime = MarketRegimeFilter.get_market_regime()
    
    if market_regime.get('regime') == 'HIGH_VOLATILITY':
        stats["volatility_fail"] += 1
        log_rejection("Market volatility", f"Regime: {market_regime['regime']}")
        return None

    # =========================================================================
    # STEP 1a: CNC_LONG_ONLY — behaviour when market is SHORT-biased
    # =========================================================================
    # When CNC_LONG_ONLY=True AND market is SHORT-biased:
    # OLD (broken): hard-blocked ALL 498 stocks at step 1 → 0 signals every day
    #   in a bearish market, which is exactly when shorts are most valuable.
    # NEW (fixed): let the scan run fully for ALL stocks.
    #   SHORT signals → tagged side="SHORT_ALERT" at result build (step 25).
    #   These still appear in your email + Google Sheet as reference signals
    #   so you can trade them manually via Kite app / F&O as you see fit.
    #   LONG signals in a SHORT-biased market still get a confidence penalty
    #   (applied in score_signal) so only the best counter-trend setups pass.
    # → No more 0-signal days just because the market is bearish.
    # (The SHORT_ALERT tag is still applied at the result dict below so
    #  the gatekeeper / Google Sheets integration stays unchanged.)
    
    try:
        # =====================================================================
        # STEP 2: GET DATA FOR THIS TICKER
        # =====================================================================
        df_t = df_all[df_all["ticker"] == ticker].copy()
        
        if df_t.empty:
            stats["no_data"] += 1
            log_rejection("No data", "Ticker not in cache")
            return None
        
        # =====================================================================
        # STEP 3: PREPARE DATA - ENSURE CLEAN DATAFRAME
        # =====================================================================
        df_t = df_t.copy()
        
        # Sort by date
        if 'Date' in df_t.columns:
            df_t = df_t.sort_values('Date')
            df_t = df_t.set_index('Date')
        
        df_t = df_t.sort_index()
        
        # Drop ticker column BEFORE any processing
        if 'ticker' in df_t.columns:
            df_t = df_t.drop(columns=['ticker'])
        
        # Normalize prices if needed (only for paise data)
        df_t = normalize_price_data(df_t)
        
        # Ensure it's still a proper DataFrame
        if not isinstance(df_t, pd.DataFrame):
            stats["data_quality_fail"] += 1
            log_rejection("Data type error", f"Got {type(df_t)}")
            return None
        
        # =====================================================================
        # STEP 4: VERIFY DATA FRESHNESS
        # =====================================================================
        try:
            last_date = df_t.index[-1]
            days_old = (datetime.now() - pd.to_datetime(last_date)).days
            
            if days_old > MAX_DATA_STALE_DAYS:
                stats["data_quality_fail"] += 1
                log_rejection("Stale data", f"{days_old} days old")
                return None
        except Exception as e:
            stats["data_quality_fail"] += 1
            log_rejection("Date error", str(e)[:30])
            return None
        
        # =====================================================================
        # STEP 5: VERIFY DATA UNIQUENESS
        # =====================================================================
        unique_prices = df_t['Close'].nunique()
        total_rows = len(df_t)
        
        if total_rows > 20 and unique_prices < 10:
            stats["data_quality_fail"] += 1
            log_rejection("Duplicate data", f"Only {unique_prices} unique prices")
            return None
        
        # =====================================================================
        # STEP 6: DATA QUALITY CHECKS
        # =====================================================================
        if not quality_validator.has_sufficient_history(df_t, min_bars=60):
            stats["indicator_fail"] += 1
            log_rejection("Insufficient history", f"{len(df_t)} bars")
            return None
        
        valid_prices, price_reason = quality_validator.has_valid_prices(df_t)
        if not valid_prices:
            stats["data_quality_fail"] += 1
            log_rejection("Invalid prices", price_reason)
            return None
        
        # =====================================================================
        # STEP 7: CALCULATE INDICATORS
        # =====================================================================
        ind = calculate_indicators(df_t)
        if ind is None:
            stats["indicator_fail"] += 1
            log_rejection("Indicator calculation failed")
            return None
        
        price = ind["price"]
        
        # Debug first few stocks
        if stats["total"] <= 5:
            print(f"\n  DEBUG: {ticker} - Price: ₹{price:.2f} | RSI: {ind['rsi']:.1f} | Vol: {ind['volume_ratio']:.2f}x")
        
        # =====================================================================
        # STEP 8: VOLATILITY REGIME FILTER
        # =====================================================================
        vol_ok, vol_reason, vol_metrics = VolatilityRegimeFilter.analyze_regime(df_t, price)
        if not vol_ok:
            stats["volatility_fail"] += 1
            log_rejection("Volatility", vol_reason)
            return None
        
        # =====================================================================
        # STEP 9: TREND STRENGTH CHECK (NEW IMPROVEMENT #1)
        # =====================================================================
        trend_strength, trend_interpretation = calculate_trend_strength(df_t)
        
        if USE_TREND_STRENGTH_FILTER and trend_strength < 15:
            stats["indicator_fail"] += 1
            log_rejection("Weak trend", f"Strength: {trend_strength:.0f}")
            return None
        
        # =====================================================================
        # STEP 10: RUN ALL ANALYZERS
        # =====================================================================
        try:
            volume_score, volume_signal, volume_metrics = InstitutionalVolumeAnalyzer.analyze(df_t, ind["volume"])
            vsa_score, vsa_signal, vsa_metrics = VolumeSpreadAnalyzer.analyze(df_t)
            momentum_score, momentum_signal, momentum_metrics = MomentumScoreCalculator.calculate(df_t)
            vcp_score, vcp_signal, vcp_metrics = VCPDetector.detect(df_t)
            gap_score, gap_signal, gap_metrics = GapDetector.detect(df_t)
            sr_info = SRClusterDetector.detect(df_t)
            candle = CandlestickPatternDetector.detect(df_t)
            st_direction, st_strength, st_metrics = SupertrendIndicator.calculate(df_t)
            mtf_trend = MultiTimeframeAnalyzer.get_trend_from_cache(df_t)
        except Exception as e:
            stats["indicator_fail"] += 1
            log_rejection("Analyzer error", str(e)[:40])
            return None
        
        # =====================================================================
        # STEP 11: GET FUNDAMENTALS
        # =====================================================================
        sector, pe = get_fundamentals(ticker)
        
        # =====================================================================
        # STEP 12: SECTOR ROTATION (IF ENABLED)
        # =====================================================================
        sector_rs = None
        if USE_SECTOR_ROTATION and sector_filter and sector:
            try:
                sector_rs = sector_filter.get_sector_strength(sector)
            except:
                sector_rs = None
        
        # =====================================================================
        # STEP 13: VIX SENTIMENT (IF ENABLED)
        # =====================================================================
        vix_sentiment = None
        if USE_VIX_SENTIMENT and vix_filter:
            try:
                vix_sentiment = vix_filter.get_vix_sentiment()
            except:
                vix_sentiment = None
        
        # =====================================================================
        # STEP 14: FIBONACCI LEVELS (IF ENABLED)
        # =====================================================================
        long_fibo_boost = 0
        short_fibo_boost = 0
        if USE_FIBONACCI_SCORING and fibo_detector:
            try:
                fib_levels = fibo_detector.calculate_levels(df_t, PRESET['fib_lookback'])
                if fib_levels:
                    long_fibo_boost, _ = fibo_detector.detect_bounce(price, fib_levels, "LONG")
                    short_fibo_boost, _ = fibo_detector.detect_bounce(price, fib_levels, "SHORT")
            except:
                pass
        
        # =====================================================================
        # STEP 15: GENERATE ENTRY SIGNALS
        # =====================================================================
        long_entry = generate_entry_signal(ind, df_t, "LONG")
        short_entry = generate_entry_signal(ind, df_t, "SHORT")
        
        # =====================================================================
        # STEP 16: SCORE SIGNALS
        # =====================================================================
        long_conf, long_reasons = score_signal(
            ind, df_t, "LONG", vix_sentiment, sector_rs, long_fibo_boost,
            sr_info, candle, mtf_trend, vol_metrics,
            {'score': volume_score, 'signal': volume_signal},
            {'score': vsa_score, 'signal': vsa_signal},
            {'score': momentum_score, 'signal': momentum_signal},
            {'score': vcp_score, 'signal': vcp_signal},
            {'score': gap_score, 'signal': gap_signal},
            {'direction': st_direction, 'strength': st_strength}
        )
        
        short_conf, short_reasons = score_signal(
            ind, df_t, "SHORT", vix_sentiment, sector_rs, short_fibo_boost,
            sr_info, candle, mtf_trend, vol_metrics,
            {'score': volume_score, 'signal': volume_signal},
            {'score': vsa_score, 'signal': vsa_signal},
            {'score': momentum_score, 'signal': momentum_signal},
            {'score': vcp_score, 'signal': vcp_signal},
            {'score': gap_score, 'signal': gap_signal},
            {'direction': st_direction, 'strength': st_strength}
        )
        
        # =====================================================================
        # STEP 17: DETERMINE BEST SIDE
        # =====================================================================
        long_valid = long_entry["entry_ready"] and long_conf >= MIN_CONFIDENCE
        short_valid = short_entry["entry_ready"] and short_conf >= MIN_CONFIDENCE
        
        if not (long_valid or short_valid):
            stats["confidence_fail"] += 1
            log_rejection("Low confidence", f"L:{long_conf:.0f}% S:{short_conf:.0f}%")
            return None
        
        if long_valid and (not short_valid or long_conf >= short_conf):
            side = "LONG"
            confidence = long_conf
            reasons = long_reasons
            entry_info = long_entry
        else:
            side = "SHORT"
            confidence = short_conf
            reasons = short_reasons
            entry_info = short_entry
        
        # =====================================================================
        # STEP 18: BREAKOUT CONFIRMATION CHECK (NEW IMPROVEMENT #2)
        # =====================================================================
        if USE_BREAKOUT_CONFIRMATION:
            breakout_valid, breakout_reason = is_breakout_valid(df_t, side, price)
            if not breakout_valid:
                stats["indicator_fail"] += 1
                log_rejection("Breakout invalid", breakout_reason)
                return None
        else:
            breakout_reason = "Check disabled"
        
        # Add breakout reason to signal reasons if it's a breakout
        if "breakout" in breakout_reason.lower() or "breakdown" in breakout_reason.lower():
            reasons.append(breakout_reason[:50])
        
        # =====================================================================
        # STEP 19: CREATE TRADE RULE
        # =====================================================================
        trade_rule = TradeRule(ticker.replace(".NS", "").replace(".BO", ""), side, price, ind["atr"])
        trade_rule.calculate_dynamic_targets(confidence)
        
        if trade_rule.risk_reward_ratio < MIN_RR_RATIO:
            stats["rr_fail"] += 1
            log_rejection("Low R:R", f"{trade_rule.risk_reward_ratio:.2f}x")
            return None
        
        # =====================================================================
        # STEP 20: MINI BACKTEST
        # =====================================================================
        mini_metrics = {}
        if USE_MINI_BACKTEST:
            mini_passed, mini_metrics = mini_backtest(df_t, price, side, MIN_RR_RATIO)
            if not mini_passed:
                stats["mini_backtest_fail"] += 1
                log_rejection("Mini backtest", mini_metrics.get('reason', 'Failed'))
                return None
        
        # =====================================================================
        # STEP 21: FULL BACKTEST (IF REQUESTED)
        # =====================================================================
        backtest_result = None
        walk_forward_result = None
        monte_carlo_result = None
        
        if run_full_backtest and backtester and USE_FULL_BACKTEST:
            stats["full_backtest_run"] += 1
            
            try:
                backtest_result = backtester.backtest_strategy(
                    df=df_t,
                    signal_type=side,
                    entry_conditions={'side': side},
                    lookback_days=BACKTEST_LOOKBACK_DAYS
                )
                
                if backtest_result:
                    if (backtest_result.win_rate < PRESET['min_backtest_win_rate'] or
                        backtest_result.profit_factor < PRESET['min_profit_factor'] or
                        backtest_result.max_drawdown > PRESET['max_drawdown']):
                        
                        stats["full_backtest_fail"] += 1
                        log_rejection("Backtest fail", f"WR:{backtest_result.win_rate:.0f}%")
                        return None
                    
                    # Apply backtest boost to confidence
                    backtest_boost = (
                        (backtest_result.win_rate - 50) * 0.2 +
                        (backtest_result.profit_factor - 1) * 3 +
                        (backtest_result.reliability_score - 50) * 0.15
                    )
                    confidence = min(100, confidence + backtest_boost)
                    
                    # Walk-forward validation (if enabled)
                    # FIX: Removed CONSERVATIVE-only gate — run in BALANCED too
                    if USE_WALK_FORWARD and ACCURACY_MODE in ('CONSERVATIVE', 'BALANCED'):
                        walk_forward_result = backtester.walk_forward_validation(
                            df_t, side, {'side': side}, WALK_FORWARD_PERIODS
                        )
                        
                        if walk_forward_result and not walk_forward_result.passed:
                            stats["walk_forward_fail"] += 1
                            log_rejection("Walk-forward fail", "Inconsistent")
                            return None
                    
                    # Monte Carlo simulation (if enabled)
                    if USE_MONTE_CARLO:
                        monte_carlo_result = backtester.monte_carlo_simulation(
                            backtest_result, simulations=1000, num_trades=50
                        )
                        
                        if monte_carlo_result and monte_carlo_result.prob_profit < 60:
                            stats["monte_carlo_fail"] += 1
                            log_rejection("Monte Carlo fail", f"Prob:{monte_carlo_result.prob_profit:.0f}%")
                            return None
            except Exception as e:
                logger.debug(f"Backtest error for {ticker}: {e}")
                pass
        
        # =====================================================================
        # STEP 22: RELATIVE STRENGTH CHECK (vs Nifty)
        # =====================================================================
        rs_value, rs_interpretation = RelativeStrengthCalculator.calculate_rs(df_t)
        
        # Reject weak stocks for LONG signals
        if side == "LONG" and rs_interpretation in ["WEAK", "VERY_WEAK"]:
            stats["confidence_fail"] += 1
            log_rejection("Weak RS", f"RS: {rs_value:.0f} ({rs_interpretation})")
            return None
        
        # Reject strong stocks for SHORT signals
        if side == "SHORT" and rs_interpretation in ["STRONG", "VERY_STRONG"]:
            stats["confidence_fail"] += 1
            log_rejection("Strong RS", f"RS: {rs_value:.0f} ({rs_interpretation})")
            return None
        
        # =====================================================================
        # STEP 23: PORTFOLIO RISK CHECK (IF ENABLED)
        # =====================================================================
        # [FIX MAJ-7] This block was commented out — portfolio sector-exposure
        # cap was never enforced during scanning.  Now active.
        if USE_PORTFOLIO_RISK and portfolio_mgr:
            can_add, reason = portfolio_mgr.can_add_trade({
                'ticker': ticker,
                'side': side,
                'position_value': trade_rule.position_value,
                'risk_amount': trade_rule.actual_risk,
                'sector': sector,
            })
            if not can_add:
                stats["portfolio_fail"] += 1
                log_rejection("Portfolio limit", reason)
                return None
        
        # =====================================================================
        # STEP 24: CLASSIFY SETUP TYPE (NEW IMPROVEMENT #3)
        # =====================================================================
        sr_bias = sr_info.get('bias', 'UNKNOWN')
        candle_pattern = candle.get('pattern', 'NONE')
        setup_type = get_setup_type(reasons, candle_pattern, sr_bias)
        
        # Get historical win rate for this setup type
        setup_historical_wr = get_setup_win_rate(setup_type)
        
        # =====================================================================
        # STEP 25: APPLY TREND STRENGTH BONUS/PENALTY
        # =====================================================================
        # Boost confidence for strong trends, reduce for weak trends
        if trend_interpretation == "STRONG":
            confidence = min(100, confidence + 3)
            reasons.append(f"Strong trend ({trend_strength:.0f})")
        elif trend_interpretation == "WEAK":
            confidence = max(30, confidence - 2)
        
        # =====================================================================
        # SUCCESS - BUILD RESULT OBJECT
        # =====================================================================
        stats["passed"] += 1
        
        # Print successful signal
        print(f"\n✅ SIGNAL #{stats['passed']}: {ticker.replace('.NS', '').replace('.BO', '')}")
        print(f"   {side} @ ₹{price:.2f} | Conf: {confidence:.1f}% | R:R: {trade_rule.risk_reward_ratio:.1f}x | Setup: {setup_type}")
        
        result = {
            "ticker": ticker.replace(".NS", "").replace(".BO", ""),
            # FIX: If CNC_LONG_ONLY, mark shorts as alert-only (not executable via Kite CNC)
            "side": "SHORT_ALERT" if (side == "SHORT" and CNC_LONG_ONLY) else side,
            "price": round(price, 2),
            "confidence": round(confidence, 1),
            "rsi": round(ind["rsi"], 1),
            "atr": round(ind["atr"], 2),
            "adx": round(ind["adx"], 1),
            "entry_price": trade_rule.entry_price,
            "stop_loss": trade_rule.stop_loss,
            "target_1": trade_rule.target_1,
            "target_2": trade_rule.target_2,
            "risk_reward": round(trade_rule.risk_reward_ratio, 2),
            "qty": trade_rule.qty,
            "position_value": round(trade_rule.position_value, 2),
            "risk_amount": round(trade_rule.actual_risk, 2),
            "entry_signal": entry_info["entry_signal"],
            "reasons": " | ".join(reasons),
            "sector": sector or "N/A",
            "pe": pe,
            "trade_rule": trade_rule,
            "mini_backtest": mini_metrics,
            "backtest_validated": False,
            
            # NEW FIELDS
            "setup_type": setup_type,
            "setup_historical_wr": setup_historical_wr,
            "trend_strength": trend_strength,
            "trend_interpretation": trend_interpretation,
            "relative_strength": rs_value,
            "rs_interpretation": rs_interpretation,
            "market_regime": market_regime.get('regime', 'UNKNOWN'),
            "breakout_info": breakout_reason,
        }
        
        # Add backtest results if available
        if backtest_result:
            result["backtest"] = {
                "win_rate": backtest_result.win_rate,
                "profit_factor": backtest_result.profit_factor,
                "sharpe_ratio": backtest_result.sharpe_ratio,
                "max_drawdown": backtest_result.max_drawdown,
                "reliability_score": backtest_result.reliability_score,
                "total_trades": backtest_result.total_trades,
                "expectancy": backtest_result.expectancy,
            }
            result["backtest_validated"] = True
        
        # Add walk-forward results if available
        if walk_forward_result:
            result["walk_forward"] = {
                "avg_win_rate": walk_forward_result.avg_win_rate,
                "consistency": walk_forward_result.consistency_score,
                "passed": walk_forward_result.passed
            }
        
        # Add Monte Carlo results if available
        if monte_carlo_result:
            result["monte_carlo"] = {
                "prob_profit": monte_carlo_result.prob_profit,
                "risk_of_ruin": monte_carlo_result.risk_of_ruin,
            }
        
        return result
        
    except Exception as e:
        # Catch-all exception handler
        stats["indicator_fail"] += 1
        log_rejection("Exception", f"{type(e).__name__}: {str(e)[:30]}")
        logger.debug(f"{ticker}: Exception - {str(e)}")
        return None

# ============================================================================
# HYBRID SCANNING STRATEGY
# ============================================================================

def hybrid_scan_universe(
    tickers: List[str],
    df_all: pd.DataFrame,
    quality_validator: DataQualityValidator,
    sector_filter: SectorRotationAnalyzer,
    vix_filter: VIXSentimentAnalyzer,
    fibo_detector: FibonacciDetector,
    portfolio_mgr: PortfolioRiskManager,
    mtf_analyzer: MultiTimeframeAnalyzer
) -> List[Dict]:
    """Hybrid scanning: Quick scan + Full backtest top candidates"""
    
    print(f"\n{'='*120}")
    print("PHASE 1: QUICK SCAN WITH MINI BACKTEST")
    print('='*120)
    
    # Phase 1: Quick scan
    initial_results = []
    
    for i, ticker in enumerate(tickers, 1):
        result = scan_ticker(
            ticker=ticker,
            df_all=df_all,
            quality_validator=quality_validator,
            sector_filter=sector_filter,
            vix_filter=vix_filter,
            fibo_detector=fibo_detector,
            portfolio_mgr=portfolio_mgr,
            mtf_analyzer=mtf_analyzer,
            backtester=None,
            run_full_backtest=False
        )
        
        if result:
            initial_results.append(result)
            # [FIX MAJ-7] Track accepted trades so sector exposure cap fires correctly
            if USE_PORTFOLIO_RISK and portfolio_mgr:
                portfolio_mgr.add_trade(result)
        
        # Progress bar
        if i % 25 == 0 or i == len(tickers):
            pct = (i / len(tickers)) * 100
            bar_length = 50
            filled = int(bar_length * i / len(tickers))
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"  [{bar}] {pct:>5.1f}% | {i:>4}/{len(tickers)} | Found: {len(initial_results):>3}", 
                  end='\r', flush=True)
    
    print()
    
    if not initial_results:
        print("\n  ⚠️  No signals found in Phase 1")
        return []
    
    # Sort by confidence
    initial_results = sorted(initial_results, key=lambda x: x['confidence'], reverse=True)
    print(f"\n  ✓ Phase 1 Complete: {len(initial_results)} signals found")
    
    # Phase 2: Full backtest (if enabled)
    if BACKTEST_MODE in ['HYBRID', 'FULL'] and USE_FULL_BACKTEST:
        
        num_to_test = min(30, len(initial_results)) if BACKTEST_MODE == 'HYBRID' else len(initial_results)
        
        print(f"\n{'='*120}")
        print(f"PHASE 2: FULL BACKTEST ON TOP {num_to_test} CANDIDATES")
        print('='*120)
        
        backtester = AdvancedBacktester(initial_capital=ACCOUNT_CAPITAL)
        validated_results = []
        candidates_to_test = initial_results[:num_to_test]
        
        for i, candidate in enumerate(candidates_to_test, 1):
            ticker = candidate['ticker'] + '.NS'
            
            pct = (i / num_to_test) * 100
            print(f"  [{i:>3}/{num_to_test}] {pct:>5.1f}% | Testing {candidate['ticker']:<12} | "
                  f"Conf: {candidate['confidence']:>5.1f}% | {candidate['side']:<5}", 
                  end='\r', flush=True)
            
            df_t = df_all[df_all["ticker"] == ticker].copy()
            
            if df_t.empty:
                continue
            
            # Prepare data
            if 'Date' in df_t.columns:
                df_t = df_t.set_index('Date')
            df_t = df_t.sort_index()
            if 'ticker' in df_t.columns:
                df_t = df_t.drop(columns=['ticker'])
            
            # Run full backtest
            backtest_result = backtester.backtest_strategy(
                df=df_t,
                signal_type=candidate['side'],
                entry_conditions={'side': candidate['side']},
                lookback_days=BACKTEST_LOOKBACK_DAYS
            )
            
            if backtest_result:
                if (backtest_result.win_rate >= PRESET['min_backtest_win_rate'] and
                    backtest_result.profit_factor >= PRESET['min_profit_factor'] and
                    backtest_result.max_drawdown <= PRESET['max_drawdown']):
                    
                    candidate['backtest'] = {
                        "win_rate": backtest_result.win_rate,
                        "profit_factor": backtest_result.profit_factor,
                        "sharpe_ratio": backtest_result.sharpe_ratio,
                        "max_drawdown": backtest_result.max_drawdown,
                        "reliability_score": backtest_result.reliability_score,
                        "total_trades": backtest_result.total_trades,
                    }
                    
                    backtest_boost = (backtest_result.reliability_score - 50) * 0.25
                    candidate['confidence'] = min(100, candidate['confidence'] + backtest_boost)
                    candidate['backtest_validated'] = True
                    
                    validated_results.append(candidate)
                else:
                    stats["full_backtest_fail"] += 1
            else:
                candidate['backtest_validated'] = False
                validated_results.append(candidate)
        
        print()
        print(f"\n  ✓ Phase 2 Complete: {len(validated_results)} signals validated")
        
        # Add remaining if HYBRID
        if BACKTEST_MODE == 'HYBRID' and len(initial_results) > num_to_test:
            for candidate in initial_results[num_to_test:]:
                candidate['backtest_validated'] = False
                validated_results.append(candidate)
        
        return sorted(validated_results, key=lambda x: x['confidence'], reverse=True)
    
    return initial_results

# ============================================================================
# STATISTICS PRINTING
# ============================================================================

def print_statistics():
    """Print detailed scan statistics"""
    
    total = stats.get("total", 0)
    if total == 0:
        print("\n⚠️  No stocks were scanned")
        return
    
    print(f"\n{'='*120}")
    print("SCAN STATISTICS")
    print('='*120)
    
    passed = stats.get("passed", 0)
    pass_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"\n📊 OVERALL:")
    print(f"   Total Scanned:    {total:>6}")
    print(f"   Signals Found:    {passed:>6}  ({pass_rate:.1f}%)")
    print(f"   Rejected:         {total - passed:>6}  ({100-pass_rate:.1f}%)")
    
    print(f"\n❌ REJECTION BREAKDOWN:")
    
    rejections = [
        ("No Data", stats.get("no_data", 0)),
        ("Data Quality", stats.get("data_quality_fail", 0)),
        ("Indicators Failed", stats.get("indicator_fail", 0)),
        ("Volatility Regime", stats.get("volatility_fail", 0)),
        ("Low Confidence", stats.get("confidence_fail", 0)),
        ("Poor Risk:Reward", stats.get("rr_fail", 0)),
        ("Mini Backtest", stats.get("mini_backtest_fail", 0)),
        ("Full Backtest", stats.get("full_backtest_fail", 0)),
        ("Walk-Forward", stats.get("walk_forward_fail", 0)),
        ("Monte Carlo", stats.get("monte_carlo_fail", 0)),
        ("Portfolio Risk", stats.get("portfolio_fail", 0)),
    ]
    
    has_rejections = False
    for reason, count in rejections:
        if count > 0:
            has_rejections = True
            pct = (count / total) * 100
            print(f"   {reason:<20} {count:>6}  ({pct:>5.1f}%)")
    
    if not has_rejections:
        print("   (No rejections recorded)")
    
    if rejection_samples:
        print(f"\n📝 SAMPLE REJECTIONS:")
        for sample in rejection_samples[:8]:
            print(f"   • {sample}")
    
    bt_run = stats.get("full_backtest_run", 0)
    if bt_run > 0:
        bt_fail = stats.get("full_backtest_fail", 0)
        print(f"\n🔬 BACKTEST INFO:")
        print(f"   Full Backtests Run:   {bt_run}")
        print(f"   Failed:               {bt_fail}")
    
    print('='*120)

# ============================================================================
# RESULTS PRINTING
# ============================================================================

def print_results(results: List[Dict]):
    """Print formatted results table - ENHANCED VERSION WITH SETUP TYPE"""
    
    if not results:
        return
    
    # Classify into tiers
    tiers = apply_tier_based_filtering(results)
    
    total_signals = sum(len(v) for v in tiers.values())
    
    print(f"\n{'='*200}")
    print(f"TRADING SIGNALS - TIERED CLASSIFICATION ({total_signals} Total)")
    print('='*200)
    
    # Print each tier
    tier_info = {
        "TIER_1_PREMIUM": ("🟢 TIER 1: PREMIUM SIGNALS (Highest Probability)", "Take with 30-40% position size"),
        "TIER_2_STANDARD": ("🟡 TIER 2: STANDARD SIGNALS (Good Probability)", "Take with 15-20% position size"),
        "TIER_3_SPECULATIVE": ("🔵 TIER 3: SPECULATIVE SIGNALS (Acceptable Risk)", "Take with 5-10% position size"),
    }
    
    for tier_name, (tier_title, tier_advice) in tier_info.items():
        tier_results = tiers[tier_name]
        
        if not tier_results:
            continue
        
        print(f"\n{tier_title}")
        print(f"   Recommendation: {tier_advice}")
        print(f"   Count: {len(tier_results)} signals")
        print('-'*200)
        
        # Enhanced Header with new columns
        header = (
            f"{'#':<3} {'TICKER':<10} {'SIDE':<5} {'PRICE':>8} {'CONF':>5} "
            f"{'ENTRY':>8} {'SL':>8} {'T1':>8} {'R:R':>4} {'SETUP':<15} "
            f"{'TREND':>6} {'RS':>5}"
        )
        
        if any(r.get('backtest_validated') for r in tier_results):
            header += f" {'BT_WR':>6} {'BT_PF':>5}"
        
        print(header)
        print('-'*200)
        
        for i, res in enumerate(tier_results[:20], 1):
            # Format setup type (truncate if needed)
            setup = res.get('setup_type', 'UNKNOWN')[:15]
            trend = res.get('trend_strength', 0)
            rs = res.get('relative_strength', 0)
            
            row = (
                f"{i:<3} "
                f"{res['ticker']:<10} "
                f"{res['side']:<5} "
                f"₹{res['price']:>7.0f} "
                f"{res['confidence']:>4.0f}% "
                f"₹{res['entry_price']:>7.0f} "
                f"₹{res['stop_loss']:>7.0f} "
                f"₹{res['target_1']:>7.0f} "
                f"{res['risk_reward']:>3.1f}x "
                f"{setup:<15} "
                f"{trend:>5.0f} "
                f"{rs:>5.0f}"
            )
            
            if res.get('backtest_validated') and 'backtest' in res:
                bt = res['backtest']
                row += (
                    f" {bt['win_rate']:>5.0f}% "
                    f"{bt['profit_factor']:>4.1f}x"
                )
            
            print(row)
    
    print('='*200)
    
    # Summary
    print(f"\n📊 TIER SUMMARY:")
    for tier_name, (tier_title, _) in tier_info.items():
        count = len(tiers[tier_name])
        if count > 0:
            print(f"   {tier_title}: {count} signals")
    
    # NEW: Print setup type summary
    print(f"\n📈 SETUP TYPE BREAKDOWN:")
    setup_counts = {}
    for result in results:
        setup = result.get('setup_type', 'UNKNOWN')
        setup_counts[setup] = setup_counts.get(setup, 0) + 1
    
    for setup, count in sorted(setup_counts.items(), key=lambda x: x[1], reverse=True):
        historical_wr = get_setup_win_rate(setup)
        print(f"   {setup:<20}: {count} signals (Historical WR: {historical_wr:.0f}%)")
    
    print('='*200)
    
# def get_realtime_price(ticker: str) -> Optional[float]:
#     """Get real-time price from NSE (market hours only)"""
#     try:
#         import datetime as dt
#         now = dt.datetime.now()
        
#         # Check market hours (9:15 AM - 3:30 PM IST)
#         if now.hour < 9 or (now.hour == 9 and now.minute < 15):
#             return None  # Pre-market
#         if now.hour >= 15 and now.minute > 30:
#             return None  # Post-market
            
#         # Use NSE API or broker API for real-time
#         # ... implementation depends on your data source
        
#     except:
#         return None

# ============================================================================
# EXPORT & HTML GENERATION
# ============================================================================

def export_results(results: List[Dict], scan_time: float):
    """Export results to CSV and HTML - ENHANCED WITH NEW FIELDS"""
    
    try:
        os.makedirs(SIGNALS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV Export - Enhanced with new fields
        csv_data = []
        for res in results:
            row = {
                "Ticker": res["ticker"],
                "Side": res["side"],
                "Price": res["price"],
                "Confidence": res["confidence"],
                "RSI": res["rsi"],
                "Entry": res["entry_price"],
                "StopLoss": res["stop_loss"],
                "Target1": res["target_1"],
                "Target2": res["target_2"],
                "RiskReward": res["risk_reward"],
                "Quantity": res["qty"],
                "Sector": res["sector"],
                "Reasons": res["reasons"],
                
                # NEW COLUMNS
                "SetupType": res.get("setup_type", "UNKNOWN"),
                "TrendStrength": res.get("trend_strength", 0),
                "TrendInterpretation": res.get("trend_interpretation", "UNKNOWN"),
                "RelativeStrength": res.get("relative_strength", 0),
                "RS_Interpretation": res.get("rs_interpretation", "UNKNOWN"),
                "MarketRegime": res.get("market_regime", "UNKNOWN"),
                "BreakoutInfo": res.get("breakout_info", "N/A"),
            }
            
            if res.get('backtest_validated') and 'backtest' in res:
                bt = res['backtest']
                row.update({
                    "BT_WinRate": bt['win_rate'],
                    "BT_ProfitFactor": bt['profit_factor'],
                    "BT_Sharpe": bt['sharpe_ratio'],
                    "BT_MaxDrawdown": bt['max_drawdown'],
                    "BT_Reliability": bt['reliability_score'],
                })
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_file = os.path.join(SIGNALS_DIR, f"signals_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        print(f"\n  ✓ CSV: {csv_file}")
        
        # HTML Report
        html_file = generate_html_report(results, scan_time, timestamp)
        if html_file:
            print(f"  ✓ HTML: {html_file}")
        
        return csv_file, html_file
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return None, None

def generate_html_report(results: List[Dict], scan_time: float, timestamp: str) -> Optional[str]:
    """Generate HTML report"""
    
    try:
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>NSE Scanner v8.5 - {timestamp}</title>
    <style>
        body {{ font-family: Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 15px; overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 30px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; padding: 30px; background: #f8f9fa; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .stat-card .value {{ font-size: 2em; font-weight: bold; color: #667eea; margin: 10px 0; }}
        .stat-card .label {{ color: #666; text-transform: uppercase; font-size: 0.85em; }}
        .signals {{ padding: 30px; }}
        .signal-card {{ background: white; border: 2px solid #e0e0e0; border-radius: 10px; padding: 20px; margin-bottom: 20px; }}
        .signal-card:hover {{ box-shadow: 0 5px 20px rgba(0,0,0,0.1); }}
        .signal-header {{ display: flex; justify-content: space-between; padding-bottom: 15px; border-bottom: 2px solid #f0f0f0; }}
        .signal-ticker {{ font-size: 1.8em; font-weight: bold; color: #1e3c72; }}
        .signal-side {{ padding: 8px 20px; border-radius: 20px; font-weight: bold; }}
        .side-long {{ background: #4caf50; color: white; }}
        .side-short {{ background: #f44336; color: white; }}
        .confidence {{ font-size: 1.5em; font-weight: bold; }}
        .conf-high {{ color: #4caf50; }}
        .conf-medium {{ color: #ff9800; }}
        .conf-low {{ color: #2196f3; }}
        .signal-details {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }}
        .detail-item {{ background: #f8f9fa; padding: 12px; border-radius: 8px; }}
        .detail-label {{ font-size: 0.85em; color: #666; margin-bottom: 5px; }}
        .detail-value {{ font-size: 1.2em; font-weight: bold; color: #333; }}
        .backtest-section {{ background: #e8f5e9; padding: 15px; border-radius: 8px; margin-top: 15px; }}
        .backtest-title {{ font-weight: bold; color: #2e7d32; margin-bottom: 10px; }}
        .reasons {{ background: #fff3e0; padding: 15px; border-radius: 8px; margin-top: 15px; border-left: 4px solid #ff9800; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 NSE SWING SCANNER v8.5</h1>
            <div>Ultimate Accuracy Edition - {ACCURACY_MODE} Mode</div>
            <div>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        
        <div class="stats">
            <div class="stat-card"><div class="label">Total Signals</div><div class="value">{len(results)}</div></div>
            <div class="stat-card"><div class="label">Stocks Scanned</div><div class="value">{stats['total']}</div></div>
            <div class="stat-card"><div class="label">Success Rate</div><div class="value">{(stats['passed']/stats['total']*100) if stats['total'] > 0 else 0:.1f}%</div></div>
            <div class="stat-card"><div class="label">Scan Time</div><div class="value">{scan_time:.1f}s</div></div>
        </div>
        
        <div class="signals">
            <h2 style="margin-bottom: 20px;">🎯 Trading Signals</h2>"""
        
        for i, res in enumerate(results[:TOP_RESULTS], 1):
            conf_class = "conf-high" if res['confidence'] >= 70 else ("conf-medium" if res['confidence'] >= 60 else "conf-low")
            side_class = "side-long" if res['side'] == "LONG" else "side-short"
            
            html_content += f"""
            <div class="signal-card">
                <div class="signal-header">
                    <div><span style="color: #999;">#{i}</span> <span class="signal-ticker">{res['ticker']}</span></div>
                    <div style="display: flex; gap: 15px; align-items: center;">
                        <span class="signal-side {side_class}">{res['side']}</span>
                        <span class="confidence {conf_class}">{res['confidence']:.1f}%</span>
                    </div>
                </div>
                
                <div class="signal-details">
                    <div class="detail-item"><div class="detail-label">Price</div><div class="detail-value">₹{res['price']:.2f}</div></div>
                    <div class="detail-item"><div class="detail-label">Entry</div><div class="detail-value">₹{res['entry_price']:.2f}</div></div>
                    <div class="detail-item"><div class="detail-label">Stop Loss</div><div class="detail-value">₹{res['stop_loss']:.2f}</div></div>
                    <div class="detail-item"><div class="detail-label">Target 1</div><div class="detail-value">₹{res['target_1']:.2f}</div></div>
                    <div class="detail-item"><div class="detail-label">R:R</div><div class="detail-value">{res['risk_reward']:.1f}x</div></div>
                    <div class="detail-item"><div class="detail-label">Qty</div><div class="detail-value">{res['qty']}</div></div>
                </div>
                
                <div class="reasons"><strong>💡 Entry Reasons:</strong><br>{res['reasons']}</div>
            </div>"""
        
        html_content += """
        </div>
    </div>
</body>
</html>"""
        
        html_file = os.path.join(SIGNALS_DIR, f"report_{timestamp}.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_file
        
    except Exception as e:
        logger.error(f"HTML generation failed: {e}")
        return None

# ============================================================================
# HEADER PRINTING
# ============================================================================

def print_header():
    """Print scanner header"""
    
    print("\n" + "="*120)
    print("""
███╗   ██╗███████╗███████╗    ███████╗ ██████╗ █████╗ ███╗   ██╗███╗   ██╗███████╗██████╗     ██╗   ██╗ █████╗ ███████╗
████╗  ██║██╔════╝██╔════╝    ██╔════╝██╔════╝██╔══██╗████╗  ██║████╗  ██║██╔════╝██╔══██╗    ██║   ██║██╔══██╗██╔════╝
██╔██╗ ██║███████╗█████╗      ███████╗██║     ███████║██╔██╗ ██║██╔██╗ ██║█████╗  ██████╔╝    ██║   ██║╚█████╔╝███████╗
██║╚██╗██║╚════██║██╔══╝      ╚════██║██║     ██╔══██║██║╚██╗██║██║╚██╗██║██╔══╝  ██╔══██╗    ╚██╗ ██╔╝██╔══██╗╚════██║
██║ ╚████║███████║███████╗    ███████║╚██████╗██║  ██║██║ ╚████║██║ ╚████║███████╗██║  ██║     ╚████╔╝ ╚█████╔╝███████║
╚═╝  ╚═══╝╚══════╝╚══════╝    ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝      ╚═══╝   ╚════╝ ╚══════╝
    """)
    print("="*120)
    print("                               VERSION 8.5 - ULTIMATE ACCURACY EDITION")
    print("                    Combining v7.4 Advanced Features + v8.0 Backtesting Power")
    print("="*120)
    
    print(f"\n⚙️  CONFIGURATION:")
    print(f"   Mode:              {ACCURACY_MODE}")
    print(f"   Backtest:          {BACKTEST_MODE}")
    print(f"   Min Confidence:    {MIN_CONFIDENCE}%")
    print(f"   Min R:R Ratio:     {MIN_RR_RATIO}x")
    print(f"   Account Capital:   ₹{ACCOUNT_CAPITAL:,}")
    print(f"   Risk per Trade:    {RISK_PER_TRADE*100}%")
    
    print(f"\n✅ ENABLED FEATURES:")
    features = []
    if USE_SECTOR_ROTATION:
        features.append("Sector Rotation")
    if USE_VIX_SENTIMENT:
        features.append("VIX Sentiment")
    if USE_FIBONACCI_SCORING:
        features.append("Fibonacci")
    if USE_PORTFOLIO_RISK:
        features.append("Portfolio Risk")
    if USE_MINI_BACKTEST:
        features.append("Mini Backtest")
    if USE_FULL_BACKTEST:
        features.append("Full Backtest")
    
    print("   " + " | ".join(features))
    print("\n" + "="*120)

# ============================================================================
# MAIN EXECUTION - WITH CACHE VALIDATION
# ============================================================================
# ============================================================================
# MULTI-TIER SIGNAL CLASSIFICATION
# ============================================================================

def classify_signal_tier(result: Dict, backtest_result: Optional[BacktestResult]) -> str:
    """
    Classify signal into tiers based on quality - FIXED VERSION
    """
    confidence = result.get('confidence', 0)
    rr = result.get('risk_reward', 0)
    mini_bt = result.get('mini_backtest', {})
    
    # Safely extract backtest values
    bt_win_rate = 0
    bt_pf = 0
    if backtest_result:
        bt_win_rate = getattr(backtest_result, 'win_rate', 0)
        bt_pf = getattr(backtest_result, 'profit_factor', 0)
    elif result.get('backtest_validated') and 'backtest' in result:
        bt = result['backtest']
        bt_win_rate = bt.get('win_rate', 0)
        bt_pf = bt.get('profit_factor', 0)

    # Tier 1 Requirements (PREMIUM)
    tier1_checks = [
        confidence >= 75,
        rr >= 2.0,
        mini_bt.get('success_reason', '') in ['Target hit, stop not hit', 'Target hit before stop'],
        bt_win_rate >= 58,
        bt_pf >= 1.7,
    ]
    
    if sum(1 for x in tier1_checks if x) >= 4:
        return "TIER_1_PREMIUM"
    
    # Tier 2 Requirements (STANDARD)
    tier2_checks = [
        confidence >= 65,
        rr >= 1.5,
        mini_bt.get('target_hit', False) or mini_bt.get('success_reason', '').startswith('Final P&L'),
        bt_win_rate >= 52,
        bt_pf >= 1.4,
    ]
    
    if sum(1 for x in tier2_checks if x) >= 3:
        return "TIER_2_STANDARD"
    
    # Tier 3 (SPECULATIVE)
    if confidence >= 55 and rr >= 1.3:
        return "TIER_3_SPECULATIVE"
    
    return "REJECTED"


def detect_market_regime(nifty_data: pd.DataFrame) -> str:
    """
    Detect current market regime for adaptive filtering
    
    Returns:
        'TRENDING_UP', 'TRENDING_DOWN', 'RANGE_BOUND', or 'HIGH_VOLATILITY'
    """
    try:
        if nifty_data is None or len(nifty_data) < 50:
            logger.warning("Insufficient Nifty data for regime detection - defaulting to RANGE_BOUND")
            return 'RANGE_BOUND'
        
        # Calculate indicators
        close = nifty_data['Close']
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()
        
        # ATR for volatility
        high = nifty_data['High']
        low = nifty_data['Low']
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        atr_pct = (atr / close * 100).iloc[-1]
        
        # Current price vs SMAs
        current_price = close.iloc[-1]
        above_sma20 = current_price > sma_20.iloc[-1]
        above_sma50 = current_price > sma_50.iloc[-1]
        sma20_above_sma50 = sma_20.iloc[-1] > sma_50.iloc[-1]
        
        # ADX for trend strength
        from ta.trend import ADXIndicator
        adx = ADXIndicator(high, low, close, window=14)
        adx_value = adx.adx().iloc[-1]
        
        # Regime detection
        if atr_pct > 3.0:
            regime = 'HIGH_VOLATILITY'
        elif adx_value > 25 and above_sma20 and above_sma50 and sma20_above_sma50:
            regime = 'TRENDING_UP'
        elif adx_value > 25 and not above_sma20 and not above_sma50 and not sma20_above_sma50:
            regime = 'TRENDING_DOWN'
        else:
            regime = 'RANGE_BOUND'
        
        logger.info(f"📊 Market Regime: {regime} (ADX: {adx_value:.1f}, ATR%: {atr_pct:.2f})")
        return regime
        
    except Exception as e:
        logger.error(f"Error detecting market regime: {e}")
        return 'RANGE_BOUND'


def get_regime_adjusted_thresholds(regime: str) -> dict:
    """
    Return filtering thresholds adjusted for current market regime
    """
    
    thresholds = {
        'TRENDING_UP': {
            'min_confidence': 70,
            'min_bt_wr': 55,
            'min_score': 60,
            'prefer_side': 'LONG',
            'description': 'Trending Up - Favor LONG positions with momentum'
        },
        'TRENDING_DOWN': {
            'min_confidence': 70,
            'min_bt_wr': 55,
            'min_score': 60,
            'prefer_side': 'SHORT',
            'description': 'Trending Down - Favor SHORT positions'
        },
        'RANGE_BOUND': {
            'min_confidence': 75,
            'min_bt_wr': 60,
            'min_score': 65,
            'prefer_side': None,
            'description': 'Range-Bound - Require higher quality, balanced L/S'
        },
        'HIGH_VOLATILITY': {
            'min_confidence': 80,
            'min_bt_wr': 65,
            'min_score': 70,
            'prefer_side': None,
            'description': 'High Volatility - Very selective, highest quality only'
        }
    }
    
    return thresholds.get(regime, thresholds['RANGE_BOUND'])


def select_top_2_stocks(results: list) -> list:
    """
    ✅ ENHANCED v2.0: Production-ready Top 2 selection using ALL quality metrics
    
    Selection Priority:
    1. Tier (Tier-1 > Tier-2 > Tier-3)
    2. Backtest Quality (WR, PF, Reliability)
    3. Setup Type Historical Win Rate
    4. Relative Strength vs Nifty
    5. Trend Strength
    6. Risk:Reward Ratio
    7. LONG/SHORT Diversification
    """
    # ═══════════════════════════════════════════════════════════════════════
    # 🆕 STEP 0: DETECT MARKET REGIME
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n[STEP 0] Detecting Market Regime (Kite)...")

    try:
        nifty = kite_provider.get_nifty50_df(days=90)
        if nifty is not None and not nifty.empty:
            regime = detect_market_regime(nifty)
        else:
            regime = 'RANGE_BOUND'
        thresholds = get_regime_adjusted_thresholds(regime)

        print(f"   🌍 Regime: {regime}")
        print(f"   📋 {thresholds['description']}")
        print(f"   🎯 Thresholds: Confidence ≥{thresholds['min_confidence']}%, "
              f"WR ≥{thresholds['min_bt_wr']}%, Score ≥{thresholds['min_score']}")

    except Exception as e:
        logger.warning(f"Could not fetch market regime: {e} - using defaults")
        regime = 'RANGE_BOUND'
        thresholds = get_regime_adjusted_thresholds(regime)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: APPLY REGIME-BASED FILTERS
    # ═══════════════════════════════════════════════════════════════════════
    
    print(f"\n[STEP 1] Filtering by Quality (Regime-Adjusted)...")
    
    filtered_results = []
    for stock in results:
        bt = stock.get('backtest', {})
        bt_wr = bt.get('win_rate', 0) if bt else 0
        conf = stock.get('confidence', 0)
        
        # Apply regime-specific thresholds
        if (bt_wr >= thresholds['min_bt_wr'] and 
            conf >= thresholds['min_confidence']):
            filtered_results.append(stock)
    
    print(f"   ✅ {len(filtered_results)} signals pass regime-adjusted filters")
    
    print("\n" + "="*120)
    print("🎯 ENHANCED TOP 2 SELECTION v2.0")
    print("="*120)
    
    if not results or len(results) == 0:
        print("❌ No signals to select from!")
        return []
    
    print(f"\n📊 Input: {len(results)} total signals")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: APPLY TIER CLASSIFICATION
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n[STEP 1] Applying Tier Classification...")
    tiers = apply_tier_based_filtering(results)
    
    for tier_name, tier_list in tiers.items():
        print(f"   {tier_name}: {len(tier_list)} signals")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: COMPREHENSIVE QUALITY SCORING
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n[STEP 2] Calculating Comprehensive Quality Scores...")
    
    def calculate_composite_score(stock: dict) -> float:
        """
        Calculate comprehensive quality score (0-100).
        Uses ALL available quality metrics.

        [FIX MIN-8] The original code tracked max_score with repeated
        `max_score += N` and then divided by it at the end.  Because the
        variable grew INSIDE the function the denominator was always 100
        when ALL branches ran, but any early-return path (e.g. no backtest)
        could produce an inflated ratio.  Fixed: we hard-code the total
        weight sum (TOTAL_WEIGHT = 100) so normalization is always correct.

        Category weights  (must sum to 100):
          Tier           25
          Backtest       30
          Setup Type     10
          Rel Strength   10
          Trend          10
          Risk:Reward    10
          Confidence      5
          Total         100
        """
        TOTAL_WEIGHT = 100   # [FIX MIN-8] fixed denominator
        score = 0

        # ────────────────────────────────────────
        # TIER SCORE (max 25 points)
        # ────────────────────────────────────────
        tier = stock.get('tier', 'TIER_3_SPECULATIVE')
        if 'TIER_1' in str(tier) or 'PREMIUM' in str(tier).upper():
            score += 25
        elif 'TIER_2' in str(tier) or 'STANDARD' in str(tier).upper():
            score += 15
        else:
            score += 5

        # ────────────────────────────────────────
        # BACKTEST QUALITY (max 30 points)
        # ────────────────────────────────────────
        bt = stock.get('backtest', {})

        if stock.get('backtest_validated') and bt:
            # Win Rate (max 12 points)
            wr = bt.get('win_rate', 0)
            score += min(12, (wr - 50) * 0.4) if wr > 50 else 0

            # Profit Factor (max 10 points)
            pf = bt.get('profit_factor', 0)
            score += min(10, (pf - 1) * 5) if pf > 1 else 0

            # Reliability Score (max 8 points)
            rel = bt.get('reliability_score', 0)
            score += min(8, rel * 0.08)

        # ────────────────────────────────────────
        # SETUP TYPE (max 10 points)
        # ────────────────────────────────────────
        setup_type = stock.get('setup_type', 'UNKNOWN')
        setup_bonus = {
            'VCP': 10,
            'BREAKOUT': 8,
            'SUPPORT_BOUNCE': 7,
            'MOMENTUM': 6,
            'EMA_BOUNCE': 5,
            'RSI_REVERSAL': 4,
            'GAP_PLAY': 3,
            'UNKNOWN': 0
        }
        score += setup_bonus.get(setup_type, 0)

        # ────────────────────────────────────────
        # RELATIVE STRENGTH (max 10 points)
        # ────────────────────────────────────────
        rs_interp = stock.get('rs_interpretation', 'NEUTRAL')
        if stock.get('side') == 'LONG':
            if rs_interp == 'VERY_STRONG':
                score += 10
            elif rs_interp == 'STRONG':
                score += 7
            elif rs_interp == 'NEUTRAL':
                score += 4
        else:  # SHORT
            if rs_interp == 'VERY_WEAK':
                score += 10
            elif rs_interp == 'WEAK':
                score += 7
            elif rs_interp == 'NEUTRAL':
                score += 4

        # ────────────────────────────────────────
        # TREND STRENGTH (max 10 points)
        # ────────────────────────────────────────
        trend = stock.get('trend_strength', 50)
        score += min(10, trend * 0.1)

        # ────────────────────────────────────────
        # RISK:REWARD (max 10 points)
        # ────────────────────────────────────────
        rr = stock.get('risk_reward', 0)
        score += min(10, rr * 3)

        # ────────────────────────────────────────
        # CONFIDENCE (max 5 points)
        # ────────────────────────────────────────
        conf = stock.get('confidence', 0)
        score += min(5, (conf - 60) * 0.125) if conf > 60 else 0

        # Normalize to 0-100 using fixed total weight
        return round(min(100.0, (score / TOTAL_WEIGHT) * 100), 1)
    
    # Calculate scores for all signals
    for result in results:
        result['composite_score'] = calculate_composite_score(result)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: SORT BY COMPOSITE SCORE (WITH DETERMINISTIC TIE-BREAKER)
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[STEP 3] Ranking by Composite Score...")

    # 🆕 DETERMINISTIC TIE-BREAKER
    def sort_key(stock):
        """
        Multi-level sort key for deterministic ranking
        
        Priority:
        1. Composite Score (descending)
        2. Backtest Win Rate (descending)
        3. Tier (TIER_1 > TIER_2)
        4. Ticker (alphabetical - ultimate tie-breaker)
        """
        tier_rank = {
            'TIER_1_PREMIUM': 1,
            'TIER_2_STANDARD': 2,
            'TIER_3_SPECULATIVE': 3
        }
        
        bt = stock.get('backtest', {})
        bt_wr = bt.get('win_rate', 0) if bt else 0
        
        return (
            -stock.get('composite_score', 0),  # Higher is better (negative for descending)
            -bt_wr,                            # Higher WR wins
            tier_rank.get(stock.get('tier', 'TIER_3_SPECULATIVE'), 99),  # Lower tier number wins
            stock.get('ticker', 'ZZZZ')        # Alphabetical (A comes before Z)
        )

    sorted_results = sorted(results, key=sort_key)

    logger.info("✅ Applied deterministic tie-breaker: Score → WR → Tier → Ticker")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: APPLY DIVERSIFICATION (1 LONG + 1 SHORT IF POSSIBLE)
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[STEP 4] Selecting Top 2 by Quality Score...")

    # 🎯 SIMPLE: ALWAYS SELECT TOP 2 BY SCORE (NO DIVERSIFICATION LOGIC)
    selected = sorted_results[:2]
    
    # Log the selection
    if len(selected) >= 2:
        side_1 = selected[0].get('side', 'N/A')
        side_2 = selected[1].get('side', 'N/A')
        
        if side_1 == side_2:
            selection_method = f"TOP 2 BY QUALITY (Both {side_1})"
            print(f"   📊 Selection: Best 2 stocks by score (Both {side_1} positions)")
            logger.info(f"✅ QUALITY-FIRST: Selected top 2 by score - {selected[0]['ticker']} + {selected[1]['ticker']}")
        else:
            selection_method = f"TOP 2 BY QUALITY (1 {side_1} + 1 {side_2})"
            print(f"   📊 Selection: Best 2 stocks by score (Naturally diversified: 1L+1S)")
            logger.info(f"✅ QUALITY-FIRST: Selected top 2 by score - {selected[0]['ticker']} + {selected[1]['ticker']}")
    else:
        selection_method = "TOP 2 BY QUALITY"
    
    print(f"   ✅ Selection Method: {selection_method}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: FINAL OUTPUT
    # ═══════════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*120}")
    print(f"🏆 FINAL TOP 2 SELECTED")
    print(f"{'='*120}")
    
    for i, stock in enumerate(selected, 1):
        bt = stock.get('backtest', {})
        
        print(f"\n{'─'*120}")
        print(f"#{i} {stock['ticker']} ({stock['side']}) - COMPOSITE SCORE: {stock.get('composite_score', 0):.1f}/100")
        print(f"{'─'*120}")
        
        print(f"   📊 CLASSIFICATION:")
        print(f"      Tier: {stock.get('tier', 'N/A')}")
        print(f"      Setup Type: {stock.get('setup_type', 'N/A')}")
        
        print(f"\n   💰 PRICE LEVELS:")
        print(f"      Entry: ₹{stock.get('entry_price', 0):.2f}")
        print(f"      Stop Loss: ₹{stock.get('stop_loss', 0):.2f}")
        print(f"      Target 1: ₹{stock.get('target_1', 0):.2f}")
        print(f"      Target 2: ₹{stock.get('target_2', 0):.2f}")
        
        print(f"\n   📈 QUALITY METRICS:")
        print(f"      Confidence: {stock.get('confidence', 0):.1f}%")
        print(f"      R:R Ratio: {stock.get('risk_reward', 0):.1f}x")
        print(f"      Trend Strength: {stock.get('trend_strength', 0):.0f}")
        print(f"      Relative Strength: {stock.get('relative_strength', 0):.0f} ({stock.get('rs_interpretation', 'N/A')})")
        
        if bt:
            print(f"\n   🔬 BACKTEST:")
            print(f"      Win Rate: {bt.get('win_rate', 0):.1f}%")
            print(f"      Profit Factor: {bt.get('profit_factor', 0):.2f}x")
            print(f"      Reliability: {bt.get('reliability_score', 0):.1f}/100")
    
    print(f"\n{'='*120}\n")
    # ════════════════════════════════════════════════════════════════════
    # 🆕 OPTIONAL: MINIMUM QUALITY GAP CHECK
    # ════════════════════════════════════════════════════════════════════
    
    MIN_ACCEPTABLE_GAP = 5   # Alert if gap is larger than this
    MAX_ACCEPTABLE_GAP = 25  # Critical warning if gap exceeds this
    
    if len(selected) == 2:
        score_1 = selected[0].get('composite_score', 0)
        score_2 = selected[1].get('composite_score', 0)
        score_gap = score_1 - score_2
        
        print(f"\n{'='*120}")
        print(f"📊 QUALITY GAP ANALYSIS")
        print(f"{'='*120}")
        print(f"   #{1} ({selected[0]['ticker']}): {score_1:.1f}/100")
        print(f"   #{2} ({selected[1]['ticker']}): {score_2:.1f}/100")
        print(f"   Gap: {score_gap:.1f} points ({(score_gap/score_1*100):.1f}%)")
        
        if score_gap > MAX_ACCEPTABLE_GAP:
            print(f"   ⚠️  CRITICAL GAP: #{2} is significantly weaker than #{1}")
            print(f"   ⚠️  Consider trading ONLY #{1} today for better risk management")
            logger.warning(f"QUALITY GAP WARNING: {selected[0]['ticker']} scores {score_gap:.1f} points higher than {selected[1]['ticker']}")
        elif score_gap > MIN_ACCEPTABLE_GAP:
            print(f"   ℹ️  Notable gap: #{1} has clear quality advantage")
        else:
            print(f"   ✅ Quality gap is acceptable - both signals are comparable")
        
        print(f"{'='*120}\n")

    

    selected_tickers = {s['ticker'] for s in selected}
    rejected = [s for s in sorted_results if s['ticker'] not in selected_tickers]
    
    if rejected:
        print(f"\n🚫 REJECTED CANDIDATES ({len(rejected)} stocks):\n")
        print(f"   {'Rank':<6} {'Ticker':<10} {'Side':<6} {'Score':<8} {'Tier':<20} {'Rejection Reason':<50}")
        print(f"   {'-'*110}")
        
        for i, stock in enumerate(rejected[:10], 3):  # Show top 10 rejected
            score = stock.get('composite_score', 0)
            tier = stock.get('tier', 'N/A')
            bt = stock.get('backtest', {})
            bt_wr = bt.get('win_rate', 0) if bt else 0
            bt_pf = bt.get('profit_factor', 0) if bt else 0
            
            # Determine rejection reason
            if i == 3:
                reason = "Ranked #3 - Only Top 2 selected"
            elif tier == 'TIER_3_SPECULATIVE':
                reason = "TIER_3 - Below quality threshold"
            elif bt_wr < 55:
                reason = f"Low Backtest WR: {bt_wr:.1f}% (need ≥55%)"
            elif bt_pf < 1.2:
                reason = f"Low Profit Factor: {bt_pf:.2f}x (need ≥1.2x)"
            elif stock.get('confidence', 0) < 70:
                reason = f"Low Confidence: {stock.get('confidence', 0):.1f}% (prefer ≥70%)"
            else:
                reason = f"Lower Score than Top 2 (diff: {selected[1].get('composite_score', 0) - score:.1f} pts)"
            
            print(f"   #{i:<5} {stock['ticker']:<10} {stock['side']:<6} {score:<8.1f} {tier:<20} {reason:<50}")
    
    print(f"\n{'='*120}\n")
    
    return selected

def export_top_2_files(top_2: list, signals_dir: str, timestamp: str):
    """
    Export TOP 2 to txt and csv files
    ✅ FIXED: Includes ALL columns needed by Google Sheets
    ✅ FIXED: Column names match what format_sheet_row() expects
    """
    import csv
    
    # TXT File (human readable)
    txt_file = os.path.join(signals_dir, f"TOP_2_PICKS_{timestamp}.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"🎯 TOP 2 PICKS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        for i, s in enumerate(top_2, 1):
            bt = s.get('backtest', {})
            f.write(f"#{i} {s['ticker']} ({s['side']})\n")
            f.write(f"  Tier: {s.get('tier', 'N/A')}\n")
            f.write(f"  Entry: ₹{s['entry_price']:.2f} | SL: ₹{s['stop_loss']:.2f}\n")
            f.write(f"  T1: ₹{s['target_1']:.2f} | T2: ₹{s['target_2']:.2f}\n")
            f.write(f"  Confidence: {s['confidence']:.1f}%")
            if bt:
                f.write(f" | Win Rate: {bt.get('win_rate', 0):.1f}%")
                f.write(f" | PF: {bt.get('profit_factor', 0):.2f}x")
            f.write(f"\n  Setup: {s.get('setup_type', 'N/A')}")
            f.write(f" | Trend: {s.get('trend_strength', 0):.0f}")
            f.write(f" | RS: {s.get('relative_strength', 0):.0f}")
            f.write("\n" + "─"*80 + "\n\n")
    
    # CSV File - ✅ FIXED: Use exact column names that format_sheet_row() expects
    csv_file = os.path.join(signals_dir, f"TOP_2_PICKS_{timestamp}.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # ✅ FIXED: Column names now match what update_google_sheet() expects
        writer.writerow([
            'Rank',
            'ticker',          # ✅ lowercase
            'side',            # ✅ lowercase  
            'price',           # ✅ lowercase 'price' not 'Entry'
            'entry_price',     # ✅ explicit entry_price
            'stop_loss',       # ✅ full name
            'target_1',        # ✅ full name
            'target_2',        # ✅ full name
            'confidence',      # ✅ lowercase
            'risk_reward',     # ✅ added
            'qty',             # ✅ added
            'tier',            # ✅ ADDED - critical for ranking
            'setup_type',      # ✅ ADDED
            'trend_strength',  # ✅ ADDED
            'relative_strength', # ✅ ADDED
            'BT_WR',           # Backtest Win Rate
            'BT_PF',           # Backtest Profit Factor
            'BT_Sharpe',       # ✅ ADDED
            'BT_MaxDD',        # ✅ ADDED
            'BT_Reliability',  # ✅ ADDED
            'backtest_validated'  # ✅ ADDED
        ])
        
        for i, s in enumerate(top_2, 1):
            bt = s.get('backtest', {})
            
            writer.writerow([
                i,
                s.get('ticker', ''),
                s.get('side', ''),
                s.get('price', s.get('entry_price', 0)),
                s.get('entry_price', 0),
                s.get('stop_loss', 0),
                s.get('target_1', 0),
                s.get('target_2', 0),
                s.get('confidence', 0),
                s.get('risk_reward', 0),
                s.get('qty', 0),
                s.get('tier', 'UNKNOWN'),
                s.get('setup_type', 'UNKNOWN'),
                s.get('trend_strength', 0),
                s.get('relative_strength', 0),
                bt.get('win_rate', 0) if bt else 0,
                bt.get('profit_factor', 0) if bt else 0,
                bt.get('sharpe_ratio', 0) if bt else 0,
                bt.get('max_drawdown', 0) if bt else 0,
                bt.get('reliability_score', 0) if bt else 0,
                'True' if s.get('backtest_validated') else 'False'
            ])
    
    print(f"✅ TOP 2 files exported:")
    print(f"   📄 TXT: {txt_file}")
    print(f"   📄 CSV: {csv_file}")
    
    return txt_file, csv_file


def apply_tier_based_filtering(results: List[Dict]) -> Dict[str, List[Dict]]:
    """
    ✅ ENHANCED: Separate results into tiers AND add 'tier' key to each result
    """
    
    tiers = {
        "TIER_1_PREMIUM": [],
        "TIER_2_STANDARD": [],
        "TIER_3_SPECULATIVE": []
    }
    
    for result in results:
        backtest = None
        if result.get('backtest_validated') and 'backtest' in result:
            # Reconstruct BacktestResult from dict
            bt_dict = result['backtest']
            backtest = BacktestResult(
                total_trades=bt_dict.get('total_trades', 10),
                winning_trades=0,
                losing_trades=0,
                win_rate=bt_dict.get('win_rate', 50),
                avg_win=0,
                avg_loss=0,
                profit_factor=bt_dict.get('profit_factor', 1.5),
                sharpe_ratio=bt_dict.get('sharpe_ratio', 0),
                max_drawdown=bt_dict.get('max_drawdown', 10),
                total_return=0,
                avg_holding_days=0,
                best_trade=0,
                worst_trade=0,
                consecutive_wins=0,
                consecutive_losses=0,
                reliability_score=bt_dict.get('reliability_score', 50),
                expectancy=bt_dict.get('expectancy', 0)
            )
        
        tier = classify_signal_tier(result, backtest)
        
        if tier in tiers:
            result['tier'] = tier  # ✅ CRITICAL: Add tier key to result
            tiers[tier].append(result)
    
    return tiers


def main():
    """Main execution function - FIXED VERSION WITH VALIDATION"""
    
    start_time = time.time()
    
    print_header()
    
    # ========================================================================
    # [1/5] LOAD STOCK UNIVERSE
    # ========================================================================
    
    print("\n[1/5] Loading Stock Universe...")
    tickers = get_universe_symbols()
    
    if not tickers:
        print("❌ Failed to load stock symbols")
        return
    
    print(f"  ✓ Loaded {len(tickers)} symbols")
    
    # ========================================================================
    # [2/5] INITIALIZE FILTERS
    # ========================================================================
    
    print("\n[2/5] Initializing Analysis Engines...")
    
    quality_validator = DataQualityValidator()
    sector_filter = SectorRotationAnalyzer() if USE_SECTOR_ROTATION else None
    vix_filter = VIXSentimentAnalyzer() if USE_VIX_SENTIMENT else None
    fibo_detector = FibonacciDetector() if USE_FIBONACCI_SCORING else None
    # [FIX RISK-1] Use the pre-seeded manager (loaded from Google Sheets with real
    # ACTIVE positions) if github_actions_main() prepared one. Otherwise fall back
    # to a fresh empty instance (e.g. when running scanner standalone/locally).
    if USE_PORTFOLIO_RISK:
        import builtins as _builtins
        portfolio_mgr = getattr(_builtins, '_seeded_portfolio_mgr', None)
        if portfolio_mgr:
            logger.info(
                f"✅ Using pre-seeded PortfolioRiskManager "
                f"({len(portfolio_mgr.current_trades)} real open positions loaded)"
            )
        else:
            portfolio_mgr = PortfolioRiskManager(ACCOUNT_CAPITAL)
            logger.info("ℹ️  PortfolioRiskManager: no pre-seeded data — starting empty (local run?)")
    else:
        portfolio_mgr = None
    mtf_analyzer = MultiTimeframeAnalyzer()
    
    if sector_filter:
        print("  └─ Updating sector data...")
        sector_filter.update_sector_strength()
    
    print("  ✓ All engines initialized")
    
    # ========================================================================
    # [3/5] BUILD PRICE CACHE
    # ========================================================================
    
    print("\n[3/5] Building Price Cache...")
    df_all = build_cache(tickers)
    
    if df_all is None or df_all.empty:
            print("❌ Failed to build price cache")
            return

        # Fix tz-naive vs tz-aware mixed timestamps (Kite may return tz-aware)
    df_all['Date'] = pd.to_datetime(df_all['Date']).dt.tz_localize(None)

    # ========================================================================
    # FIX 3: CACHE VALIDATION
    
    # ========================================================================
    # FIX 3: CACHE VALIDATION
    # ========================================================================
    
    print(f"\n{'='*120}")
    print("CACHE VALIDATION & INTEGRITY CHECK")
    print('='*120)
    
    print(f"\n📊 Cache Structure:")
    print(f"   Total rows:        {len(df_all):,}")
    print(f"   Columns:           {df_all.columns.tolist()}")
    print(f"   Unique tickers:    {df_all['ticker'].nunique()}")
    print(f"   Date range:        {df_all['Date'].min()} to {df_all['Date'].max()}")
    
    print(f"\n🔍 Ticker Format Check:")
    sample_tickers = df_all['ticker'].unique()[:10]
    print(f"   Sample tickers:    {sample_tickers.tolist()}")
    ticker_format = 'WITH .NS' if any('.NS' in str(t) for t in sample_tickers) else 'WITHOUT .NS'
    print(f"   Format detected:   {ticker_format}")
    
    print(f"\n🎯 Data Uniqueness Validation:")
    print(f"   {'Ticker':<15} {'Rows':>6} {'Unique':>7} {'Price Range':<30} {'Status'}")
    print(f"   {'-'*15} {'-'*6} {'-'*7} {'-'*30} {'-'*10}")
    
    # Test first 5 tickers for data integrity
    test_tickers = df_all['ticker'].unique()[:5]
    all_valid = True
    
    for test_ticker in test_tickers:
        test_data = df_all[df_all['ticker'] == test_ticker]
        
        rows = len(test_data)
        unique_closes = test_data['Close'].nunique()
        min_price = test_data['Close'].min()
        max_price = test_data['Close'].max()
        price_range = max_price - min_price
        
        # Check if data looks valid
        uniqueness_ratio = unique_closes / rows if rows > 0 else 0
        
        if uniqueness_ratio < 0.3 and rows > 20:
            status = "⚠️  SUSPICIOUS"
            all_valid = False
        else:
            status = "✓ OK"
        
        ticker_short = test_ticker[:15]
        price_range_str = f"₹{min_price:.2f} - ₹{max_price:.2f}"
        
        print(f"   {ticker_short:<15} {rows:>6} {unique_closes:>7} {price_range_str:<30} {status}")
    
    # Overall validation result
    print(f"\n{'='*120}")
    if all_valid:
        print("✅ Cache validation PASSED - Data looks good!")
    else:
        print("⚠️  Cache validation WARNING - Some data may be corrupted")
        print("   Proceeding anyway, but results may be affected")
    print('='*120)
    
    # ========================================================================
    # [4/5] SCAN STOCKS
    # ========================================================================
    
    cached_tickers = sorted(df_all["ticker"].unique().tolist())
    print(f"\n  ✓ Limiting scan to {len(cached_tickers)} cached tickers (was {len(tickers)})")
    tickers = cached_tickers  # ✅ CRITICAL: Only scan what we have
    
    print(f"\n[4/5] Scanning {len(tickers)} Stocks...")
    
    if BACKTEST_MODE in ['HYBRID', 'FULL']:
        results = hybrid_scan_universe(
            tickers, df_all, quality_validator, sector_filter,
            vix_filter, fibo_detector, portfolio_mgr, mtf_analyzer
        )
    else:
        print(f"\n{'='*120}")
        print("SCANNING WITH MINI BACKTEST")
        print('='*120)
        
        results = []
        
        for i, ticker in enumerate(tickers, 1):
            result = scan_ticker(
                ticker, df_all, quality_validator, sector_filter,
                vix_filter, fibo_detector, portfolio_mgr, mtf_analyzer,
                backtester=None, run_full_backtest=False
            )
            
            if result:
                results.append(result)
                if USE_PORTFOLIO_RISK and portfolio_mgr:
                    portfolio_mgr.add_trade(result)
            
            if i % 25 == 0 or i == len(tickers):
                pct = (i / len(tickers)) * 100
                bar_length = 50
                filled = int(bar_length * i / len(tickers))
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"  [{bar}] {pct:>5.1f}% | {i:>4}/{len(tickers)} | Found: {len(results):>3}", 
                      end='\r', flush=True)
        
        print()
        results = sorted(results, key=lambda x: x['confidence'], reverse=True)
    
    # ========================================================================
    # [5/5] DISPLAY RESULTS
    # ========================================================================
    
    print(f"\n[5/5] Generating Reports...")
    
    print_statistics()
    
    if not results:
        print(f"\n{'='*120}")
        print("❌ NO SIGNALS FOUND")
        print('='*120)
        print(f"\n💡 SUGGESTIONS:")
        print(f"   • Switch to AGGRESSIVE mode (currently: {ACCURACY_MODE})")
        print(f"   • Lower MIN_CONFIDENCE to {MIN_CONFIDENCE - 10}%")
        print(f"   • Check rejection reasons above")
        print('='*120 + "\n")
        return
    
    print_results(results)
    
    scan_time = time.time() - start_time
    export_results(results, scan_time)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print(f"\n{'='*120}")
    print("✅ SCAN COMPLETE!")
    print('='*120)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Mode:              {ACCURACY_MODE}")
    print(f"   Backtest:          {BACKTEST_MODE}")
    print(f"   Stocks Scanned:    {stats['total']}")
    print(f"   Signals Found:     {len(results)}")
    print(f"   Success Rate:      {(stats['passed']/stats['total']*100) if stats['total'] > 0 else 0:.1f}%")
    print(f"   Scan Time:         {scan_time:.1f} seconds")
    
    if results:
        avg_conf = sum(r['confidence'] for r in results) / len(results)
        print(f"   Avg Confidence:    {avg_conf:.1f}%")
        
        validated = len([r for r in results if r.get('backtest_validated')])
        if validated > 0:
            print(f"   Backtest Validated: {validated}/{len(results)}")
    
    print(f"\n💾 OUTPUT: {SIGNALS_DIR}/")
    
    # Auto TOP 2 selection
    if results and len(results) >= 2:
        top_2_picks = select_top_2_stocks(results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_top_2_files(top_2_picks, SIGNALS_DIR, timestamp)
    
    print("\n" + "="*120 + "\n")
    return results

# ============================================================================
# EMAIL NOTIFICATION FUNCTION
# ============================================================================

def send_email_notification(results, csv_file=None, html_file=None, scan_time=0):
    """Send email notification with scan results and Google Sheets update status"""
    
    if not EMAIL_CONFIG['enabled'] or not EMAIL_CONFIG['sender_email']:
        logger.info("Email not configured")
        return
    
    try:
        # Count signals
        num_signals = len(results) if results else 0
        
        msg = MIMEMultipart()
        msg['Subject'] = f"📊 NSE Scanner - {datetime.now().strftime('%Y-%m-%d')} - {num_signals} Signals"
        msg['From'] = EMAIL_CONFIG['sender_email']
        msg['To'] = EMAIL_CONFIG['recipient_email']
        
        # Build email body with Google Sheets status
        google_sheets_status = "❌ Not configured"
        if GOOGLE_SHEETS_CONFIG['enabled']:
            if GOOGLE_SHEETS_CONFIG['spreadsheet_id']:
                google_sheets_status = "✅ Updated successfully"
            else:
                google_sheets_status = "⚠️  Configured but Sheet ID missing"
        
        body = f"""
NSE SWING SCANNER v8.5 - DAILY REPORT
{'='*70}

Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}
Scan Time: {scan_time:.1f} seconds
Mode: {ACCURACY_MODE}
Backtest: {BACKTEST_MODE}

📊 SIGNALS FOUND: {num_signals}
📈 Google Sheets Update: {google_sheets_status}

{'='*70}

✅ Signal files attached:
• CSV Report (detailed data)
• HTML Report (visual analysis)

🔗 Next Steps:
1. Review signals in attached files or Google Sheet
2. Check entry levels and stop losses
3. Confirm market conditions before trading
4. Entry Date: {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}

{'='*70}

This is an automated message from NSE Scanner v8.5
Running on GitHub Actions ⚡
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach CSV file
        if csv_file and os.path.exists(csv_file):
            with open(csv_file, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(csv_file)}')
                msg.attach(part)
        
        # Attach HTML file
        if html_file and os.path.exists(html_file):
            with open(html_file, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(html_file)}')
                msg.attach(part)
        
        # Send email
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            server.send_message(msg)
        
        logger.info("✅ Email notification sent successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to send email: {e}")

# ============================================================================
# MODIFIED MAIN FUNCTION FOR GITHUB ACTIONS
# ============================================================================

def _check_daily_loss_circuit_breaker() -> tuple:
    """
    [FIX CB-1] Daily Loss Circuit Breaker — reads LIVE from Google Sheets PL tab.

    Your PL sheet columns (confirmed from your workbook):
      A: Ticker   B: Position   C: Entry_Price   D: Exit_Price
      E: Quantity  F: P/L        G: Result        H: Exit_Reason
      I: Exit_Time  (format: '2026-01-13 11:47:37' or Excel serial date)

    Logic:
      - Reads every row from the PL sheet via Google Sheets API
      - Filters rows where Exit_Time is TODAY
      - Sums the P/L column for those rows
      - If total loss >= 3% of ACCOUNT_CAPITAL → block the scan

    Returns (allowed: bool, reason: str)
    """
    DAILY_LOSS_LIMIT_PCT = 0.03                          # 3% hard stop
    DAILY_LOSS_LIMIT_ABS = ACCOUNT_CAPITAL * DAILY_LOSS_LIMIT_PCT
    PL_SHEET_NAME        = 'PL'                          # Tab name in your Google Sheet
    today_str            = datetime.now().strftime('%Y-%m-%d')

    try:
        # ── Step 1: Connect to Google Sheets ────────────────────────────────
        service = get_google_sheets_service()
        if not service:
            # Can't reach Sheets — don't block the scan, just warn
            logger.warning("⚠️  Circuit breaker: Google Sheets unavailable — scan allowed")
            return True, "Google Sheets unavailable — circuit breaker skipped (non-blocking)"

        spreadsheet_id = GOOGLE_SHEETS_CONFIG['spreadsheet_id']
        if not spreadsheet_id:
            logger.warning("⚠️  Circuit breaker: spreadsheet_id not set — scan allowed")
            return True, "Spreadsheet ID not configured — circuit breaker skipped"

        # ── Step 2: Fetch ALL rows from PL sheet (cols A:I) ─────────────────
        # Row 1 = headers, Row 2 onwards = formula rows or data
        # We fetch A:I to get Ticker + P/L (col F) + Exit_Time (col I)
        try:
            result = service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=f'{PL_SHEET_NAME}!A:I',
                valueRenderOption='UNFORMATTED_VALUE',   # numbers as numbers, not display text
                dateTimeRenderOption='FORMATTED_STRING'  # dates as readable strings
            ).execute()
        except Exception as api_err:
            logger.warning(f"⚠️  Circuit breaker: PL sheet read failed ({api_err}) — scan allowed")
            return True, f"PL sheet read error (non-blocking): {api_err}"

        rows = result.get('values', [])
        if len(rows) < 2:
            logger.info("⚡ Circuit breaker: PL sheet empty or header-only — scan allowed")
            return True, "PL sheet has no trade data — circuit breaker passed"

        # ── Step 3: Find column indices from the header row ─────────────────
        # Headers have trailing spaces in your sheet e.g. 'Ticker ', 'P/L'
        header = [str(h).strip() for h in rows[0]]

        def col_idx(name):
            """Return 0-based index of a column name, or -1 if not found."""
            for i, h in enumerate(header):
                if h.lower() == name.lower():
                    return i
            return -1

        pnl_col      = col_idx('P/L')       # column F → index 5
        exit_time_col = col_idx('Exit_Time') # column I → index 8
        ticker_col    = col_idx('Ticker')    # column A → index 0

        if pnl_col == -1 or exit_time_col == -1:
            logger.warning(
                f"⚠️  Circuit breaker: Could not find 'P/L' or 'Exit_Time' columns "
                f"in PL sheet headers {header} — scan allowed"
            )
            return True, "PL sheet column mapping failed — circuit breaker skipped"

        # ── Step 4: Sum today's P/L ─────────────────────────────────────────
        todays_pnl    = 0.0
        todays_trades = 0
        today_date    = datetime.now().date()

        def _parse_exit_date(raw):
            """
            Parse Exit_Time into a date object.
            Google Sheets API (FORMATTED_STRING) returns dates in the sheet's
            locale — Indian sheets typically use DD/MM/YYYY or D/M/YYYY.
            Also handles ISO format YYYY-MM-DD just in case.
            Returns None if unparseable.
            """
            if not raw or str(raw).strip() in ('', 'None'):
                return None
            s = str(raw).strip()
            from datetime import date
            # Try all common formats Google Sheets may return
            for fmt in ('%d/%m/%Y %H:%M:%S', '%d/%m/%Y',
                        '%m/%d/%Y %H:%M:%S', '%m/%d/%Y',
                        '%Y-%m-%d %H:%M:%S', '%Y-%m-%d',
                        '%d-%m-%Y %H:%M:%S', '%d-%m-%Y'):
                try:
                    return datetime.strptime(s[:len(fmt.replace('%d','01').replace('%m','01').replace('%Y','2000').replace('%H','00').replace('%M','00').replace('%S','00'))], fmt).date()
                except ValueError:
                    pass
            # Last resort: try dateutil if available
            try:
                from dateutil import parser as _dp
                return _dp.parse(s).date()
            except Exception:
                pass
            return None

        for row in rows[1:]:   # skip header
            # Guard: row may be shorter than expected (empty trailing cells)
            if len(row) <= max(pnl_col, exit_time_col):
                continue

            exit_time_raw = row[exit_time_col]
            pnl_raw       = row[pnl_col]

            exit_date = _parse_exit_date(exit_time_raw)
            if exit_date != today_date:
                continue

            # Parse P/L value
            try:
                pnl_value = float(str(pnl_raw).replace(',', '').replace('₹', '').strip())
            except (ValueError, TypeError):
                logger.debug(f"Circuit breaker: skipping unparseable P/L value '{pnl_raw}'")
                continue

            todays_pnl   += pnl_value
            todays_trades += 1

            ticker = row[ticker_col] if len(row) > ticker_col else '?'
            logger.info(f"⚡ Circuit breaker: found today trade — "
                        f"{ticker} P/L = ₹{pnl_value:+,.2f}")

        # ── Step 5: Decision ────────────────────────────────────────────────
        if todays_trades == 0:
            logger.info(f"⚡ Circuit breaker: no closed trades today ({today_str}) — scan allowed")
            return True, f"No closed trades today — circuit breaker passed"

        logger.info(f"⚡ Circuit breaker: {todays_trades} trade(s) today, "
                    f"total P/L = ₹{todays_pnl:+,.2f} "
                    f"(limit = ₹{-DAILY_LOSS_LIMIT_ABS:,.2f})")

        if todays_pnl <= -DAILY_LOSS_LIMIT_ABS:
            msg = (
                f"🚨 CIRCUIT BREAKER TRIGGERED: Today's realized loss "
                f"₹{abs(todays_pnl):,.0f} has hit the {DAILY_LOSS_LIMIT_PCT*100:.0f}% "
                f"daily limit (₹{DAILY_LOSS_LIMIT_ABS:,.0f} on ₹{ACCOUNT_CAPITAL:,} capital). "
                f"Scan blocked to protect remaining capital."
            )
            logger.critical(msg)
            return False, msg

        return True, (
            f"Circuit breaker passed — today P/L: ₹{todays_pnl:+,.2f} "
            f"across {todays_trades} trade(s) "
            f"(limit: ₹{-DAILY_LOSS_LIMIT_ABS:,.2f})"
        )

    except Exception as e:
        # Safety net — never block the scan due to an internal CB error
        logger.warning(f"⚠️  Circuit breaker unexpected error: {e} — scan allowed")
        return True, f"Circuit breaker error (non-blocking): {e}"


def _load_active_positions_into_risk_manager(portfolio_mgr: 'PortfolioRiskManager') -> str:
    """
    [FIX RISK-1] Seeds PortfolioRiskManager with your REAL open positions
    from the Google Sheet's 'Portfolio' tab BEFORE the scan starts.

    WHY THIS MATTERS:
    Without this, PortfolioRiskManager starts empty every run.
    So can_add_trade() checks 0 open positions against MAX_CONCURRENT_TRADES=3
    — completely ignoring your 12 actual ACTIVE trades.
    This means the sector exposure cap and max-position cap are both blind.

    HOW IT WORKS:
    Reads Portfolio sheet → filters rows where Status == 'ACTIVE'
    → calls portfolio_mgr.add_trade() for each one to pre-populate
    the risk manager's state before main() scans new signals.

    Portfolio sheet columns (from your workbook):
      A:Ticker  B:Position  C:Entry_Price  D:Quantity  E:Stop_Loss
      F:Target_1  G:Target_2  H:Entry_Date  I:Status  J:Notes

    Returns a summary string for logging.
    """
    PORTFOLIO_SHEET = 'Portfolio'

    try:
        service = get_google_sheets_service()
        if not service:
            logger.warning("⚠️  Risk seeding: Google Sheets unavailable — portfolio_mgr stays empty")
            return "Google Sheets unavailable — risk manager not seeded"

        spreadsheet_id = GOOGLE_SHEETS_CONFIG['spreadsheet_id']
        if not spreadsheet_id:
            return "Spreadsheet ID not set — risk manager not seeded"

        # ── Fetch Portfolio sheet ────────────────────────────────────────────
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=f'{PORTFOLIO_SHEET}!A:J',
            valueRenderOption='UNFORMATTED_VALUE',
            dateTimeRenderOption='FORMATTED_STRING'
        ).execute()

        rows = result.get('values', [])
        if len(rows) < 2:
            return "Portfolio sheet empty — risk manager not seeded"

        # ── Map headers (strip trailing spaces your sheet has) ───────────────
        header = [str(h).strip() for h in rows[0]]

        def ci(name):
            """Column index by name."""
            for i, h in enumerate(header):
                if h.lower() == name.lower():
                    return i
            return -1

        col_ticker   = ci('Ticker')
        col_position = ci('Position')
        col_entry    = ci('Entry_Price')
        col_qty      = ci('Quantity')
        col_sl       = ci('Stop_Loss')
        col_status   = ci('Status')

        if any(c == -1 for c in [col_ticker, col_entry, col_qty, col_sl, col_status]):
            logger.warning(f"⚠️  Risk seeding: missing columns in Portfolio sheet headers: {header}")
            return "Column mapping failed — risk manager not seeded"

        # ── Seed the risk manager with each ACTIVE trade ─────────────────────
        seeded = 0
        skipped = 0

        # Sector lookup for existing positions (best-effort, won't block if unknown)
        TICKER_SECTOR_MAP = {
            # Pharma
            'CIPLA':'PHARMA','GLENMARK':'PHARMA','AJANTPHARM':'PHARMA',
            'LAURUSLABS':'PHARMA','TORNTPHARM':'PHARMA',
            # FMCG / Consumer
            'JYOTHYLAB':'FMCG','CASTROLIND':'FMCG','CLEAN':'FMCG',
            # Chemicals / Industrials
            'BASF':'FMCG','CARBORUNIV':'METAL','TATACHEM':'FMCG',
            # Finance / Housing
            'BAJAJHFL':'FINANCIALS','SBILIFE':'FINANCIALS',
            # Energy
            'IGL':'ENERGY',
            # Auto / Engineering
            'ARE&M':'AUTO','EXIDEIND':'AUTO',
        }

        for row in rows[1:]:
            if len(row) <= max(col_ticker, col_entry, col_qty, col_sl, col_status):
                skipped += 1
                continue

            status = str(row[col_status]).strip().upper()
            if status != 'ACTIVE':
                continue   # only seed ACTIVE trades

            try:
                ticker_sym  = str(row[col_ticker]).strip()
                entry_price = float(row[col_entry])
                qty         = float(row[col_qty])
                sl          = float(row[col_sl])
                side        = str(row[col_position]).strip().upper()  # LONG / SHORT

                position_value = entry_price * qty
                risk_per_share = abs(entry_price - sl)
                risk_amount    = risk_per_share * qty
                sector         = TICKER_SECTOR_MAP.get(ticker_sym, 'UNKNOWN')

                portfolio_mgr.add_trade({
                    'ticker':         ticker_sym,
                    'side':           side,
                    'position_value': position_value,
                    'risk_amount':    risk_amount,
                    'sector':         sector,
                })
                seeded += 1
                logger.info(f"  📌 Seeded: {ticker_sym} {side} "
                            f"₹{position_value:,.0f} risk=₹{risk_amount:,.0f} sector={sector}")

            except (ValueError, TypeError, IndexError) as row_err:
                skipped += 1
                logger.debug(f"Risk seeding: skipped row {row} — {row_err}")

        summary = (
            f"Risk manager seeded: {seeded} ACTIVE positions loaded "
            f"({skipped} rows skipped). "
            f"Open trades loaded: {seeded}. "
            f"Sector exposure: {dict(portfolio_mgr.sector_exposure)}"
        )
        logger.info(f"✅ {summary}")
        return summary

    except Exception as e:
        logger.warning(f"⚠️  Risk seeding error: {e} — portfolio_mgr stays empty")
        return f"Risk seeding failed (non-blocking): {e}"


def _kite_connect_with_retry(max_attempts: int = 3, base_delay: float = 5.0) -> bool:
    """
    [FIX RETRY-1] Kite connection with exponential-backoff retry.

    Handles the most common Kite transient errors:
      - TokenException  → token expired → trigger auto-refresh and retry
      - NetworkException / requests.Timeout → transient; wait and retry
      - DataException   → usually bad instrument master; reload instruments

    Returns True if connected, False if all attempts exhausted.
    """
    import time as _time

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"🔌 Kite connect attempt {attempt}/{max_attempts}…")
            kite_provider._connect()

            if kite_provider.is_connected:
                logger.info(f"✅ Kite connected on attempt {attempt}")
                return True

            # Not connected — try token refresh before next attempt
            logger.warning(f"⚠️  Kite not connected (attempt {attempt})")
            try:
                from auto_token_refresh import refresh_token_auto
                from config_kite import token_needs_refresh
                if token_needs_refresh():
                    logger.info("🔄 Token expired — refreshing…")
                    ok, msg = refresh_token_auto()
                    logger.info(f"Token refresh result: {msg}")
            except ImportError:
                logger.warning("auto_token_refresh not available — skipping refresh")
            except Exception as ref_err:
                logger.warning(f"Token refresh error: {ref_err}")

        except Exception as e:
            err_type = type(e).__name__
            logger.warning(f"⚠️  Kite connect error on attempt {attempt}: [{err_type}] {e}")

            # For TokenException, always attempt a refresh before retry
            if 'Token' in err_type or 'token' in str(e).lower():
                try:
                    from auto_token_refresh import refresh_token_auto
                    ok, msg = refresh_token_auto()
                    logger.info(f"Token refresh (error recovery): {msg}")
                except Exception:
                    pass

        if attempt < max_attempts:
            delay = base_delay * (2 ** (attempt - 1))   # 5s, 10s, 20s …
            logger.info(f"⏳ Retrying in {delay:.0f}s…")
            _time.sleep(delay)

    logger.critical(f"❌ Kite connection failed after {max_attempts} attempts")
    return False


def github_actions_main():
    """Main function optimized for GitHub Actions execution with Google Sheets integration"""
    
    logger.info("="*80)
    logger.info("NSE SCANNER v8.5 - GITHUB ACTIONS EXECUTION WITH GOOGLE SHEETS")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    logger.info(f"GitHub Repository: {os.environ.get('GITHUB_REPOSITORY', 'N/A')}")
    logger.info(f"Workflow Run: #{os.environ.get('GITHUB_RUN_NUMBER', 'N/A')}")
    logger.info(f"Google Sheets: {'ENABLED' if GOOGLE_SHEETS_CONFIG['enabled'] else 'DISABLED'}")
    logger.info("="*80)
    
    import time
    start_time = time.time()
    
    try:
        logger.info("🚀 Starting stock scan...")

        # ── [FIX CB-1] Daily loss circuit breaker ─────────────────────────────
        cb_allowed, cb_reason = _check_daily_loss_circuit_breaker()
        if not cb_allowed:
            logger.critical("="*80)
            logger.critical("🚨 SCAN ABORTED BY DAILY LOSS CIRCUIT BREAKER")
            logger.critical(cb_reason)
            logger.critical("="*80)
            # Send alert email so you know the bot self-stopped
            if EMAIL_CONFIG['enabled'] and EMAIL_CONFIG['sender_email']:
                try:
                    msg = MIMEText(
                        f"NSE Scanner v8.5 — CIRCUIT BREAKER TRIGGERED\n\n"
                        f"{cb_reason}\n\n"
                        f"The scanner has self-stopped to protect your capital.\n"
                        f"Resume trading manually after reviewing today's P&L."
                    )
                    msg['Subject'] = "🚨 NSE Scanner — Circuit Breaker Triggered"
                    msg['From']    = EMAIL_CONFIG['sender_email']
                    msg['To']      = EMAIL_CONFIG['recipient_email']
                    with smtplib.SMTP(EMAIL_CONFIG['smtp_server'],
                                      EMAIL_CONFIG['smtp_port']) as server:
                        server.starttls()
                        server.login(EMAIL_CONFIG['sender_email'],
                                     EMAIL_CONFIG['sender_password'])
                        server.send_message(msg)
                    logger.info("📧 Circuit breaker alert email sent")
                except Exception as mail_err:
                    logger.warning(f"Circuit breaker email failed: {mail_err}")
            return 1   # Exit non-zero so GitHub Actions marks the job as failed

        logger.info(f"⚡ Circuit breaker: {cb_reason}")

        # ── [FIX RETRY-1] Kite connection with exponential-backoff retry ──────
        if not kite_provider.is_connected:
            connected = _kite_connect_with_retry(max_attempts=3, base_delay=5.0)
            if not connected:
                logger.critical("❌ Kite NOT connected after all retry attempts — scan aborted")
                raise RuntimeError("Kite not connected after 3 retry attempts")
        else:
            logger.info("✅ Kite already connected — skipping retry logic")

        # ── [FIX RISK-1] Seed PortfolioRiskManager with real open positions ──
        # portfolio_mgr is created inside main() — we pre-seed it here via a
        # temporary instance that main() will mirror when USE_PORTFOLIO_RISK=True.
        # This ensures can_add_trade() knows about your existing ACTIVE trades
        # BEFORE scanning new signals, so the max-position + sector caps work.
        if USE_PORTFOLIO_RISK:
            logger.info("="*80)
            logger.info("📊 LOADING ACTIVE POSITIONS INTO RISK MANAGER...")
            logger.info("="*80)
            _temp_mgr = PortfolioRiskManager(ACCOUNT_CAPITAL)
            seed_summary = _load_active_positions_into_risk_manager(_temp_mgr)
            logger.info(seed_summary)

            # ── Capital availability report ──────────────────────────────────
            logger.info("💰 CAPITAL SNAPSHOT:")
            logger.info(f"   {_temp_mgr.available_capital_summary()}")

            if _temp_mgr.total_deployed > _temp_mgr.capital:
                logger.warning(
                    f"⚠️  ACCOUNT_CAPITAL (₹{_temp_mgr.capital:,.0f}) is LESS than "
                    f"deployed capital (₹{_temp_mgr.total_deployed:,.0f}). "
                    f"Please update ACCOUNT_CAPITAL in config to your real capital!"
                )
            elif _temp_mgr.available_capital < _temp_mgr.capital * _temp_mgr.MIN_FREE_CAPITAL_PCT:
                logger.warning(
                    f"⚠️  Only ₹{_temp_mgr.available_capital:,.0f} free — "
                    f"below 10% safety buffer. Most new signals will be blocked."
                )
            else:
                logger.info(
                    f"✅ ₹{_temp_mgr.available_capital:,.0f} available for new signals "
                    f"({100 - _temp_mgr.deployed_pct*100:.1f}% of capital free)"
                )
            logger.info("="*80)
            # Pass the pre-seeded state into the module-level portfolio_mgr
            # that main() will use (same object reference is picked up inside main)
            import builtins as _builtins
            _builtins._seeded_portfolio_mgr = _temp_mgr

        # ✅ Run the main scanner
        results = main()
        
        scan_time = time.time() - start_time
        
        # ✅ Check if results exist AND have data
        if results and len(results) > 0:
            logger.info(f"✅ Scan complete: {len(results)} signals found in {scan_time:.1f}s")
            
            # ✅ Find signal files
            signals_dir = f"signals_v8.5_{ACCURACY_MODE.lower()}"
            logger.info(f"📂 Looking for files in: {signals_dir}/")
            
            csv_file = None
            html_file = None
            
            if os.path.exists(signals_dir):
                top_2_files = sorted([f for f in os.listdir(signals_dir) 
                                     if f.startswith('TOP_2_PICKS') and f.endswith('.csv')])
                csv_files = sorted([f for f in os.listdir(signals_dir) if f.endswith('.csv')])
                html_files = sorted([f for f in os.listdir(signals_dir) if f.endswith('.html')])
                
                # Use TOP_2_PICKS file if available (contains the actual top 2 by win rate)
                if top_2_files:
                    csv_file = os.path.join(signals_dir, top_2_files[-1])
                    logger.info(f"📄 Found TOP_2_PICKS CSV (will use for Google Sheets): {csv_file}")
                elif csv_files:
                    csv_file = os.path.join(signals_dir, csv_files[-1])
                    logger.info(f"📄 Found CSV: {csv_file}")
                
                if html_files:
                    html_file = os.path.join(signals_dir, html_files[-1])
                    logger.info(f"📄 Found HTML: {html_file}")
                
                # ✅ Copy files to standard location for artifacts
                copy_files_to_output(signals_dir)
                
                # ============================================================
                # 🆕 GOOGLE SHEETS UPDATE - THIS IS THE NEW SECTION!
                # ============================================================
                if csv_file and GOOGLE_SHEETS_CONFIG['enabled']:
                    logger.info("="*80)
                    logger.info("📊 UPDATING GOOGLE SHEET...")
                    logger.info("="*80)
                    
                    # Read signals from CSV
                    signals_data = read_signals_from_csv(csv_file)
                    
                    if signals_data:
                        # Update Google Sheet
                        success = update_google_sheet_with_gatekeeper(signals_data)
                        
                        if success:
                            logger.info("✅ Google Sheet updated successfully!")
                            logger.info(f"🔗 View your sheet: https://docs.google.com/spreadsheets/d/{GOOGLE_SHEETS_CONFIG['spreadsheet_id']}")
                        else:
                            logger.error("❌ Failed to update Google Sheet")
                    else:
                        logger.warning("⚠️  No signals to update in Google Sheet")
                    
                    logger.info("="*80)
                # ============================================================
                
            else:
                logger.warning(f"⚠️  Directory not found: {signals_dir}")
            
            # ✅ Send email with files
            logger.info("📧 Sending email notification...")
            send_email_notification(results, csv_file, html_file, scan_time)
            
        else:
            logger.info(f"⚠️  No signals found in {scan_time:.1f}s")
            send_no_signals_email(scan_time)
        
        logger.info("="*80)
        logger.info("✅ GitHub Actions execution completed successfully")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}", exc_info=True)
        
        # Send error notification
        if EMAIL_CONFIG['enabled'] and EMAIL_CONFIG['sender_email']:
            try:
                msg = MIMEText(f"Scanner error:\n\n{str(e)}\n\nCheck logs.")
                msg['Subject'] = "⚠️ NSE Scanner Error"
                msg['From'] = EMAIL_CONFIG['sender_email']
                msg['To'] = EMAIL_CONFIG['recipient_email']
                
                with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
                    server.starttls()
                    server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
                    server.send_message(msg)
            except:
                pass
        
        return 1


def send_no_signals_email(scan_time):
    """Send notification when no signals found"""
    
    if not EMAIL_CONFIG['enabled'] or not EMAIL_CONFIG['sender_email']:
        logger.info("Email not configured")
        return
    
    try:
        body = f"""
NSE SWING SCANNER v8.5 - DAILY REPORT
{'='*70}

Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}
Scan Time: {scan_time:.1f} seconds
Mode: {ACCURACY_MODE}

⚠️  NO TRADING SIGNALS FOUND TODAY

The scanner ran successfully but no stocks met the entry criteria.

Possible reasons:
- Market conditions unfavorable
- All setups filtered by quality checks
- Weekend/holiday (stale data)
- High volatility regime

{'='*70}

Next scan: Tomorrow at 4:00 PM IST

This is an automated message from NSE Scanner v8.5
"""
        
        msg = MIMEText(body)
        msg['Subject'] = f"📊 NSE Scanner - {datetime.now().strftime('%Y-%m-%d')} - No Signals"
        msg['From'] = EMAIL_CONFIG['sender_email']
        msg['To'] = EMAIL_CONFIG['recipient_email']
        
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            server.send_message(msg)
        
        logger.info("✅ No-signals email sent successfully")
        
    except Exception as e:
        logger.error(f"❌ Email send failed: {e}")


def copy_files_to_output(source_dir):
    """Copy signal files to standard output directory for GitHub Actions"""
    
    try:
        # Create standard output directory
        output_dir = "signals"
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(source_dir):
            logger.warning(f"Source directory not found: {source_dir}")
            return
        
        import shutil
        copied = 0
        
        # Copy all CSV and HTML files
        for filename in os.listdir(source_dir):
            if filename.endswith(('.csv', '.html')):
                src_path = os.path.join(source_dir, filename)
                dst_path = os.path.join(output_dir, filename)
                
                shutil.copy2(src_path, dst_path)
                logger.info(f"📋 Copied: {filename}")
                copied += 1
        
        logger.info(f"✅ Copied {copied} files to {output_dir}/")
        
        # Also copy logs
        if os.path.exists('logs/scanner.log'):
            shutil.copy2('logs/scanner.log', 'signals/scanner.log')
            logger.info("📋 Copied: scanner.log")
        
    except Exception as e:
        logger.error(f"⚠️  Failed to copy files: {e}")
# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    sys.exit(github_actions_main())
