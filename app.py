"""
🧠 SMART PORTFOLIO MONITOR v6.0 - COMPLETE EDITION
==================================================
"""
import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import hashlib
import time
import json
from typing import Tuple, Optional, Dict, List, Any
from pathlib import Path
from zoneinfo import ZoneInfo
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import streamlit-autorefresh
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ============================================================================
# ZERODHA KITE INTEGRATION
# ============================================================================
try:
    from kite_connector import kite_session, HAS_KITE
    from config_kite import (
        is_configured as kite_is_configured,
        get_login_url, load_token, save_token, clear_token
    )
except ImportError:
    HAS_KITE = False
    kite_session = None
    logger.warning("Kite connector not found. Create kite_connector.py")
# ============================================================================
# GOOGLE SHEETS API SETUP
# ============================================================================

try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_GSPREAD = True
except ImportError:
    HAS_GSPREAD = False
    # DO NOT call st.warning() here - set_page_config hasn't been called yet!

# ✅ FIX: Get credentials WITHOUT any st.* calls
credentials = None
_credential_error = None  # Store error message for later display

try:
    if 'GCP_SA_KEY' in os.environ:
        service_account_info = json.loads(os.environ.get('GCP_SA_KEY'))
        credentials = Credentials.from_service_account_info(
            service_account_info, 
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
        )
    else:
        _credential_error = "GCP_SA_KEY not found in Environment/Secrets"
except Exception as e:
    _credential_error = f"Failed to load credentials: {e}"

# ============================================================================
# SAFE UTILITY FUNCTIONS
# ============================================================================

def safe_divide(numerator, denominator, default=0.0):
    """Safe division that handles zero and NaN"""
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        result = numerator / denominator
        return default if pd.isna(result) or np.isinf(result) else result
    except (TypeError, ValueError, ZeroDivisionError, FloatingPointError):
        return default

def safe_float(value, default=0.0):
    """Safely convert value to float"""
    try:
        result = float(value)
        return default if pd.isna(result) else result
    except (TypeError, ValueError, ZeroDivisionError) as e:
        logging.warning(f"Error in calculation: {e}")
        return default


def round_to_tick_size(price):
    """
    Round price according to NSE tick size rules:
    - Price >= 1000: Round to 0.05 (₹2500.00, ₹2500.05, ₹2500.10...)
    - Price >= 10: Round to 0.05 (₹500.00, ₹500.05, ₹500.10...)
    - Price >= 1: Round to 0.05 (₹5.00, ₹5.05, ₹5.10...)
    - Price < 1: Round to 0.01 (₹0.20, ₹0.21, ₹0.22... for penny stocks)
    
    Returns: Properly rounded price as float
    """
    if pd.isna(price) or price is None:
        return 0.0
    
    price = float(price)
    
    if price < 0:
        return 0.0
    
    # NSE Tick Size Rules
    if price < 1:
        # Penny stocks: Round to 0.01 (paise level)
        return round(price / 0.01) * 0.01
    else:
        # All other stocks: Round to 0.05
        return round(price / 0.05) * 0.05

# ============================================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND!)
# ============================================================================
st.set_page_config(
    page_title="Smart Portfolio Monitor v6.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .critical-box {
        background: linear-gradient(135deg, #dc3545, #c82333);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
    .success-box {
        background: linear-gradient(135deg, #28a745, #218838);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #ffc107, #e0a800);
        color: black;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
    .info-box {
        background: linear-gradient(135deg, #17a2b8, #138496);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 5px 0;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DEFERRED STARTUP WARNINGS (Now safe to call st.* functions)
# ============================================================================
if not HAS_GSPREAD:
    st.warning("⚠️ Install gspread: pip install gspread google-auth")

if _credential_error:
    st.error(f"❌ {_credential_error}")




TRADE_HISTORY_FILE = "trade_history.json"
PERFORMANCE_STATS_FILE = "performance_stats.json"


def save_trade_history():
    """Save trade history to file for persistence across sessions"""
    try:
        # Convert datetime objects to strings for JSON
        serializable_history = []
        for trade in st.session_state.trade_history:
            trade_copy = trade.copy()
            if isinstance(trade_copy.get('timestamp'), datetime):
                trade_copy['timestamp'] = trade_copy['timestamp'].isoformat()
            serializable_history.append(trade_copy)
        
        with open(TRADE_HISTORY_FILE, 'w') as f:
            json.dump(serializable_history, f, indent=2, default=str)
        
        with open(PERFORMANCE_STATS_FILE, 'w') as f:
            json.dump(st.session_state.performance_stats, f, indent=2)
        
        logger.info(
            f"Trade history saved: {len(serializable_history)} trades"
        )
    except Exception as e:
        logger.error(f"Failed to save trade history: {e}")


def load_trade_history():
    """Load trade history from file"""
    history = []
    stats = {
        'total_trades': 0, 'wins': 0, 'losses': 0,
        'total_profit': 0, 'total_loss': 0
    }
    
    try:
        if Path(TRADE_HISTORY_FILE).exists():
            with open(TRADE_HISTORY_FILE, 'r') as f:
                raw_history = json.load(f)
            
            # Convert timestamp strings back to datetime
            for trade in raw_history:
                if isinstance(trade.get('timestamp'), str):
                    try:
                        trade['timestamp'] = datetime.fromisoformat(
                            trade['timestamp']
                        )
                    except (ValueError, TypeError):
                        trade['timestamp'] = datetime.now()
                history.append(trade)
            
            logger.info(f"Trade history loaded: {len(history)} trades")
    except Exception as e:
        logger.error(f"Failed to load trade history: {e}")
    
    try:
        if Path(PERFORMANCE_STATS_FILE).exists():
            with open(PERFORMANCE_STATS_FILE, 'r') as f:
                stats = json.load(f)
            logger.info("Performance stats loaded")
    except Exception as e:
        logger.error(f"Failed to load performance stats: {e}")
    
    return history, stats

def init_session_state():
    """Initialize all session state variables"""
    # Load persistent data
    saved_history, saved_stats = load_trade_history()
    
    defaults = {
        'email_sent_alerts': {},
        'last_email_time': {},
        'email_log': [],
        'trade_history': saved_history,
        'portfolio_values': [],
        'performance_stats': saved_stats,
        'drawdown_history': [],
        'peak_portfolio_value': 0,
        'current_drawdown': 0,
        'max_drawdown': 0,
        'partial_exits': {},
        'holding_periods': {},
        'last_api_call': {},
        'api_call_count': 0,
        'correlation_matrix': None,
        'last_correlation_calc': None,
        'kite_connected': False,
        'kite_mode': 'PAPER',
        'kite_order_log': [],
        'kite_daily_order_count': 0
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_ist_now():
    """Get current IST time"""
    return datetime.now(ZoneInfo("Asia/Kolkata"))

def is_market_hours():
    """Check if market is open"""
    ist_now = get_ist_now()
    
    if ist_now.weekday() >= 5:
        return False, "WEEKEND", "Markets closed for weekend", "🔴"
    
    market_open = datetime.strptime("09:15", "%H:%M").time()
    market_close = datetime.strptime("15:30", "%H:%M").time()
    current_time = ist_now.time()
    
    if current_time < market_open:
        return False, "PRE-MARKET", f"Opens at 09:15 IST", "🟡"
    elif current_time > market_close:
        return False, "CLOSED", "Market closed for today", "🔴"
    else:
        return True, "OPEN", f"Closes at 15:30 IST", "🟢"

# ============================================================================
# GAP 1: MARKET HEALTH CHECK (NIFTY + VIX)
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_market_health():
    """
    Analyze NIFTY 50 and India VIX to determine overall market health
    Returns: dict with status, message, color, action, metrics
    """
    try:
        # Get NIFTY 50 data
        nifty = yf.Ticker("^NSEI")
        nifty_df = nifty.history(period="1mo")
        
        if nifty_df.empty:
            return None
        
        nifty_price = float(nifty_df['Close'].iloc[-1])
        nifty_prev = float(nifty_df['Close'].iloc[-2]) if len(nifty_df) > 1 else nifty_price
        nifty_change = ((nifty_price - nifty_prev) / nifty_prev) * 100
        
        # Calculate NIFTY indicators
        nifty_sma20 = nifty_df['Close'].rolling(20).mean().iloc[-1]
        nifty_sma50 = nifty_df['Close'].rolling(50).mean().iloc[-1] if len(nifty_df) >= 50 else nifty_sma20
        nifty_rsi = calculate_rsi(nifty_df['Close']).iloc[-1]
        
        if pd.isna(nifty_rsi):
            nifty_rsi = 50
        
        # Get India VIX (Volatility Index)
        vix = yf.Ticker("^INDIAVIX")
        vix_df = vix.history(period="5d")
        vix_value = float(vix_df['Close'].iloc[-1]) if not vix_df.empty else 15
        
        # Calculate Market Health Score (0-100)
        health_score = 50  # Start neutral
        
        # NIFTY Price vs SMA20 (0-20 points)
        if nifty_price > nifty_sma20:
            health_score += 15
        else:
            health_score -= 15
        
        # NIFTY Price vs SMA50 (0-15 points)
        if nifty_price > nifty_sma50:
            health_score += 10
        else:
            health_score -= 10
        
        # NIFTY RSI (0-20 points)
        if nifty_rsi > 55:
            health_score += 15
        elif nifty_rsi > 45:
            health_score += 5
        elif nifty_rsi < 35:
            health_score -= 15
        elif nifty_rsi < 45:
            health_score -= 10
        
        # VIX Level (0-25 points)
        if vix_value < 12:
            health_score += 20  # Very low volatility = good
        elif vix_value < 15:
            health_score += 10
        elif vix_value > 25:
            health_score -= 20  # High volatility = bad
        elif vix_value > 18:
            health_score -= 10
        
        # NIFTY Trend (SMA20 vs SMA50) (0-20 points)
        if nifty_sma20 > nifty_sma50:
            health_score += 10  # Golden cross
        else:
            health_score -= 10  # Death cross
        
        # Cap between 0-100
        health_score = max(0, min(100, health_score))
        
        # Determine Status
        if health_score >= 70:
            status = "BULLISH"
            color = "#28a745"
            icon = "🟢"
            action = "✅ Good environment for trading"
            sl_adjustment = "NORMAL"
        elif health_score >= 50:
            status = "NEUTRAL"
            color = "#ffc107"
            icon = "🟡"
            action = "⚠️ Be selective with new positions"
            sl_adjustment = "NORMAL"
        elif health_score >= 30:
            status = "WEAK"
            color = "#fd7e14"
            icon = "🟠"
            action = "⚠️ Tighten stop losses, avoid new longs"
            sl_adjustment = "TIGHTEN"
        else:
            status = "BEARISH"
            color = "#dc3545"
            icon = "🔴"
            action = "🚨 HIGH RISK - Consider reducing exposure"
            sl_adjustment = "AGGRESSIVE"
        
        # Build message
        message = f"NIFTY: ₹{nifty_price:,.0f} ({nifty_change:+.2f}%) | RSI: {nifty_rsi:.0f} | VIX: {vix_value:.1f}"
        
        return {
            'status': status,
            'health_score': health_score,
            'message': message,
            'color': color,
            'icon': icon,
            'action': action,
            'sl_adjustment': sl_adjustment,
            'nifty_price': nifty_price,
            'nifty_change': nifty_change,
            'nifty_rsi': nifty_rsi,
            'nifty_sma20': nifty_sma20,
            'nifty_sma50': nifty_sma50,
            'vix': vix_value,
            'above_sma20': nifty_price > nifty_sma20,
            'above_sma50': nifty_price > nifty_sma50
        }
    
    except Exception as e:
        logger.error(f"Market health check failed: {e}")
        return None
# ============================================================================
# GAP 2: EMERGENCY EXIT DETECTOR
# ============================================================================

def detect_emergency_exit(result, market_health):
    """
    Detect critical exit conditions that override normal analysis
    Returns: (is_emergency, reasons, urgency_level)
    """
    emergency = False
    reasons = []
    urgency = "NORMAL"
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EMERGENCY CONDITION 1: Market Crash + Losing Position
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if market_health and market_health['status'] in ['BEARISH', 'WEAK']:
        if result['pnl_percent'] < -2:
            emergency = True
            urgency = "CRITICAL"
            reasons.append(f"🚨 Market {market_health['status']} + Position down {result['pnl_percent']:.1f}%")
        elif result['pnl_percent'] < 0:
            urgency = "HIGH"
            reasons.append(f"⚠️ Weak market + Position negative ({result['pnl_percent']:.1f}%)")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EMERGENCY CONDITION 2: Gap Down Below SL
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if result['position_type'] == 'LONG':
        if result['day_low'] < result['stop_loss'] * 0.98:
            emergency = True
            urgency = "CRITICAL"
            reasons.append(f"🚨 Gap down: Day low ₹{result['day_low']:.2f} below SL ₹{result['stop_loss']:.2f}")
    else:  # SHORT
        if result['day_high'] > result['stop_loss'] * 1.02:
            emergency = True
            urgency = "CRITICAL"
            reasons.append(f"🚨 Gap up: Day high ₹{result['day_high']:.2f} above SL ₹{result['stop_loss']:.2f}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EMERGENCY CONDITION 3: VIX Spike + High SL Risk
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if market_health and market_health['vix'] > 25:
        if result['sl_risk'] > 60:
            emergency = True
            urgency = "CRITICAL"
            reasons.append(f"🚨 VIX spike ({market_health['vix']:.1f}) + SL Risk {result['sl_risk']}%")
        elif result['sl_risk'] > 50:
            urgency = "HIGH"
            reasons.append(f"⚠️ High VIX ({market_health['vix']:.1f}) + SL Risk {result['sl_risk']}%")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EMERGENCY CONDITION 4: Heavy Selling Volume + Negative
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if result['position_type'] == 'LONG':
        if result['volume_signal'] in ['STRONG_SELLING', 'SELLING']:
            if result['volume_ratio'] > 2.5 and result['pnl_percent'] < -1:
                emergency = True
                urgency = "HIGH"
                reasons.append(f"⚠️ Heavy selling volume ({result['volume_ratio']:.1f}x) + Position down")
    else:  # SHORT
        if result['volume_signal'] in ['STRONG_BUYING', 'BUYING']:
            if result['volume_ratio'] > 2.5 and result['pnl_percent'] < -1:
                emergency = True
                urgency = "HIGH"
                reasons.append(f"⚠️ Heavy buying volume ({result['volume_ratio']:.1f}x) + Position down")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EMERGENCY CONDITION 5: All Timeframes Against Position
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if result['mtf_alignment'] < 20 and result['sl_risk'] > 50:
        emergency = True
        urgency = "HIGH"
        reasons.append(f"⚠️ All timeframes against position (MTF: {result['mtf_alignment']}%)")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EMERGENCY CONDITION 6: Breakdown Below Support with Volume
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if result['position_type'] == 'LONG':
        if result['current_price'] < result['support'] * 0.99:
            if result['volume_ratio'] > 1.5:
                emergency = True
                urgency = "HIGH"
                reasons.append(f"⚠️ Breakdown below support ₹{result['support']:.2f} with volume")
    else:  # SHORT
        if result['current_price'] > result['resistance'] * 1.01:
            if result['volume_ratio'] > 1.5:
                emergency = True
                urgency = "HIGH"
                reasons.append(f"⚠️ Breakout above resistance ₹{result['resistance']:.2f} with volume")
    
    return emergency, reasons, urgency    

# ============================================================================
# GAP 3: STOCK-SPECIFIC WIN RATE ANALYSIS
# ============================================================================

def get_stock_performance_history(ticker):
    """
    Analyze historical performance for specific stock
    Returns: dict with win_rate, trade_count, avg_win, avg_loss, recommendation
    """
    # Get trades for this ticker
    stock_trades = [t for t in st.session_state.trade_history if t['ticker'] == ticker]
    
    if len(stock_trades) < 3:
        return {
            'has_history': False,
            'trade_count': len(stock_trades),
            'message': f"Only {len(stock_trades)} trade(s) logged - need 3+ for analysis"
        }
    
    # Calculate statistics
    total_trades = len(stock_trades)
    wins = sum(1 for t in stock_trades if t['win'])
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100
    
    winning_trades = [t for t in stock_trades if t['win']]
    losing_trades = [t for t in stock_trades if not t['win']]
    
    avg_win = sum(t['pnl'] for t in winning_trades) / max(wins, 1)
    avg_loss = sum(abs(t['pnl']) for t in losing_trades) / max(losses, 1)
    
    # Calculate expectancy
    expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
    
    # Profit factor
    total_profit = sum(t['pnl'] for t in winning_trades)
    total_loss = sum(abs(t['pnl']) for t in losing_trades)
    profit_factor = total_profit / max(total_loss, 1)
    
    # Determine recommendation
    if win_rate < 30:
        quality = "POOR"
        color = "#dc3545"
        icon = "🔴"
        recommendation = "⚠️ AVOID - Very low win rate"
    elif win_rate < 40:
        quality = "WEAK"
        color = "#fd7e14"
        icon = "🟠"
        recommendation = "⚠️ CAUTION - Below average performance"
    elif win_rate < 50:
        quality = "AVERAGE"
        color = "#ffc107"
        icon = "🟡"
        recommendation = "ℹ️ Acceptable - Monitor closely"
    elif win_rate < 60:
        quality = "GOOD"
        color = "#28a745"
        icon = "🟢"
        recommendation = "✅ Good track record"
    else:
        quality = "EXCELLENT"
        color = "#20c997"
        icon = "💚"
        recommendation = "✅ Excellent performance!"
    
    # Check expectancy
    if expectancy < 0:
        recommendation = "🚨 NEGATIVE EXPECTANCY - Stop trading this stock!"
        quality = "LOSING"
        color = "#dc3545"
    
    return {
        'has_history': True,
        'trade_count': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
        'profit_factor': profit_factor,
        'quality': quality,
        'color': color,
        'icon': icon,
        'recommendation': recommendation
    }

# ============================================================================
# GAP 4: CHART PATTERN DETECTION
# ============================================================================

def detect_chart_patterns(df, current_price):
    """
    Detect common chart patterns
    Returns: list of detected patterns with signals
    """
    patterns = []
    
    if len(df) < 30:
        return patterns
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PATTERN 1: DOUBLE TOP (Bearish)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    last_30 = df.tail(30)
    highs_30 = last_30['High']
    
    # Find two highest peaks
    sorted_highs = highs_30.nlargest(5)
    if len(sorted_highs) >= 2:
        peak1 = sorted_highs.iloc[0]
        peak2 = sorted_highs.iloc[1]
        
        # Check if peaks are similar (within 2%)
        if abs(peak1 - peak2) / peak1 < 0.02:
            # Check if current price is below peaks
            if current_price < peak1 * 0.98:
                patterns.append({
                    'name': 'DOUBLE TOP',
                    'signal': 'BEARISH',
                    'strength': 'HIGH',
                    'icon': '📉',
                    'description': f'Resistance at ₹{peak1:.2f} tested twice',
                    'action': 'Watch for breakdown - potential reversal'
                })
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PATTERN 2: DOUBLE BOTTOM (Bullish)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    lows_30 = last_30['Low']
    sorted_lows = lows_30.nsmallest(5)
    
    if len(sorted_lows) >= 2:
        bottom1 = sorted_lows.iloc[0]
        bottom2 = sorted_lows.iloc[1]
        
        if abs(bottom1 - bottom2) / bottom1 < 0.02:
            if current_price > bottom1 * 1.02:
                patterns.append({
                    'name': 'DOUBLE BOTTOM',
                    'signal': 'BULLISH',
                    'strength': 'HIGH',
                    'icon': '📈',
                    'description': f'Support at ₹{bottom1:.2f} held twice',
                    'action': 'Watch for breakout - potential reversal'
                })
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PATTERN 3: BULLISH ENGULFING (Bullish)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if len(df) >= 2:
        prev_candle = df.iloc[-2]
        curr_candle = df.iloc[-1]
        
        # Previous red, current green, and current engulfs previous
        if (prev_candle['Close'] < prev_candle['Open'] and
            curr_candle['Close'] > curr_candle['Open'] and
            curr_candle['Open'] < prev_candle['Close'] and
            curr_candle['Close'] > prev_candle['Open']):
            
            patterns.append({
                'name': 'BULLISH ENGULFING',
                'signal': 'BULLISH',
                'strength': 'MEDIUM',
                'icon': '🟢',
                'description': 'Strong reversal candle pattern',
                'action': 'Potential upward momentum'
            })
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PATTERN 4: BEARISH ENGULFING (Bearish)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if len(df) >= 2:
        prev_candle = df.iloc[-2]
        curr_candle = df.iloc[-1]
        
        if (prev_candle['Close'] > prev_candle['Open'] and
            curr_candle['Close'] < curr_candle['Open'] and
            curr_candle['Open'] > prev_candle['Close'] and
            curr_candle['Close'] < prev_candle['Open']):
            
            patterns.append({
                'name': 'BEARISH ENGULFING',
                'signal': 'BEARISH',
                'strength': 'MEDIUM',
                'icon': '🔴',
                'description': 'Strong reversal candle pattern',
                'action': 'Potential downward momentum'
            })
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PATTERN 5: ASCENDING TRIANGLE (Bullish)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if len(df) >= 20:
        last_20 = df.tail(20)
        
        # Check if highs are flat (resistance)
        highs_20 = last_20['High']
        high_max = highs_20.max()
        high_std = highs_20.std()
        
        # Check if lows are rising
        lows_20 = last_20['Low']
        low_trend = lows_20.iloc[-5:].mean() > lows_20.iloc[:5].mean()
        
        if high_std / high_max < 0.015 and low_trend:  # Flat top + rising lows
            patterns.append({
                'name': 'ASCENDING TRIANGLE',
                'signal': 'BULLISH',
                'strength': 'HIGH',
                'icon': '📐',
                'description': f'Breakout potential above ₹{high_max:.2f}',
                'action': 'Watch for volume spike on breakout'
            })
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PATTERN 6: DESCENDING TRIANGLE (Bearish)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if len(df) >= 20:
        last_20 = df.tail(20)
        
        # Check if lows are flat (support)
        lows_20 = last_20['Low']
        low_min = lows_20.min()
        low_std = lows_20.std()
        
        # Check if highs are falling
        highs_20 = last_20['High']
        high_trend = highs_20.iloc[-5:].mean() < highs_20.iloc[:5].mean()
        
        if low_std / low_min < 0.015 and high_trend:  # Flat bottom + falling highs
            patterns.append({
                'name': 'DESCENDING TRIANGLE',
                'signal': 'BEARISH',
                'strength': 'HIGH',
                'icon': '📐',
                'description': f'Breakdown potential below ₹{low_min:.2f}',
                'action': 'Watch for volume spike on breakdown'
            })
    
    return patterns

def send_email_alert(subject, html_content, sender, password, recipient):
    """
    Send email alert - Returns (success, message)
    """
    if not sender or not password or not recipient:
        log_email("❌ Missing email credentials")
        return False, "Missing email credentials"
    
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = recipient
        msg.attach(MIMEText(html_content, 'html'))
        
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=10)  # ✅ Add timeout
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())
        server.quit()
        
        log_email(f"✅ Email sent: {subject}")  # ✅ Add logging
        return True, "Email sent successfully"
    
    except smtplib.SMTPAuthenticationError:
        log_email("❌ SMTP Authentication failed")
        return False, "Authentication failed - check App Password"
    except smtplib.SMTPRecipientsRefused:
        log_email("❌ Invalid recipient email")
        return False, "Invalid recipient email address"
    except smtplib.SMTPException as e:
        log_email(f"❌ SMTP error: {str(e)}")
        return False, f"SMTP error: {str(e)}"
    except Exception as e:
        log_email(f"❌ Email failed: {str(e)}")
        return False, f"Email failed: {str(e)}"
    

def log_email(message):
    """Add to email log"""
    timestamp = get_ist_now().strftime("%H:%M:%S")
    st.session_state.email_log.append(f"[{timestamp}] {message}")
    if len(st.session_state.email_log) > 50:
        st.session_state.email_log = st.session_state.email_log[-50:]

def generate_alert_hash(ticker, alert_type, key_value=""):
    """Generate unique hash for an alert"""
    alert_string = f"{ticker}_{alert_type}_{key_value}_{get_ist_now().strftime('%Y%m%d')}"
    return hashlib.md5(alert_string.encode()).hexdigest()[:12]


def can_send_email(alert_hash, cooldown_minutes=15):
    """
    Check if enough time has passed since last email
    """
    if alert_hash not in st.session_state.last_email_time:
        return True
    
    last_sent = st.session_state.last_email_time[alert_hash]
    now = datetime.now()  # ✅ Use simple datetime
    
    try:
        # ✅ Handle timezone issues properly
        if isinstance(last_sent, datetime):
            # Remove timezone info for comparison
            if last_sent.tzinfo is not None:
                last_sent = last_sent.replace(tzinfo=None)
            if now.tzinfo is not None:
                now = now.replace(tzinfo=None)
            
            time_diff = (now - last_sent).total_seconds() / 60.0
        else:
            # Invalid last_sent, allow email
            return True
        
        # ✅ Debug log
        if time_diff < cooldown_minutes:
            logger.info(f"Email cooldown: {time_diff:.1f}/{cooldown_minutes} min for {alert_hash}")
        
        return time_diff >= cooldown_minutes
    
    except Exception as e:
        logger.error(f"Email cooldown check failed: {e}")
        return True  # Allow email on error
    

def mark_email_sent(alert_hash):
    """Mark an alert as sent"""
    st.session_state.last_email_time[alert_hash] = datetime.now()  # ✅ Use datetime.now()
    st.session_state.email_sent_alerts[alert_hash] = True
    logger.info(f"Email marked sent: {alert_hash} at {datetime.now().strftime('%H:%M:%S')}")

MAX_TRADE_HISTORY = 500


TRANSACTION_COSTS = {
    'brokerage_per_leg_pct': 0.03,     # Zerodha 0.03% or ₹20 max
    'brokerage_max_per_leg': 20,        # ₹20 cap per leg
    'stt_sell_pct': 0.1,                # 0.1% on sell side delivery
    'stt_buy_intraday_pct': 0.025,      # 0.025% intraday
    'exchange_fees_pct': 0.00345,       # NSE charges
    'gst_pct': 18,                      # 18% GST on brokerage
    'sebi_charges_pct': 0.0001,         # SEBI turnover fee
    'stamp_duty_buy_pct': 0.015,        # Stamp duty on buy
}


def calculate_transaction_costs(entry_price, exit_price, quantity, 
                                 product="CNC"):
    """
    Calculate actual transaction costs for a round-trip trade.
    
    Args:
        entry_price: Buy/Entry price
        exit_price: Sell/Exit price
        quantity: Number of shares
        product: "CNC" (delivery) or "MIS" (intraday)
    
    Returns:
        dict with individual costs and total
    """
    buy_value = entry_price * quantity
    sell_value = exit_price * quantity
    turnover = buy_value + sell_value
    
    # Brokerage (capped at ₹20 per leg for Zerodha)
    buy_brokerage = min(
        buy_value * TRANSACTION_COSTS['brokerage_per_leg_pct'] / 100,
        TRANSACTION_COSTS['brokerage_max_per_leg']
    )
    sell_brokerage = min(
        sell_value * TRANSACTION_COSTS['brokerage_per_leg_pct'] / 100,
        TRANSACTION_COSTS['brokerage_max_per_leg']
    )
    total_brokerage = buy_brokerage + sell_brokerage
    
    # STT
    if product == "CNC":
        stt = sell_value * TRANSACTION_COSTS['stt_sell_pct'] / 100
    else:
        stt = (
            sell_value * TRANSACTION_COSTS['stt_buy_intraday_pct'] / 100
        )
    
    # Exchange fees
    exchange_fees = (
        turnover * TRANSACTION_COSTS['exchange_fees_pct'] / 100
    )
    
    # GST on brokerage
    gst = total_brokerage * TRANSACTION_COSTS['gst_pct'] / 100
    
    # SEBI charges
    sebi = turnover * TRANSACTION_COSTS['sebi_charges_pct'] / 100
    
    # Stamp duty (on buy side)
    stamp_duty = (
        buy_value * TRANSACTION_COSTS['stamp_duty_buy_pct'] / 100
    )
    
    # Total
    total_cost = (
        total_brokerage + stt + exchange_fees + gst + sebi + stamp_duty
    )
    cost_pct = (total_cost / buy_value) * 100 if buy_value > 0 else 0
    
    return {
        'brokerage': round(total_brokerage, 2),
        'stt': round(stt, 2),
        'exchange_fees': round(exchange_fees, 2),
        'gst': round(gst, 2),
        'sebi': round(sebi, 2),
        'stamp_duty': round(stamp_duty, 2),
        'total_cost': round(total_cost, 2),
        'cost_pct': round(cost_pct, 4),
        'buy_value': round(buy_value, 2),
        'sell_value': round(sell_value, 2),
        'turnover': round(turnover, 2)
    }


MAX_TRADE_HISTORY = 500

def log_trade(ticker, entry_price, exit_price, quantity, position_type, 
              exit_reason):
    """
    Log completed trade with transaction costs.
    Saves to file for persistence.
    """
    # Calculate gross P&L
    if position_type == "LONG":
        gross_pnl = (exit_price - entry_price) * quantity
        gross_pnl_pct = (
            (exit_price - entry_price) / entry_price
        ) * 100
    else:
        gross_pnl = (entry_price - exit_price) * quantity
        gross_pnl_pct = (
            (entry_price - exit_price) / entry_price
        ) * 100
    
    # Calculate transaction costs
    costs = calculate_transaction_costs(
        entry_price, exit_price, quantity, "CNC"
    )
    
    # Net P&L (after costs)
    net_pnl = gross_pnl - costs['total_cost']
    net_pnl_pct = gross_pnl_pct - costs['cost_pct']
    
    trade = {
        'timestamp': get_ist_now(),
        'ticker': ticker,
        'type': position_type,
        'entry': entry_price,
        'exit': exit_price,
        'quantity': quantity,
        'gross_pnl': gross_pnl,
        'gross_pnl_pct': gross_pnl_pct,
        'transaction_costs': costs['total_cost'],
        'cost_pct': costs['cost_pct'],
        'cost_breakdown': costs,
        'pnl': net_pnl,           # Net P&L (after costs)
        'pnl_pct': net_pnl_pct,   # Net P&L % (after costs)
        'reason': exit_reason,
        'win': net_pnl > 0         # Win/loss based on NET P&L
    }
    
    st.session_state.trade_history.append(trade)
    if len(st.session_state.trade_history) > MAX_TRADE_HISTORY:
        st.session_state.trade_history = (
            st.session_state.trade_history[-MAX_TRADE_HISTORY:]
        )
    
    # Update stats (using NET P&L)
    stats = st.session_state.performance_stats
    stats['total_trades'] += 1
    
    if net_pnl > 0:
        stats['wins'] += 1
        stats['total_profit'] += net_pnl
    else:
        stats['losses'] += 1
        stats['total_loss'] += abs(net_pnl)
    
    # Track total costs
    if 'total_costs' not in stats:
        stats['total_costs'] = 0
    stats['total_costs'] += costs['total_cost']
    
    # Auto-save
    save_trade_history()
    
    log_email(
        f"📊 Trade logged: {ticker} | "
        f"Gross: ₹{gross_pnl:+,.0f} | "
        f"Costs: ₹{costs['total_cost']:.0f} | "
        f"Net: ₹{net_pnl:+,.0f}"
    )

def get_performance_stats():
    """Calculate performance statistics"""
    stats = st.session_state.performance_stats
    history = st.session_state.trade_history
    
    if stats['total_trades'] == 0:
        return None
    
    win_rate = (stats['wins'] / stats['total_trades'] * 100)
    avg_win = stats['total_profit'] / stats['wins'] if stats['wins'] > 0 else 0
    avg_loss = stats['total_loss'] / stats['losses'] if stats['losses'] > 0 else 0
    
    expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
    profit_factor = stats['total_profit'] / stats['total_loss'] if stats['total_loss'] > 0 else float('inf')
    
    return {
        'total_trades': stats['total_trades'],
        'wins': stats['wins'],
        'losses': stats['losses'],
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
        'profit_factor': profit_factor,
        'net_profit': stats['total_profit'] - stats['total_loss']
    }

def update_drawdown(current_portfolio_value):
    """Update drawdown tracking"""
    if current_portfolio_value > st.session_state.peak_portfolio_value:
        st.session_state.peak_portfolio_value = current_portfolio_value
    
    if st.session_state.peak_portfolio_value > 0:
        drawdown = ((st.session_state.peak_portfolio_value - current_portfolio_value) / 
                   st.session_state.peak_portfolio_value) * 100
        st.session_state.current_drawdown = drawdown
        
        if drawdown > st.session_state.max_drawdown:
            st.session_state.max_drawdown = drawdown
        
        # Store history
        st.session_state.drawdown_history.append({
            'timestamp': get_ist_now(),
            'value': current_portfolio_value,
            'drawdown': drawdown
        })
        
        # Keep last 1000 records
        if len(st.session_state.drawdown_history) > 1000:
            st.session_state.drawdown_history = st.session_state.drawdown_history[-1000:]

def rate_limited_api_call(ticker, min_interval=0.3):
    """
    Ensure minimum interval between API calls.
    Reduced from 1.0s to 0.3s to minimize UI blocking.
    """
    current_time = time.time()
    
    if ticker in st.session_state.last_api_call:
        elapsed = current_time - st.session_state.last_api_call[ticker]
        if elapsed < min_interval:
            # Use shorter sleep to reduce UI blocking
            sleep_time = min(min_interval - elapsed, 0.3)
            time.sleep(sleep_time)
    
    st.session_state.last_api_call[ticker] = time.time()
    st.session_state.api_call_count += 1
    return True

def get_stock_data_safe(ticker, period="6mo"):
    """Safely fetch stock data with rate limiting"""
    symbol = ticker if '.NS' in str(ticker) or '.BO' in str(ticker) else f"{ticker}.NS"
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            rate_limited_api_call(symbol)
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if not df.empty:
                df.reset_index(inplace=True)
                return df
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))  # Reduced backoff
                continue
            logger.error(f"API Error for {ticker}: {str(e)}")
            log_email(f"API Error for {ticker}: {str(e)}")
    
    return None
def get_realtime_price(ticker):
    """Get real-time price (15-min delayed on free yfinance)"""
    symbol = ticker if '.NS' in str(ticker) or '.BO' in str(ticker) else f"{ticker}.NS"
    try:
        rate_limited_api_call(symbol)
        stock = yf.Ticker(symbol)
        intraday = stock.history(period="1d", interval="5m")
        
        if not intraday.empty:
            return {
                'price': float(intraday['Close'].iloc[-1]),
                'high': float(intraday['High'].max()),
                'low': float(intraday['Low'].min()),
                'volume': int(intraday['Volume'].sum()),
                'last_update': str(intraday.index[-1]),
                'is_realtime': True
            }
    except Exception as e:
        logger.error(f"Realtime price error for {ticker}: {e}")
    return None


def monitor_intraday_sl(ticker, stop_loss, position_type):
    """
    Check if intraday price has touched/breached SL
    Even if it recovers by EOD, we should know!
    """
    symbol = ticker if '.NS' in str(ticker) or '.BO' in str(ticker) else f"{ticker}.NS"
    try:
        rate_limited_api_call(symbol)
        stock = yf.Ticker(symbol)
        intraday = stock.history(period="1d", interval="5m")
        
        if intraday.empty:
            return None
        
        day_low = float(intraday['Low'].min())
        day_high = float(intraday['High'].max())
        current = float(intraday['Close'].iloc[-1])
        
        result = {
            'current': current,
            'day_low': day_low,
            'day_high': day_high,
            'sl_touched': False,
            'sl_breached': False,
            'worst_breach': 0,
            'recovered': False,
            'close_below_sl': False  # ✅ For swing trading
        }
        
        if position_type == 'LONG':
            if day_low <= stop_loss:
                result['sl_touched'] = True
                result['sl_breached'] = current < stop_loss
                result['worst_breach'] = (
                    (stop_loss - day_low) / stop_loss
                ) * 100
                result['recovered'] = current > stop_loss
                # ✅ Swing trading: Only true breach if CLOSING below SL
                result['close_below_sl'] = current < stop_loss
        else:  # SHORT
            if day_high >= stop_loss:
                result['sl_touched'] = True
                result['sl_breached'] = current > stop_loss
                result['worst_breach'] = (
                    (day_high - stop_loss) / stop_loss
                ) * 100
                result['recovered'] = current < stop_loss
                result['close_below_sl'] = current > stop_loss
        
        return result
    except Exception as e:
        logger.error(f"Intraday SL monitor error for {ticker}: {e}")
        return None

def calculate_holding_period(entry_date):
    """Calculate holding period in days with multiple format support"""
    if entry_date is None or entry_date == '' or (isinstance(entry_date, float) and pd.isna(entry_date)):
        return 0
    
    if isinstance(entry_date, str):
        entry_date = entry_date.strip()
        
        # Try multiple date formats
        formats_to_try = [
            "%Y-%m-%d",
            "%d-%m-%Y", 
            "%d/%m/%Y",
            "%Y/%m/%d",
            "%d-%b-%Y",
            "%d %b %Y",
            "%Y-%m-%d %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
        ]
        
        parsed = None
        for fmt in formats_to_try:
            try:
                parsed = datetime.strptime(entry_date, fmt)
                break
            except ValueError:
                continue
        
        if parsed is None:
            log_email(f"Could not parse entry date: {entry_date}")
            return 0
        
        entry_date = parsed
    
    # Handle pandas Timestamp
    if hasattr(entry_date, 'to_pydatetime'):
        entry_date = entry_date.to_pydatetime()
    
    if isinstance(entry_date, datetime):
        now = get_ist_now()
        try:
            # Handle timezone-aware and naive datetime
            if entry_date.tzinfo is not None:
                delta = now - entry_date
            else:
                delta = now.replace(tzinfo=None) - entry_date
            return max(0, delta.days)
        except (TypeError, ValueError, AttributeError):
             return 0
    
    return 0

def get_tax_implication(holding_days, pnl):
    """Get tax implication based on holding period"""
    if pnl <= 0:
        return "Loss - Can be set off", "🟢"
    
    if holding_days >= 365:
        # LTCG - 10% above 1 lakh
        return "LTCG (10% above ₹1L)", "🟢"
    else:
        # STCG - 15%
        return "STCG (15%)", "🟡"

# ============================================================================
# TECHNICAL ANALYSIS FUNCTIONS
# ============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using Wilder's smoothing method"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Use Wilder's smoothing (EWM with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # Handle division by zero
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(
    prices: pd.Series, 
    fast: int = 12, 
    slow: int = 26, 
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp_fast = prices.ewm(span=fast, adjust=False).mean()
    exp_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = exp_fast - exp_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Use Wilder's smoothing
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return atr

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

# ============================================================================
# NEW: VOLATILITY-ADJUSTED SMART STOP LOSS
# ============================================================================

def calculate_smart_stop_loss(df, entry_price, current_price, position_type,
                               risk_percent=2.0, market_health=None):
    """
    Calculate stop loss that adapts to:
    1. Current ATR (volatility)
    2. Market conditions (VIX)
    3. Support/Resistance levels
    4. Recent price action
    """
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Method 1: ATR-based SL
    atr = calculate_atr(high, low, close, period=14).iloc[-1]
    if pd.isna(atr) or atr <= 0:
        atr = current_price * 0.02
    
    atr_multiplier = 2.0  # Default
    
    # Adjust multiplier based on market conditions
    if market_health:
        vix = market_health.get('vix', 15)
        if vix > 25:
            atr_multiplier = 1.5  # Tighter SL in high volatility
        elif vix > 20:
            atr_multiplier = 1.8
        elif vix < 12:
            atr_multiplier = 2.5  # Wider SL in low volatility
    
    if position_type == 'LONG':
        atr_sl = current_price - (atr * atr_multiplier)
    else:
        atr_sl = current_price + (atr * atr_multiplier)
    
    # Method 2: Support/Resistance based SL
    sr = find_support_resistance(df)
    
    if position_type == 'LONG':
        sr_sl = sr['nearest_support'] * 0.995
    else:
        sr_sl = sr['nearest_resistance'] * 1.005
    
    # Method 3: Percentage-based SL
    if position_type == 'LONG':
        pct_sl = entry_price * (1 - risk_percent / 100)
    else:
        pct_sl = entry_price * (1 + risk_percent / 100)
    
    # Method 4: Swing Low/High based
    if position_type == 'LONG':
        recent_lows = low.tail(10)
        swing_sl = float(recent_lows.min()) * 0.995
    else:
        recent_highs = high.tail(10)
        swing_sl = float(recent_highs.max()) * 1.005
    
    # SMART SELECTION
    if position_type == 'LONG':
        candidates = [atr_sl, sr_sl, pct_sl, swing_sl]
        valid_candidates = [sl for sl in candidates if sl < current_price]
        
        if valid_candidates:
            smart_sl = max(valid_candidates)
            min_distance = current_price - atr
            if smart_sl > min_distance:
                smart_sl = min_distance
        else:
            smart_sl = atr_sl
    else:
        candidates = [atr_sl, sr_sl, pct_sl, swing_sl]
        valid_candidates = [sl for sl in candidates if sl > current_price]
        
        if valid_candidates:
            smart_sl = min(valid_candidates)
            max_distance = current_price + atr
            if smart_sl < max_distance:
                smart_sl = max_distance
        else:
            smart_sl = atr_sl
    
    return {
        'smart_sl': round_to_tick_size(smart_sl),
        'atr_sl': round_to_tick_size(atr_sl),
        'sr_sl': round_to_tick_size(sr_sl),
        'pct_sl': round_to_tick_size(pct_sl),
        'swing_sl': round_to_tick_size(swing_sl),
        'atr': atr,
        'atr_multiplier': atr_multiplier,
        'method_used': 'COMPOSITE',
        'reasoning': f"ATR: ₹{atr:.2f} x {atr_multiplier} | VIX adjustment applied"
    }



def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=period, adjust=False).mean()

def calculate_sma(prices, period):
    """Calculate Simple Moving Average"""
    return prices.rolling(window=period).mean()

def calculate_adx(high, low, close, period=14):
    """Calculate ADX correctly with proper index alignment"""
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    # ✅ FIX: Use pd.Series with SAME index as input data
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0),
        index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0),
        index=high.index
    )
    
    # Wilder's smoothing
    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    
    # ✅ FIX: Now indices align correctly
    plus_di = 100 * plus_dm.ewm(
        alpha=alpha, min_periods=period, adjust=False
    ).mean() / atr
    minus_di = 100 * minus_dm.ewm(
        alpha=alpha, min_periods=period, adjust=False
    ).mean() / atr
    
    # Avoid division by zero
    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, np.finfo(float).eps)
    
    dx = 100 * abs(plus_di - minus_di) / di_sum
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    
    return adx

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    EPSILON = np.finfo(float).eps
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + EPSILON)
    d = k.rolling(window=d_period).mean()
    
    return k, d

# ============================================================================
# VOLUME ANALYSIS
# ============================================================================

def analyze_volume(df):
    """
    Analyze volume to confirm price movements
    Returns: volume_signal, volume_ratio, description, volume_trend
    """
    if 'Volume' not in df.columns or len(df) < 20:
        return "NEUTRAL", 1.0, "Volume data not available", "NEUTRAL"
    
    if df['Volume'].iloc[-1] == 0:
        return "NEUTRAL", 1.0, "No volume data", "NEUTRAL"
    
    # Calculate average volume (20-day)
    avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
    current_volume = df['Volume'].iloc[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
    
    # Get price direction
    price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
    
    # Volume trend (is volume increasing?)
    vol_5d = df['Volume'].tail(5).mean()
    vol_20d = df['Volume'].tail(20).mean()
    volume_trend = "INCREASING" if vol_5d > vol_20d else "DECREASING"
    
    # Determine signal
    if price_change > 0 and volume_ratio > 1.5:
        signal = "STRONG_BUYING"
        desc = f"Strong buying pressure ({volume_ratio:.1f}x avg volume)"
    elif price_change > 0 and volume_ratio > 1.0:
        signal = "BUYING"
        desc = f"Buying with good volume ({volume_ratio:.1f}x)"
    elif price_change > 0 and volume_ratio < 0.7:
        signal = "WEAK_BUYING"
        desc = f"Weak rally, low volume ({volume_ratio:.1f}x)"
    elif price_change < 0 and volume_ratio > 1.5:
        signal = "STRONG_SELLING"
        desc = f"Strong selling pressure ({volume_ratio:.1f}x avg volume)"
    elif price_change < 0 and volume_ratio > 1.0:
        signal = "SELLING"
        desc = f"Selling with volume ({volume_ratio:.1f}x)"
    elif price_change < 0 and volume_ratio < 0.7:
        signal = "WEAK_SELLING"
        desc = f"Weak decline, low volume ({volume_ratio:.1f}x)"
    else:
        signal = "NEUTRAL"
        desc = f"Normal volume ({volume_ratio:.1f}x)"
    
    return signal, volume_ratio, desc, volume_trend

# ============================================================================
# SUPPORT/RESISTANCE DETECTION
# ============================================================================

def find_support_resistance(df, lookback=60):
    """
    Find key support and resistance levels using multiple methods.
    Uses pivot points, volume profile, and clustering.
    """
    if len(df) < lookback:
        lookback = len(df)
    
    if lookback < 10:
        current_price = df['Close'].iloc[-1]
        return {
            'support_levels': [],
            'resistance_levels': [],
            'nearest_support': current_price * 0.95,
            'nearest_resistance': current_price * 1.05,
            'distance_to_support': 5.0,
            'distance_to_resistance': 5.0,
            'support_strength': 'WEAK',
            'resistance_strength': 'WEAK',
            'support_touches': 0,
            'resistance_touches': 0,
            'psychological_levels': []
        }
    
    high = df['High'].tail(lookback)
    low = df['Low'].tail(lookback)
    close = df['Close'].tail(lookback)
    volume = df['Volume'].tail(lookback) if 'Volume' in df.columns else None
    current_price = float(close.iloc[-1])
    
    # METHOD 1: PIVOT POINTS
    pivot_highs = []
    pivot_lows = []
    
    for i in range(3, len(high) - 3):
        # Pivot high
        if (high.iloc[i] >= high.iloc[i-1] and high.iloc[i] >= high.iloc[i-2] and
            high.iloc[i] >= high.iloc[i-3] and high.iloc[i] >= high.iloc[i+1] and
            high.iloc[i] >= high.iloc[i+2] and high.iloc[i] >= high.iloc[i+3]):
            
            vol_weight = 1.0
            if volume is not None and volume.iloc[i] > volume.mean():
                vol_weight = 1.5
            
            pivot_highs.append({
                'price': float(high.iloc[i]),
                'index': i,
                'weight': vol_weight
            })
        
        # Pivot low
        if (low.iloc[i] <= low.iloc[i-1] and low.iloc[i] <= low.iloc[i-2] and
            low.iloc[i] <= low.iloc[i-3] and low.iloc[i] <= low.iloc[i+1] and
            low.iloc[i] <= low.iloc[i+2] and low.iloc[i] <= low.iloc[i+3]):
            
            vol_weight = 1.0
            if volume is not None and volume.iloc[i] > volume.mean():
                vol_weight = 1.5
            
            pivot_lows.append({
                'price': float(low.iloc[i]),
                'index': i,
                'weight': vol_weight
            })
    
    # METHOD 2: CLUSTER NEARBY LEVELS
    def cluster_levels(pivots, threshold_pct=1.5):
        """Cluster nearby pivot points and calculate strength."""
        if not pivots:
            return []
        
        sorted_pivots = sorted(pivots, key=lambda x: x['price'])
        clusters = []
        current_cluster = [sorted_pivots[0]]
        
        for pivot in sorted_pivots[1:]:
            cluster_center = sum(p['price'] for p in current_cluster) / len(current_cluster)
            if (pivot['price'] - cluster_center) / cluster_center * 100 < threshold_pct:
                current_cluster.append(pivot)
            else:
                avg_price = sum(p['price'] * p['weight'] for p in current_cluster) / sum(p['weight'] for p in current_cluster)
                total_weight = sum(p['weight'] for p in current_cluster)
                touch_count = len(current_cluster)
                
                clusters.append({
                    'price': avg_price,
                    'touches': touch_count,
                    'weight': total_weight,
                    'strength': 'STRONG' if touch_count >= 3 else 'MODERATE' if touch_count >= 2 else 'WEAK'
                })
                
                current_cluster = [pivot]
        
        # Last cluster
        if current_cluster:
            avg_price = sum(p['price'] * p['weight'] for p in current_cluster) / sum(p['weight'] for p in current_cluster)
            total_weight = sum(p['weight'] for p in current_cluster)
            touch_count = len(current_cluster)
            
            clusters.append({
                'price': avg_price,
                'touches': touch_count,
                'weight': total_weight,
                'strength': 'STRONG' if touch_count >= 3 else 'MODERATE' if touch_count >= 2 else 'WEAK'
            })
        
        return clusters
    
    # ✅ Dynamic threshold based on ATR percentage
    if 'Volume' in df.columns and len(df) >= 14:
        _atr = calculate_atr(
            df['High'].tail(lookback),
            df['Low'].tail(lookback),
            df['Close'].tail(lookback)
        ).iloc[-1]
        
        if not pd.isna(_atr) and current_price > 0:
            # Use ATR as percentage for dynamic threshold
            dynamic_threshold = max(
                0.5,
                min(3.0, (_atr / current_price) * 100 * 1.5)
            )
        else:
            dynamic_threshold = 1.5
    else:
        dynamic_threshold = 1.5
    
    support_clusters = cluster_levels(pivot_lows, dynamic_threshold)
    resistance_clusters = cluster_levels(pivot_highs, dynamic_threshold)
    
    # Find nearest support
    supports_below = [s for s in support_clusters if s['price'] < current_price]
    if supports_below:
        nearest_support_data = max(supports_below, key=lambda x: x['price'])
        nearest_support = nearest_support_data['price']
        support_strength = nearest_support_data['strength']
        support_touches = nearest_support_data['touches']
    else:
        nearest_support = float(low.min()) * 0.99
        support_strength = 'WEAK'
        support_touches = 0
    
    # Find nearest resistance
    resistances_above = [r for r in resistance_clusters if r['price'] > current_price]
    if resistances_above:
        nearest_resistance_data = min(resistances_above, key=lambda x: x['price'])
        nearest_resistance = nearest_resistance_data['price']
        resistance_strength = nearest_resistance_data['strength']
        resistance_touches = nearest_resistance_data['touches']
    else:
        nearest_resistance = float(high.max()) * 1.01
        resistance_strength = 'WEAK'
        resistance_touches = 0
    
    # METHOD 3: PSYCHOLOGICAL LEVELS (Round Numbers)
    def find_round_numbers(price, range_pct=5):
        levels = []
        magnitude = 10 ** (len(str(int(price))) - 2)
        base = int(price / magnitude) * magnitude
        
        for offset in range(-3, 4):
            level = base + (offset * magnitude)
            if abs(level - price) / price * 100 < range_pct:
                levels.append(level)
        
        half_magnitude = magnitude / 2
        for offset in range(-5, 6):
            level = base + (offset * half_magnitude)
            if abs(level - price) / price * 100 < range_pct:
                if level not in levels:
                    levels.append(level)
        
        return sorted(levels)
    
    psychological_levels = find_round_numbers(current_price)
    
    # Calculate distances
    distance_to_support = ((current_price - nearest_support) / current_price) * 100
    distance_to_resistance = ((nearest_resistance - current_price) / current_price) * 100
    
    return {
        'support_levels': [s['price'] for s in support_clusters[-5:]],
        'resistance_levels': [r['price'] for r in resistance_clusters[-5:]],
        'nearest_support': nearest_support,
        'nearest_resistance': nearest_resistance,
        'distance_to_support': distance_to_support,
        'distance_to_resistance': distance_to_resistance,
        'support_strength': support_strength,
        'resistance_strength': resistance_strength,
        'support_touches': support_touches,
        'resistance_touches': resistance_touches,
        'psychological_levels': psychological_levels
    }

# ============================================================================
# MOMENTUM SCORING (0-100)
# ============================================================================

def calculate_momentum_score(df):
    """
    Calculate comprehensive momentum score (0-100)
    Higher = More bullish, Lower = More bearish
    """
    close = df['Close']
    score = 50  # Start neutral
    components = {}
    
    # RSI Component (0-20 points)
    rsi = calculate_rsi(close).iloc[-1]
    if pd.isna(rsi):
        rsi = 50
    
    if rsi > 70:
        rsi_score = -10  # Overbought
    elif rsi > 60:
        rsi_score = 15
    elif rsi > 50:
        rsi_score = 10
    elif rsi > 40:
        rsi_score = -5
    elif rsi > 30:
        rsi_score = -15
    else:
        rsi_score = 10  # Oversold bounce
    
    score += rsi_score
    components['RSI'] = rsi_score
    
    # MACD Component (0-20 points)
    macd, signal, histogram = calculate_macd(close)
    hist_current = histogram.iloc[-1] if len(histogram) > 0 else 0
    hist_prev = histogram.iloc[-2] if len(histogram) > 1 else 0
    
    if pd.isna(hist_current):
        hist_current = 0
    if pd.isna(hist_prev):
        hist_prev = 0
    
    if hist_current > 0:
        if hist_current > hist_prev:
            macd_score = 20
        else:
            macd_score = 10
    else:
        if hist_current < hist_prev:
            macd_score = -20
        else:
            macd_score = -10
    
    score += macd_score
    components['MACD'] = macd_score
    
    # Moving Average Component (0-20 points)
    current_price = close.iloc[-1]
    sma_20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else close.mean()
    sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20
    ema_9 = close.ewm(span=9).mean().iloc[-1]
    
    ma_score = 0
    if current_price > ema_9:
        ma_score += 5
    if current_price > sma_20:
        ma_score += 5
    if current_price > sma_50:
        ma_score += 5
    if sma_20 > sma_50:
        ma_score += 5
    
    if current_price < ema_9:
        ma_score -= 5
    if current_price < sma_20:
        ma_score -= 5
    if current_price < sma_50:
        ma_score -= 5
    if sma_20 < sma_50:
        ma_score -= 5
    
    score += ma_score
    components['MA'] = ma_score
    
    # Price Momentum (0-15 points)
    returns_5d = ((close.iloc[-1] / close.iloc[-6]) - 1) * 100 if len(close) > 6 else 0
    momentum_score = min(15, max(-15, returns_5d * 3))
    score += momentum_score
    components['Momentum'] = momentum_score
    
 

    if sma_50 != 0:
        sma_spread_pct = safe_divide(
            abs(sma_20 - sma_50), sma_50, 0
        ) * 100
    else:
        sma_spread_pct = 0
    
    if current_price > sma_20:
        trend_score = min(10, sma_spread_pct * 2)
    else:
        trend_score = -min(10, sma_spread_pct * 2)
    
    score += trend_score
    components['Trend'] = trend_score
    
    # Cap between 0-100
    final_score = max(0, min(100, score))
    
    # Determine trend direction
    if final_score >= 70:
        trend = "STRONG BULLISH"
    elif final_score >= 55:
        trend = "BULLISH"
    elif final_score >= 45:
        trend = "NEUTRAL"
    elif final_score >= 30:
        trend = "BEARISH"
    else:
        trend = "STRONG BEARISH"
    
    return final_score, trend, components

# ============================================================================
# MULTI-TIMEFRAME ANALYSIS
# ============================================================================

def multi_timeframe_analysis(ticker, position_type):
    """Analyze multiple timeframes with rate limiting."""
    symbol = ticker if '.NS' in str(ticker) else f"{ticker}.NS"
    
    try:
        rate_limited_api_call(symbol)
        stock = yf.Ticker(symbol)
        
        timeframes = {}
        
        # Daily
        try:
            daily_df = stock.history(period="3mo", interval="1d")
            if len(daily_df) >= 20:
                timeframes['Daily'] = daily_df
        except:
            pass
        
        time.sleep(0.3)
        
        # Weekly
        try:
            weekly_df = stock.history(period="1y", interval="1wk")
            if len(weekly_df) >= 10:
                timeframes['Weekly'] = weekly_df
        except:
            pass
        
        # Hourly (only during market hours)
        is_open, _, _, _ = is_market_hours()
        if is_open:
            time.sleep(0.3)
            try:
                hourly_df = stock.history(period="5d", interval="1h")
                if len(hourly_df) >= 10:
                    timeframes['Hourly'] = hourly_df
            except:
                pass
        
        if not timeframes:
            return {
                'signals': {},
                'details': {},
                'alignment_score': 50,
                'recommendation': "Unable to fetch data",
                'aligned_count': 0,
                'against_count': 0,
                'total_timeframes': 0,
                'trend_strength': 'UNKNOWN'
            }
        
        signals = {}
        details = {}
        
        for tf_name, tf_df in timeframes.items():
            if len(tf_df) >= 14:
                close = tf_df['Close']
                current = float(close.iloc[-1])
                
                rsi = calculate_rsi(close).iloc[-1]
                if pd.isna(rsi):
                    rsi = 50
                
                sma_20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else close.mean()
                ema_9 = close.ewm(span=9).mean().iloc[-1]
                ema_21 = close.ewm(span=21).mean().iloc[-1] if len(close) >= 21 else close.mean()
                
                macd, signal_line, histogram = calculate_macd(close)
                macd_hist = histogram.iloc[-1] if len(histogram) > 0 else 0
                if pd.isna(macd_hist):
                    macd_hist = 0
                
                bullish_points = 0
                total_points = 8
                
                if rsi > 50:
                    bullish_points += 2
                if current > sma_20:
                    bullish_points += 2
                if ema_9 > ema_21:
                    bullish_points += 2
                if macd_hist > 0:
                    bullish_points += 2
                
                bullish_pct = (bullish_points / total_points) * 100
                
                if bullish_pct >= 75:
                    signal = "BULLISH"
                    strength = "Strong"
                elif bullish_pct >= 50:
                    signal = "BULLISH"
                    strength = "Moderate"
                elif bullish_pct <= 25:
                    signal = "BEARISH"
                    strength = "Strong"
                elif bullish_pct < 50:
                    signal = "BEARISH"
                    strength = "Moderate"
                else:
                    signal = "NEUTRAL"
                    strength = "Weak"
                
                signals[tf_name] = signal
                details[tf_name] = {
                    'signal': signal,
                    'strength': strength,
                    'rsi': rsi,
                    'above_sma20': current > sma_20,
                    'ema_bullish': ema_9 > ema_21,
                    'macd_bullish': macd_hist > 0,
                    'bullish_score': bullish_pct
                }
        
        # Calculate alignment
        if position_type == "LONG":
            aligned = sum(1 for s in signals.values() if s == "BULLISH")
            against = sum(1 for s in signals.values() if s == "BEARISH")
        else:
            aligned = sum(1 for s in signals.values() if s == "BEARISH")
            against = sum(1 for s in signals.values() if s == "BULLISH")
        
        total = len(signals)
        alignment_score = int((aligned / total) * 100) if total > 0 else 50
        
        if alignment_score >= 80:
            recommendation = f"✅ Strong alignment with {position_type}"
        elif alignment_score >= 60:
            recommendation = f"👍 Good alignment with {position_type}"
        elif alignment_score >= 40:
            recommendation = f"⚠️ Mixed signals"
        else:
            recommendation = f"🚨 Against {position_type}"
        
        return {
            'signals': signals,
            'details': details,
            'alignment_score': alignment_score,
            'recommendation': recommendation,
            'aligned_count': aligned,
            'against_count': against,
            'total_timeframes': total,
            'trend_strength': 'STRONG' if alignment_score >= 70 else 'MODERATE' if alignment_score >= 50 else 'WEAK'
        }
    
    except Exception as e:
        return {
            'signals': {},
            'details': {},
            'alignment_score': 50,
            'recommendation': f"Error: {str(e)}",
            'aligned_count': 0,
            'against_count': 0,
            'total_timeframes': 0,
            'trend_strength': 'UNKNOWN'
        }

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def calculate_correlation_matrix(tickers, period="3mo"):
    """Calculate correlation matrix between stocks"""
    price_data = {}
    
    for ticker in tickers:
        df = get_stock_data_safe(ticker, period=period)
        if df is not None and len(df) > 20:
            price_data[ticker] = df['Close'].pct_change().dropna()
        time.sleep(0.2)
    
    if len(price_data) < 2:
        return None, "Not enough data"
    
    # Align all series
    combined = pd.DataFrame(price_data)
    combined = combined.dropna()
    
    if len(combined) < 20:
        return None, "Insufficient overlapping data"
    
    correlation_matrix = combined.corr()
    
    return correlation_matrix, "Success"

def analyze_correlation_risk(correlation_matrix, threshold=0.7):
    """Analyze correlation risk in portfolio"""
    if correlation_matrix is None:
        return [], 0, "No correlation data"
    
    high_correlations = []
    tickers = correlation_matrix.columns.tolist()
    
    for i, ticker1 in enumerate(tickers):
        for j, ticker2 in enumerate(tickers):
            if i < j:
                corr = correlation_matrix.loc[ticker1, ticker2]
                if abs(corr) >= threshold:
                    high_correlations.append({
                        'pair': f"{ticker1} - {ticker2}",
                        'correlation': corr,
                        'risk': 'HIGH' if abs(corr) >= 0.85 else 'MEDIUM'
                    })
    
    avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
    
    if avg_correlation > 0.6:
        status = "🔴 High portfolio correlation - diversification needed"
    elif avg_correlation > 0.4:
        status = "🟡 Moderate correlation - acceptable"
    else:
        status = "🟢 Low correlation - well diversified"
    
    return high_correlations, avg_correlation, status

# ============================================================================
# NEW: MARKET-AWARE SL ADJUSTMENT
# ============================================================================

def adjust_sl_for_market_conditions(current_sl, entry_price, current_price,
                                      position_type, market_health, atr):
    """
    Automatically adjust SL based on market conditions.
    Only TIGHTENS SL, never loosens it.
    """
    adjustments = []
    new_sl = current_sl
    
    if not market_health:
        return current_sl, adjustments
    
    vix = market_health.get('vix', 15)
    market_status = market_health.get('status', 'NEUTRAL')
    nifty_rsi = market_health.get('nifty_rsi', 50)
    
    # RULE 1: VIX Spike → Tighten SL by 20%
    if vix > 25:
        if position_type == 'LONG':
            distance = current_price - current_sl
            new_distance = distance * 0.80
            proposed_sl = current_price - new_distance
            
            if proposed_sl > current_sl:
                new_sl = proposed_sl
                adjustments.append(f"⚡ VIX {vix:.1f} → SL tightened 20%")
        else:
            distance = current_sl - current_price
            new_distance = distance * 0.80
            proposed_sl = current_price + new_distance
            
            if proposed_sl < current_sl:
                new_sl = proposed_sl
                adjustments.append(f"⚡ VIX {vix:.1f} → SL tightened 20%")
    
    # RULE 2: Market Bearish + LONG → Tighten by 15%
    if market_status == 'BEARISH' and position_type == 'LONG':
        distance = current_price - new_sl
        proposed_sl = current_price - (distance * 0.85)
        
        if proposed_sl > new_sl:
            new_sl = proposed_sl
            adjustments.append("🐻 Bearish market → LONG SL tightened 15%")
    
    elif market_status == 'BULLISH' and position_type == 'SHORT':
        distance = new_sl - current_price
        proposed_sl = current_price + (distance * 0.85)
        
        if proposed_sl < new_sl:
            new_sl = proposed_sl
            adjustments.append("🐂 Bullish market → SHORT SL tightened 15%")
    
    # RULE 3: NIFTY RSI extreme → Protect profits
    pnl_pct = ((current_price - entry_price) / entry_price * 100
               if position_type == 'LONG'
               else (entry_price - current_price) / entry_price * 100)
    
    if pnl_pct > 3:
        if nifty_rsi > 75 and position_type == 'LONG':
            proposed_sl = entry_price + (current_price - entry_price) * 0.50
            if proposed_sl > new_sl:
                new_sl = proposed_sl
                adjustments.append(f"📈 NIFTY RSI {nifty_rsi:.0f} → Protecting 50% of {pnl_pct:.1f}% profit")
        
        elif nifty_rsi < 25 and position_type == 'SHORT':
            proposed_sl = entry_price - (entry_price - current_price) * 0.50
            if proposed_sl < new_sl:
                new_sl = proposed_sl
                adjustments.append(f"📉 NIFTY RSI {nifty_rsi:.0f} → Protecting 50% of {pnl_pct:.1f}% profit")
    
    new_sl = round_to_tick_size(new_sl)
    
    return new_sl, adjustments

# ============================================================================
# STOP LOSS RISK PREDICTION (0-100)
# ============================================================================

def predict_sl_risk(df, current_price, stop_loss, position_type, entry_price, sl_alert_threshold=50):
    """
    Predict likelihood of hitting stop loss
    Returns: risk_score (0-100), reasons, recommendation, priority
    """
    risk_score = 0
    reasons = []
    close = df['Close']
    
    # Distance to Stop Loss (0-40 points)
    if position_type == "LONG":
        distance_pct = ((current_price - stop_loss) / current_price) * 100
    else:
        distance_pct = ((stop_loss - current_price) / current_price) * 100
    
    if distance_pct < 0:  # Already hit SL
        risk_score = 100
        reasons.append("⚠️ SL already breached!")
    elif distance_pct < 1:
        risk_score += 40
        reasons.append(f"🔴 Very close to SL ({distance_pct:.1f}% away)")
    elif distance_pct < 2:
        risk_score += 30
        reasons.append(f"🟠 Close to SL ({distance_pct:.1f}% away)")
    elif distance_pct < 3:
        risk_score += 15
        reasons.append(f"🟡 Approaching SL ({distance_pct:.1f}% away)")
    elif distance_pct < 5:
        risk_score += 5
    
    # Trend Against Position (0-25 points)
    sma_20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else close.mean()
    sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20
    ema_9 = close.ewm(span=9).mean().iloc[-1]
    
    if position_type == "LONG":
        if current_price < ema_9:
            risk_score += 8
            reasons.append("📉 Below EMA 9")
        if current_price < sma_20:
            risk_score += 10
            reasons.append("📉 Below SMA 20")
        if current_price < sma_50:
            risk_score += 7
            reasons.append("📉 Below SMA 50")
        if sma_20 < sma_50:
            risk_score += 5
            reasons.append("📉 Death cross forming")
    else:  # SHORT
        if current_price > ema_9:
            risk_score += 8
            reasons.append("📈 Above EMA 9")
        if current_price > sma_20:
            risk_score += 10
            reasons.append("📈 Above SMA 20")
        if current_price > sma_50:
            risk_score += 7
            reasons.append("📈 Above SMA 50")
        if sma_20 > sma_50:
            risk_score += 5
            reasons.append("📈 Golden cross forming")
    
    # MACD Against Position (0-15 points)
    macd, signal, histogram = calculate_macd(close)
    hist_current = histogram.iloc[-1] if len(histogram) > 0 else 0
    hist_prev = histogram.iloc[-2] if len(histogram) > 1 else 0
    
    if pd.isna(hist_current):
        hist_current = 0
    if pd.isna(hist_prev):
        hist_prev = 0
    
    if position_type == "LONG":
        if hist_current < 0:
            risk_score += 8
            reasons.append("📊 MACD bearish")
        if hist_current < hist_prev:
            risk_score += 7
            reasons.append("📊 MACD declining")
    else:
        if hist_current > 0:
            risk_score += 8
            reasons.append("📊 MACD bullish")
        if hist_current > hist_prev:
            risk_score += 7
            reasons.append("📊 MACD rising")
    
    # RSI Extreme (0-10 points)
    rsi = calculate_rsi(close).iloc[-1]
    if pd.isna(rsi):
        rsi = 50
    
    if position_type == "LONG" and rsi < 35:
        risk_score += 10
        reasons.append(f"📉 RSI weak ({rsi:.0f})")
    elif position_type == "SHORT" and rsi > 65:
        risk_score += 10
        reasons.append(f"📈 RSI strong ({rsi:.0f})")
    
    # Consecutive Candles Against Position (0-10 points)
    # Consecutive Candles Against Position (0-10 points)
    # ✅ SWING FIX: Reduced weight - 3 red candles in a pullback is normal
    if len(close) >= 4:
        last_3 = close.tail(4).diff().dropna()
        if position_type == "LONG" and all(last_3 < 0):
            # Check if it's a DEEP pullback or just normal
            total_drop = abs(
                (close.iloc[-1] - close.iloc[-4]) / close.iloc[-4]
            ) * 100
            
            if total_drop > 5:
                risk_score += 10
                reasons.append(
                    f"🕯️ 3 red candles with {total_drop:.1f}% drop - serious"
                )
            elif total_drop > 2:
                risk_score += 5
                reasons.append(
                    f"🕯️ 3 red candles ({total_drop:.1f}% drop) - moderate"
                )
            else:
                risk_score += 2
                reasons.append(
                    f"🕯️ 3 red candles ({total_drop:.1f}% drop) "
                    f"- normal pullback"
                )
        
        elif position_type == "SHORT" and all(last_3 > 0):
            total_rise = abs(
                (close.iloc[-1] - close.iloc[-4]) / close.iloc[-4]
            ) * 100
            
            if total_rise > 5:
                risk_score += 10
                reasons.append(
                    f"🕯️ 3 green candles with {total_rise:.1f}% rise - serious"
                )
            elif total_rise > 2:
                risk_score += 5
                reasons.append(
                    f"🕯️ 3 green candles ({total_rise:.1f}% rise) - moderate"
                )
            else:
                risk_score += 2
                reasons.append(
                    f"🕯️ 3 green candles ({total_rise:.1f}% rise) "
                    f"- normal bounce"
                )
    
    # Volume Confirmation (0-10 points)
    volume_signal, volume_ratio, _, _ = analyze_volume(df)
    
    if position_type == "LONG" and volume_signal in ["STRONG_SELLING", "SELLING"]:
        risk_score += 10
        reasons.append(f"📊 Selling volume ({volume_ratio:.1f}x)")
    elif position_type == "SHORT" and volume_signal in ["STRONG_BUYING", "BUYING"]:
        risk_score += 10
        reasons.append(f"📊 Buying volume ({volume_ratio:.1f}x)")
    
    # Cap at 100
    risk_score = min(100, risk_score)
    
    # Generate recommendation based on threshold
    if risk_score >= 80:
        recommendation = "🚨 EXIT NOW - Very high risk"
        priority = "CRITICAL"
    elif risk_score >= sl_alert_threshold + 20:
        recommendation = "⚠️ CONSIDER EXIT - High risk"
        priority = "HIGH"
    elif risk_score >= sl_alert_threshold:
        recommendation = "👀 WATCH CLOSELY - Moderate risk"
        priority = "MEDIUM"
    elif risk_score >= 20:
        recommendation = "✅ MONITOR - Low risk"
        priority = "LOW"
    else:
        recommendation = "✅ SAFE - Very low risk"
        priority = "SAFE"
    
    return risk_score, reasons, recommendation, priority

# ============================================================================
# UPSIDE POTENTIAL PREDICTION
# ============================================================================

def predict_upside_potential(df, current_price, target1, target2, position_type):
    """
    Predict if stock can continue after hitting target
    Returns: upside_score (0-100), new_target, reasons, recommendation, action
    """
    score = 50  # Start neutral
    reasons = []
    close = df['Close']
    
    # Momentum still strong?
    momentum_score, trend, _ = calculate_momentum_score(df)
    
    if position_type == "LONG":
        if momentum_score >= 70:
            score += 25
            reasons.append(f"🚀 Strong momentum ({momentum_score:.0f})")
        elif momentum_score >= 55:
            score += 15
            reasons.append(f"📈 Good momentum ({momentum_score:.0f})")
        elif momentum_score <= 40:
            score -= 20
            reasons.append(f"📉 Weak momentum ({momentum_score:.0f})")
    else:  # SHORT
        if momentum_score <= 30:
            score += 25
            reasons.append(f"🚀 Strong bearish momentum ({momentum_score:.0f})")
        elif momentum_score <= 45:
            score += 15
            reasons.append(f"📉 Good bearish momentum ({momentum_score:.0f})")
        elif momentum_score >= 60:
            score -= 20
            reasons.append(f"📈 Bullish reversal ({momentum_score:.0f})")
    
    # RSI not extreme?
    rsi = calculate_rsi(close).iloc[-1]
    if pd.isna(rsi):
        rsi = 50
    
    if position_type == "LONG":
        if rsi < 60:
            score += 15
            reasons.append(f"✅ RSI has room ({rsi:.0f})")
        elif rsi > 75:
            score -= 25
            reasons.append(f"⚠️ RSI overbought ({rsi:.0f})")
        elif rsi > 65:
            score -= 10
            reasons.append(f"🟡 RSI getting high ({rsi:.0f})")
    else:
        if rsi > 40:
            score += 15
            reasons.append(f"✅ RSI has room ({rsi:.0f})")
        elif rsi < 25:
            score -= 25
            reasons.append(f"⚠️ RSI oversold ({rsi:.0f})")
    
    # Volume confirming?
    volume_signal, volume_ratio, _, volume_trend = analyze_volume(df)
    
    if position_type == "LONG" and volume_signal in ["STRONG_BUYING", "BUYING"]:
        score += 15
        reasons.append(f"📊 Buying volume ({volume_ratio:.1f}x)")
    elif position_type == "SHORT" and volume_signal in ["STRONG_SELLING", "SELLING"]:
        score += 15
        reasons.append(f"📊 Selling volume ({volume_ratio:.1f}x)")
    elif volume_ratio < 0.7:
        score -= 10
        reasons.append("📊 Low volume")
    
    # Bollinger Band position
    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(close)
    if len(upper_bb) > 0 and len(lower_bb) > 0:
        bb_upper = upper_bb.iloc[-1]
        bb_lower = lower_bb.iloc[-1]
        bb_range = bb_upper - bb_lower
        
        if bb_range > 0:
            if position_type == "LONG":
                bb_position = (current_price - bb_lower) / bb_range
                if bb_position < 0.7:
                    score += 10
                    reasons.append("📈 Room to upper BB")
                elif bb_position > 0.95:
                    score -= 15
                    reasons.append("⚠️ At upper BB")
            else:
                bb_position = (current_price - bb_lower) / bb_range
                if bb_position > 0.3:
                    score += 10
                    reasons.append("📉 Room to lower BB")
                elif bb_position < 0.05:
                    score -= 15
                    reasons.append("⚠️ At lower BB")
    
    # Calculate new target based on ATR and S/R
    atr = calculate_atr(df['High'], df['Low'], close).iloc[-1]
    if pd.isna(atr):
        atr = current_price * 0.02
    
    sr_levels = find_support_resistance(df)
    
    if position_type == "LONG":
        atr_target = current_price + (atr * 3)
        sr_target = sr_levels['nearest_resistance']
        new_target = min(atr_target, sr_target) if sr_target > current_price else atr_target
        potential_gain = ((new_target - current_price) / current_price) * 100
    else:
        atr_target = current_price - (atr * 3)
        sr_target = sr_levels['nearest_support']
        new_target = max(atr_target, sr_target) if sr_target < current_price else atr_target
        potential_gain = ((current_price - new_target) / current_price) * 100
    
    # ✅ Round new_target to NSE tick size
    new_target = round_to_tick_size(new_target)
    
    if potential_gain > 5:
        score += 10
        reasons.append(f"🎯 {potential_gain:.1f}% more potential")
    
    # Cap score
    score = max(0, min(100, score))
    
    # Generate recommendation
    if score >= 70:
        recommendation = "HOLD"
        action = f"Strong upside - New target: ₹{new_target:.2f}"
    elif score >= 50:
        recommendation = "PARTIAL_EXIT"
        action = f"Book 50%, hold rest for ₹{new_target:.2f}"
    else:
        recommendation = "EXIT"
        action = "Book full profits now"
    
    return score, new_target, reasons, recommendation, action

# ============================================================================
# NEW: SMART EXIT TIMING ENGINE
# ============================================================================

def smart_exit_decision(result, market_health, df):
    """
    Makes intelligent exit decisions beyond simple SL/Target.
    Returns action, confidence, reasons, and suggested exit price.
    """
    score = 50  # Start neutral (50 = HOLD)
    reasons = []
    
    current_price = result['current_price']
    entry_price = result['entry_price']
    position_type = result['position_type']
    pnl_pct = result['pnl_percent']
    
    # FACTOR 1: MOMENTUM EXHAUSTION (0-25 points)
    rsi = result['rsi']
    macd_hist = result['macd_hist']
    momentum = result['momentum_score']
    
    if position_type == 'LONG':
        if rsi > 75 and macd_hist < 0:
            score -= 25
            reasons.append("🔴 RSI overbought + MACD turning negative")
        elif rsi > 70:
            score -= 15
            reasons.append("🟠 RSI overbought territory")
        
        if momentum < 35:
            score -= 15
            reasons.append(f"📉 Weak momentum ({momentum:.0f}/100)")
        elif momentum > 70:
            score += 10
            reasons.append(f"📈 Strong momentum ({momentum:.0f}/100)")
    else:
        if rsi < 25 and macd_hist > 0:
            score -= 25
            reasons.append("🔴 RSI oversold + MACD turning positive")
        elif rsi < 30:
            score -= 15
            reasons.append("🟠 RSI oversold territory")
        
        if momentum > 65:
            score -= 15
            reasons.append(f"📈 Strong bullish momentum against SHORT ({momentum:.0f}/100)")
    
    # FACTOR 2: VOLUME CONFIRMATION (0-15 points)
    vol_signal = result['volume_signal']
    vol_ratio = result['volume_ratio']
    
    if position_type == 'LONG':
        if vol_signal in ['STRONG_SELLING', 'SELLING'] and vol_ratio > 2.0:
            score -= 15
            reasons.append(f"📊 Heavy selling volume ({vol_ratio:.1f}x)")
        elif vol_signal == 'WEAK_BUYING':
            score -= 5
            reasons.append("📊 Weak buying - rally losing steam")
    else:
        if vol_signal in ['STRONG_BUYING', 'BUYING'] and vol_ratio > 2.0:
            score -= 15
            reasons.append(f"📊 Heavy buying volume ({vol_ratio:.1f}x)")
        elif vol_signal == 'WEAK_SELLING':
            score -= 5
            reasons.append("📊 Weak selling - decline losing steam")
    
    # FACTOR 3: MARKET CONTEXT (0-15 points)
    if market_health:
        if market_health['status'] == 'BEARISH' and position_type == 'LONG':
            score -= 15
            reasons.append("🌧️ Bearish market - against LONG position")
        elif market_health['status'] == 'BULLISH' and position_type == 'SHORT':
            score -= 15
            reasons.append("☀️ Bullish market - against SHORT position")
        
        if market_health.get('vix', 15) > 25:
            score -= 10
            reasons.append(f"⚡ High VIX ({market_health['vix']:.1f}) - increased risk")
    
    # FACTOR 4: PRICE ACTION PATTERNS (0-15 points)
    close = df['Close']
    
    if len(df) >= 4:
        last_3_changes = close.tail(4).pct_change().dropna()
        
        if position_type == 'LONG':
            if all(c < 0 for c in last_3_changes):
                score -= 15
                reasons.append("🕯️ 3 consecutive red candles")
        else:
            if all(c > 0 for c in last_3_changes):
                score -= 15
                reasons.append("🕯️ 3 consecutive green candles")
    
    # Check for doji at key levels
    if len(df) >= 1:
        last_candle = df.iloc[-1]
        body = abs(last_candle['Close'] - last_candle['Open'])
        range_candle = last_candle['High'] - last_candle['Low']
        if range_candle > 0 and body / range_candle < 0.1:
            if position_type == 'LONG' and current_price > result['resistance'] * 0.98:
                score -= 10
                reasons.append("🕯️ Doji at resistance - indecision")
            elif position_type == 'SHORT' and current_price < result['support'] * 1.02:
                score -= 10
                reasons.append("🕯️ Doji at support - indecision")
    
    # FACTOR 5: SUPPORT/RESISTANCE PROXIMITY (0-10 points)
    if position_type == 'LONG':
        if result['distance_to_resistance'] < 1.0:
            score -= 10
            reasons.append(f"🧱 Near resistance ({result['distance_to_resistance']:.1f}% away)")
    else:
        if result['distance_to_support'] < 1.0:
            score -= 10
            reasons.append(f"🧱 Near support ({result['distance_to_support']:.1f}% away)")
    
    # FACTOR 6: PROFIT PROTECTION (0-15 points)
    if pnl_pct > 8:
        if momentum < 50 or (position_type == 'LONG' and rsi > 65):
            score -= 15
            reasons.append(f"💰 Protect {pnl_pct:.1f}% profit - momentum fading")
    elif pnl_pct > 5:
        if momentum < 40:
            score -= 10
            reasons.append(f"💰 Protect {pnl_pct:.1f}% profit - weak momentum")
    elif pnl_pct < -3:
        if result['sl_risk'] > 60:
            score -= 15
            reasons.append(f"🚨 Losing {pnl_pct:.1f}% + High SL risk")
    
    # FACTOR 7: TIME-BASED
    holding_days = result.get('holding_days', 0)
    if holding_days > 30 and pnl_pct < 2:
        score -= 5
        reasons.append(f"⏰ Held {holding_days} days with only {pnl_pct:.1f}% gain")
    elif holding_days > 60 and pnl_pct < 5:
        score -= 10
        reasons.append(f"⏰ Held {holding_days} days - capital locked")
    
    # FACTOR 8: MTF ALIGNMENT
    mtf_alignment = result.get('mtf_alignment', 50)
    if mtf_alignment < 25:
        score -= 15
        reasons.append(f"📊 All timeframes against ({mtf_alignment}%)")
    elif mtf_alignment > 75:
        score += 10
        reasons.append(f"📊 Strong MTF alignment ({mtf_alignment}%)")
    
    # GENERATE DECISION
    score = max(0, min(100, score))
    
    if score <= 15:
        action = "EXIT_NOW"
        confidence = 100 - score
        exit_price = current_price
        summary = "🚨 EXIT IMMEDIATELY - Multiple factors against position"
    elif score <= 25:
        action = "EXIT_TODAY"
        confidence = 90 - score
        exit_price = current_price
        summary = "⚠️ EXIT TODAY - Deteriorating conditions"
    elif score <= 35:
        action = "PARTIAL_EXIT"
        confidence = 80 - score
        exit_price = current_price
        summary = "📊 BOOK 50% PROFITS - Mixed signals"
    elif score <= 45:
        action = "TIGHTEN_SL"
        confidence = 60
        exit_price = None
        summary = "🔧 TIGHTEN STOP LOSS - Caution warranted"
    elif score <= 60:
        action = "HOLD"
        confidence = 50
        exit_price = None
        summary = "✅ HOLD - Conditions acceptable"
    elif score <= 75:
        action = "HOLD_STRONG"
        confidence = 70
        exit_price = None
        summary = "💪 HOLD STRONG - Good conditions"
    else:
        action = "ADD_MORE"
        confidence = score
        exit_price = None
        summary = "🚀 CONSIDER ADDING - Very strong conditions"
    
    return {
        'action': action,
        'score': score,
        'confidence': confidence,
        'exit_price': exit_price,
        'summary': summary,
        'reasons': reasons,
        'factors_count': len(reasons)
    }

# ============================================================================
# DYNAMIC TARGET & TRAIL STOP CALCULATION
# ============================================================================

def calculate_dynamic_levels(df, entry_price, current_price, stop_loss, position_type,
                            pnl_percent, trail_trigger=2.0):
    """
    Calculate dynamic targets and trailing stop loss.
    Uses ATR-based dynamic trailing instead of fixed percentages.
    """
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Calculate ATR
    atr = calculate_atr(high, low, close).iloc[-1]
    if pd.isna(atr) or atr <= 0:
        atr = current_price * 0.02
    
    atr_pct = (atr / current_price) * 100
    
    # Get support/resistance
    sr_levels = find_support_resistance(df)
    
    result = {
        'atr': atr,
        'atr_pct': atr_pct,
        'support': sr_levels['nearest_support'],
        'resistance': sr_levels['nearest_resistance'],
        'support_strength': sr_levels.get('support_strength', 'UNKNOWN'),
        'resistance_strength': sr_levels.get('resistance_strength', 'UNKNOWN')
    }
    
    # DYNAMIC TRAIL STOP CALCULATION
    if position_type == "LONG":
        # Calculate dynamic targets
        result['target1'] = current_price + (atr * 1.5)
        result['target2'] = current_price + (atr * 3)
        result['target3'] = min(current_price + (atr * 5), sr_levels['nearest_resistance'])
        # ✅ Round targets to NSE tick size AFTER all are defined
        result["target1"] = round_to_tick_size(result["target1"])
        result["target2"] = round_to_tick_size(result["target2"])
        result["target3"] = round_to_tick_size(result["target3"])
        
        # Dynamic trail based on profit level AND volatility (ATR)
        if pnl_percent >= trail_trigger * 5:  # e.g., 10% profit
            atr_trail = current_price - (atr * 1.0)
            pct_trail = entry_price + (current_price - entry_price) * 0.70
            result['trail_stop'] = max(atr_trail, pct_trail)
            result['trail_reason'] = f"Locking 70%+ profit (P&L: {pnl_percent:.1f}%)"
            result['trail_action'] = "LOCK_MAJOR_PROFIT"
        
        elif pnl_percent >= trail_trigger * 4:  # e.g., 8%
            atr_trail = current_price - (atr * 1.2)
            pct_trail = entry_price + (current_price - entry_price) * 0.60
            result['trail_stop'] = max(atr_trail, pct_trail)
            result['trail_reason'] = f"Locking 60% profit (P&L: {pnl_percent:.1f}%)"
            result['trail_action'] = "LOCK_PROFITS"
        
        elif pnl_percent >= trail_trigger * 3:  # e.g., 6%
            atr_trail = current_price - (atr * 1.5)
            pct_trail = entry_price + (current_price - entry_price) * 0.50
            result['trail_stop'] = max(atr_trail, pct_trail)
            result['trail_reason'] = f"Locking 50% profit (P&L: {pnl_percent:.1f}%)"
            result['trail_action'] = "SECURE_GAINS"
        
        elif pnl_percent >= trail_trigger * 2:  # e.g., 4%
            atr_trail = current_price - (atr * 2.0)
            pct_trail = entry_price + (current_price - entry_price) * 0.30
            result['trail_stop'] = max(atr_trail, pct_trail, entry_price * 1.005)
            result['trail_reason'] = f"Securing gains (P&L: {pnl_percent:.1f}%)"
            result['trail_action'] = "SECURE_GAINS"
        
        elif pnl_percent >= trail_trigger:  # e.g., 2%
            atr_trail = current_price - (atr * 2.5)
            result['trail_stop'] = max(atr_trail, entry_price)
            result['trail_reason'] = f"Moving to breakeven (P&L: {pnl_percent:.1f}%)"
            result['trail_action'] = "BREAKEVEN"
        
        elif pnl_percent >= trail_trigger * 0.5:  # e.g., 1%
            atr_trail = current_price - (atr * 3.0)
            result['trail_stop'] = max(atr_trail, stop_loss)
            if result['trail_stop'] > stop_loss:
                result['trail_reason'] = f"Tightening SL (P&L: {pnl_percent:.1f}%)"
                result['trail_action'] = "TIGHTEN"
            else:
                result['trail_reason'] = "Keep original SL"
                result['trail_action'] = "HOLD"
        else:
            result['trail_stop'] = stop_loss
            result['trail_reason'] = "Keep original SL - profit not enough to trail"
            result['trail_action'] = "HOLD"
        
        # Ensure trail stop is not below original SL
        result['trail_stop'] = max(result['trail_stop'], stop_loss)
        result['should_trail'] = result['trail_stop'] > stop_loss
        result['trail_improvement'] = result['trail_stop'] - stop_loss if result['should_trail'] else 0
        result['trail_improvement_pct'] = (result['trail_improvement'] / entry_price * 100) if result['should_trail'] else 0
        
        # ✅ Round trail_stop to NSE tick size
        result["trail_stop"] = round_to_tick_size(result["trail_stop"])
    
    else:  # SHORT position
        result['target1'] = current_price - (atr * 1.5)
        result['target2'] = current_price - (atr * 3)
        result['target3'] = max(current_price - (atr * 5), sr_levels['nearest_support'])
        # ✅ Round targets to NSE tick size
        result["target1"] = round_to_tick_size(result["target1"])
        result["target2"] = round_to_tick_size(result["target2"])
        result["target3"] = round_to_tick_size(result["target3"])
        
        if pnl_percent >= trail_trigger * 5:
            atr_trail = current_price + (atr * 1.0)
            pct_trail = entry_price - (entry_price - current_price) * 0.70
            result['trail_stop'] = min(atr_trail, pct_trail)
            result['trail_reason'] = f"Locking 70%+ profit (P&L: {pnl_percent:.1f}%)"
            result['trail_action'] = "LOCK_MAJOR_PROFIT"
        
        elif pnl_percent >= trail_trigger * 4:
            atr_trail = current_price + (atr * 1.2)
            pct_trail = entry_price - (entry_price - current_price) * 0.60
            result['trail_stop'] = min(atr_trail, pct_trail)
            result['trail_reason'] = f"Locking 60% profit (P&L: {pnl_percent:.1f}%)"
            result['trail_action'] = "LOCK_PROFITS"
        
        elif pnl_percent >= trail_trigger * 3:
            atr_trail = current_price + (atr * 1.5)
            pct_trail = entry_price - (entry_price - current_price) * 0.50
            result['trail_stop'] = min(atr_trail, pct_trail)
            result['trail_reason'] = f"Locking 50% profit (P&L: {pnl_percent:.1f}%)"
            result['trail_action'] = "SECURE_GAINS"
        
        elif pnl_percent >= trail_trigger * 2:
            atr_trail = current_price + (atr * 2.0)
            pct_trail = entry_price - (entry_price - current_price) * 0.30
            result['trail_stop'] = min(atr_trail, pct_trail, entry_price * 0.995)
            result['trail_reason'] = f"Securing gains (P&L: {pnl_percent:.1f}%)"
            result['trail_action'] = "SECURE_GAINS"
        
        elif pnl_percent >= trail_trigger:
            atr_trail = current_price + (atr * 2.5)
            result['trail_stop'] = min(atr_trail, entry_price)
            result['trail_reason'] = f"Moving to breakeven (P&L: {pnl_percent:.1f}%)"
            result['trail_action'] = "BREAKEVEN"
        
        elif pnl_percent >= trail_trigger * 0.5:
            atr_trail = current_price + (atr * 3.0)
            result['trail_stop'] = min(atr_trail, stop_loss)
            if result['trail_stop'] < stop_loss:
                result['trail_reason'] = f"Tightening SL (P&L: {pnl_percent:.1f}%)"
                result['trail_action'] = "TIGHTEN"
            else:
                result['trail_reason'] = "Keep original SL"
                result['trail_action'] = "HOLD"
        else:
            result['trail_stop'] = stop_loss
            result['trail_reason'] = "Keep original SL - profit not enough to trail"
            result['trail_action'] = "HOLD"
        
        result['trail_stop'] = min(result['trail_stop'], stop_loss)
        result['should_trail'] = result['trail_stop'] < stop_loss
        result['trail_improvement'] = stop_loss - result['trail_stop'] if result['should_trail'] else 0
        result['trail_improvement_pct'] = (result['trail_improvement'] / entry_price * 100) if result['should_trail'] else 0
        
        # ✅ Round trail_stop to NSE tick size
        result["trail_stop"] = round_to_tick_size(result["trail_stop"])
    
    return result

# ============================================================================
# NEW: DYNAMIC SMART TARGET CALCULATION
# ============================================================================

def calculate_smart_target(df, entry_price, current_price, position_type,
                           original_target, market_health):
    """
    Dynamically adjust targets based on momentum, volume,
    market conditions, and Fibonacci extensions.
    """
    close = df['Close']
    high = df['High']
    low = df['Low']
    atr = calculate_atr(high, low, close).iloc[-1]
    
    if pd.isna(atr) or atr <= 0:
        atr = current_price * 0.02
    
    # Get momentum
    momentum_score, trend, _ = calculate_momentum_score(df)
    rsi = calculate_rsi(close).iloc[-1]
    if pd.isna(rsi):
        rsi = 50
    
    # Fibonacci Extension levels
    recent_low = float(low.tail(20).min())
    recent_high = float(high.tail(20).max())
    fib_range = recent_high - recent_low
    
    if fib_range <= 0:
        fib_range = atr * 5  # Fallback
    
    if position_type == 'LONG':
        fib_targets = {
            '1.000': recent_high,
            '1.272': recent_low + fib_range * 1.272,
            '1.618': recent_low + fib_range * 1.618,
            '2.000': recent_low + fib_range * 2.0,
            '2.618': recent_low + fib_range * 2.618
        }
        
        atr_t1 = current_price + atr * 2
        atr_t2 = current_price + atr * 3.5
        atr_t3 = current_price + atr * 5
    else:
        fib_targets = {
            '1.000': recent_low,
            '1.272': recent_high - fib_range * 1.272,
            '1.618': recent_high - fib_range * 1.618,
            '2.000': recent_high - fib_range * 2.0,
        }
        
        atr_t1 = current_price - atr * 2
        atr_t2 = current_price - atr * 3.5
        atr_t3 = current_price - atr * 5
    
    # Adjust based on momentum
    if momentum_score > 70:
        multiplier = 1.2
        target_note = "Extended (strong momentum)"
    elif momentum_score > 55:
        multiplier = 1.0
        target_note = "Standard targets"
    elif momentum_score > 40:
        multiplier = 0.8
        target_note = "Reduced (weak momentum)"
    else:
        multiplier = 0.6
        target_note = "Conservative (very weak momentum)"
    
    # Market adjustment
    if market_health:
        if market_health['status'] == 'BULLISH' and position_type == 'LONG':
            multiplier *= 1.1
            target_note += " + Bull market boost"
        elif market_health['status'] == 'BEARISH' and position_type == 'LONG':
            multiplier *= 0.85
            target_note += " + Bear market discount"
        elif market_health['status'] == 'BEARISH' and position_type == 'SHORT':
            multiplier *= 1.1
            target_note += " + Bear market boost"
        elif market_health['status'] == 'BULLISH' and position_type == 'SHORT':
            multiplier *= 0.85
            target_note += " + Bull market discount"
    
    # Get S/R levels
    sr = find_support_resistance(df)
    
    # Calculate final smart targets
    if position_type == 'LONG':
        smart_t1 = min(
            atr_t1 * multiplier,
            sr['nearest_resistance'],
            original_target
        )
        if smart_t1 <= current_price:
            smart_t1 = current_price + atr * 1.5 * multiplier
        
        smart_t2 = atr_t2 * multiplier
        smart_t3 = atr_t3 * multiplier
    else:
        smart_t1 = max(
            atr_t1 * (2 - multiplier),
            sr['nearest_support'],
            original_target
        )
        if smart_t1 >= current_price:
            smart_t1 = current_price - atr * 1.5 * multiplier
        
        smart_t2 = atr_t2 * (2 - multiplier)
        smart_t3 = atr_t3 * (2 - multiplier)
    
    return {
        'smart_target1': round_to_tick_size(smart_t1),
        'smart_target2': round_to_tick_size(smart_t2),
        'smart_target3': round_to_tick_size(smart_t3),
        'fib_targets': {k: round_to_tick_size(v) for k, v in fib_targets.items()},
        'momentum_multiplier': multiplier,
        'target_note': target_note,
        'confidence': min(100, momentum_score +
                         (10 if market_health and
                          market_health['status'] in ['BULLISH', 'BEARISH'] else 0))
    }


# ============================================================================
# SECTOR EXPOSURE ANALYSIS
# ============================================================================

# Stock to Sector Mapping (NSE Top 100+)
SECTOR_MAP = {
    # IT
    'TCS': 'IT', 'INFY': 'IT', 'WIPRO': 'IT', 'HCLTECH': 'IT', 'TECHM': 'IT',
    'LTIM': 'IT', 'MPHASIS': 'IT', 'COFORGE': 'IT', 'PERSISTENT': 'IT', 'LTTS': 'IT',
    
    # Banking
    'HDFCBANK': 'Banking', 'ICICIBANK': 'Banking', 'SBIN': 'Banking', 'KOTAKBANK': 'Banking',
    'AXISBANK': 'Banking', 'INDUSINDBK': 'Banking', 'BANDHANBNK': 'Banking', 'FEDERALBNK': 'Banking',
    'IDFCFIRSTB': 'Banking', 'PNB': 'Banking', 'BANKBARODA': 'Banking', 'CANBK': 'Banking',
    
    # NBFC/Finance
    'HDFC': 'Finance', 'BAJFINANCE': 'Finance', 'BAJAJFINSV': 'Finance', 'SBICARD': 'Finance',
    'CHOLAFIN': 'Finance', 'M&MFIN': 'Finance', 'MUTHOOTFIN': 'Finance', 'LICHSGFIN': 'Finance',
    
    # Energy/Oil & Gas
    'RELIANCE': 'Energy', 'ONGC': 'Energy', 'IOC': 'Energy', 'BPCL': 'Energy',
    'GAIL': 'Energy', 'PETRONET': 'Energy', 'HINDPETRO': 'Energy', 'ADANIGREEN': 'Energy',
    'ADANIPOWER': 'Energy', 'TATAPOWER': 'Energy', 'POWERGRID': 'Energy', 'NTPC': 'Energy',
    
    # Auto
    'MARUTI': 'Auto', 'TATAMOTORS': 'Auto', 'M&M': 'Auto', 'BAJAJ-AUTO': 'Auto',
    'HEROMOTOCO': 'Auto', 'EICHERMOT': 'Auto', 'ASHOKLEY': 'Auto', 'TVSMOTOR': 'Auto',
    'MOTHERSON': 'Auto', 'BHARATFORG': 'Auto', 'BALKRISIND': 'Auto', 'MRF': 'Auto',
    
    # FMCG
    'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'NESTLEIND': 'FMCG', 'BRITANNIA': 'FMCG',
    'DABUR': 'FMCG', 'MARICO': 'FMCG', 'GODREJCP': 'FMCG', 'COLPAL': 'FMCG',
    'TATACONSUM': 'FMCG', 'VBL': 'FMCG', 'MCDOWELL-N': 'FMCG', 'UBL': 'FMCG',
    
    # Pharma
    'SUNPHARMA': 'Pharma', 'DRREDDY': 'Pharma', 'CIPLA': 'Pharma', 'DIVISLAB': 'Pharma',
    'APOLLOHOSP': 'Pharma', 'LUPIN': 'Pharma', 'AUROPHARMA': 'Pharma', 'BIOCON': 'Pharma',
    'TORNTPHARM': 'Pharma', 'ALKEM': 'Pharma', 'GLENMARK': 'Pharma', 'LAURUSLABS': 'Pharma',
    
    # Telecom
    'BHARTIARTL': 'Telecom', 'IDEA': 'Telecom', 'INDUSTOWER': 'Telecom',
    
    # Infrastructure/Construction
    'LT': 'Infrastructure', 'ADANIENT': 'Infrastructure', 'ADANIPORTS': 'Infrastructure',
    'ULTRACEMCO': 'Infrastructure', 'GRASIM': 'Infrastructure', 'SHREECEM': 'Infrastructure',
    'AMBUJACEM': 'Infrastructure', 'ACC': 'Infrastructure', 'DALBHARAT': 'Infrastructure',
    
    # Metals
    'TATASTEEL': 'Metals', 'JSWSTEEL': 'Metals', 'HINDALCO': 'Metals', 'VEDL': 'Metals',
    'COALINDIA': 'Metals', 'NMDC': 'Metals', 'SAIL': 'Metals', 'JINDALSTEL': 'Metals',
    
    # Retail
    'TITAN': 'Retail', 'TRENT': 'Retail', 'DMART': 'Retail', 'PAGEIND': 'Retail',
    'ABFRL': 'Retail', 'RELAXO': 'Retail',
    
    # Insurance
    'SBILIFE': 'Insurance', 'HDFCLIFE': 'Insurance', 'ICICIPRULI': 'Insurance',
    'ICICIGI': 'Insurance', 'BAJAJHLDNG': 'Insurance', 'NIACL': 'Insurance',
    
    # Real Estate
    'DLF': 'Real Estate', 'GODREJPROP': 'Real Estate', 'OBEROIRLTY': 'Real Estate',
    'PHOENIXLTD': 'Real Estate', 'PRESTIGE': 'Real Estate', 'BRIGADE': 'Real Estate',
    
    # Chemicals
    'PIDILITIND': 'Chemicals', 'SRF': 'Chemicals', 'ATUL': 'Chemicals',
    'NAVINFLUOR': 'Chemicals', 'DEEPAKNI': 'Chemicals', 'CLEAN': 'Chemicals',
}

def analyze_sector_exposure(results):
    """
    Analyze sector exposure across portfolio
    Returns sector breakdown and warnings
    """
    sector_exposure = {}
    total_value = 0
    
    for r in results:
        ticker = r['ticker'].replace('.NS', '').replace('.BO', '').upper()
        sector = SECTOR_MAP.get(ticker, 'Other')
        position_value = r['entry_price'] * r['quantity']
        
        if sector not in sector_exposure:
            sector_exposure[sector] = {
                'value': 0,
                'count': 0,
                'stocks': [],
                'pnl': 0
            }
        
        sector_exposure[sector]['value'] += position_value
        sector_exposure[sector]['count'] += 1
        sector_exposure[sector]['stocks'].append(ticker)
        sector_exposure[sector]['pnl'] += r['pnl_amount']
        total_value += position_value
    
    # Calculate percentages
    sector_pct = {}
    for sector, data in sector_exposure.items():
        sector_pct[sector] = {
            'percentage': (data['value'] / total_value * 100) if total_value > 0 else 0,
            'count': data['count'],
            'stocks': data['stocks'],
            'value': data['value'],
            'pnl': data['pnl']
        }
    
    # Sort by percentage
    sector_pct_sorted = dict(sorted(sector_pct.items(),
                                    key=lambda x: x[1]['percentage'],
                                    reverse=True))
    
    # Warnings
    warnings = []
    for sector, data in sector_pct_sorted.items():
        if data['percentage'] > 40:
            warnings.append(f"🚨 {sector}: {data['percentage']:.1f}% - Highly over-exposed!")
        elif data['percentage'] > 30:
            warnings.append(f"⚠️ {sector}: {data['percentage']:.1f}% - Over-concentrated")
    
    # Diversification score
    num_sectors = len([s for s in sector_pct if sector_pct[s]['percentage'] > 5])
    if num_sectors >= 6:
        diversification_score = 90
    elif num_sectors >= 4:
        diversification_score = 70
    elif num_sectors >= 2:
        diversification_score = 50
    else:
        diversification_score = 30
    
    # Adjust for concentration
    max_concentration = max([d['percentage'] for d in sector_pct.values()]) if sector_pct else 0
    if max_concentration > 50:
        diversification_score -= 30
    elif max_concentration > 35:
        diversification_score -= 15
    
    diversification_score = max(0, min(100, diversification_score))
    
    return {
        'sectors': sector_pct_sorted,
        'warnings': warnings,
        'total_sectors': len(sector_pct),
        'diversification_score': diversification_score,
        'max_concentration': max_concentration,
        'total_value': total_value
    }

# ============================================================================
# PORTFOLIO RISK CALCULATION
# ============================================================================

def calculate_portfolio_risk(results):
    """
    Calculate overall portfolio risk metrics
    """
    if not results:
        return None
    
    total_capital = sum(r['entry_price'] * r['quantity'] for r in results)
    total_current_value = sum(r['current_price'] * r['quantity'] for r in results)
    total_pnl = sum(r['pnl_amount'] for r in results)
    
    # Calculate total risk amount (if all SL hit)
    total_risk_amount = 0
    for r in results:
        if r['position_type'] == 'LONG':
            loss_if_sl = (r['entry_price'] - r['stop_loss']) * r['quantity']
        else:
            loss_if_sl = (r['stop_loss'] - r['entry_price']) * r['quantity']
        total_risk_amount += max(loss_if_sl, 0)
    
    portfolio_risk_pct = (total_risk_amount / total_capital * 100) if total_capital > 0 else 0
    
    # Risk status
    if portfolio_risk_pct <= 5:
        risk_status = "SAFE"
        risk_color = "#28a745"
        risk_icon = "✅"
    elif portfolio_risk_pct <= 10:
        risk_status = "MEDIUM"
        risk_color = "#ffc107"
        risk_icon = "⚠️"
    else:
        risk_status = "HIGH"
        risk_color = "#dc3545"
        risk_icon = "🚨"
    
    # Count risky positions
    risky_positions = sum(1 for r in results if r['sl_risk'] >= 50)
    critical_positions = sum(1 for r in results if r['overall_status'] == 'CRITICAL')
    
    # Average SL risk
    avg_sl_risk = sum(r['sl_risk'] for r in results) / len(results) if results else 0
    
    return {
        'total_capital': total_capital,
        'current_value': total_current_value,
        'total_pnl': total_pnl,
        'total_pnl_pct': (total_pnl / total_capital * 100) if total_capital > 0 else 0,
        'total_risk_amount': total_risk_amount,
        'portfolio_risk_pct': portfolio_risk_pct,
        'risk_status': risk_status,
        'risk_color': risk_color,
        'risk_icon': risk_icon,
        'risky_positions': risky_positions,
        'critical_positions': critical_positions,
        'avg_sl_risk': avg_sl_risk,
        'total_positions': len(results)
    }

# ============================================================================
# NEW: POSITION PRIORITY SCORING
# ============================================================================

def calculate_position_priority(result, market_health):
    """
    Score each position for urgency of attention.
    Higher score = needs MORE immediate attention.
    Scale: 0-100
    """
    urgency = 0
    factors = []
    
    # SL Risk (0-30 points)
    sl_risk = result['sl_risk']
    if sl_risk >= 80:
        urgency += 30
        factors.append(f"SL Risk {sl_risk}%")
    elif sl_risk >= 60:
        urgency += 20
        factors.append(f"SL Risk {sl_risk}%")
    elif sl_risk >= 40:
        urgency += 10
    
    # P&L extremes (0-20 points)
    pnl = result['pnl_percent']
    if pnl < -5:
        urgency += 20
        factors.append(f"Heavy loss: {pnl:.1f}%")
    elif pnl < -3:
        urgency += 15
        factors.append(f"Loss: {pnl:.1f}%")
    elif pnl > 10:
        urgency += 15
        factors.append(f"Big profit: {pnl:.1f}% - protect it!")
    elif pnl > 5:
        urgency += 10
        factors.append(f"Good profit: {pnl:.1f}%")
    
    # Target/SL hit (0-25 points)
    if result['sl_hit']:
        urgency += 25
        factors.append("🚨 SL HIT!")
    elif result['target2_hit']:
        urgency += 20
        factors.append("🎯 Target 2 hit - book profits!")
    elif result['target1_hit']:
        urgency += 15
        factors.append("🎯 Target 1 hit")
    
    # Market against position (0-15 points)
    if market_health:
        if (market_health['status'] == 'BEARISH' and
            result['position_type'] == 'LONG' and pnl < 0):
            urgency += 15
            factors.append("🐻 Bear market + losing LONG")
        elif (market_health['status'] == 'BULLISH' and
              result['position_type'] == 'SHORT' and pnl < 0):
            urgency += 15
            factors.append("🐂 Bull market + losing SHORT")
    
    # MTF misalignment (0-10 points)
    if result.get('mtf_alignment', 50) < 25:
        urgency += 10
        factors.append(f"MTF against ({result['mtf_alignment']}%)")
    
    urgency = min(100, urgency)
    
    if urgency >= 70:
        priority = "🔴 CRITICAL"
    elif urgency >= 50:
        priority = "🟠 HIGH"
    elif urgency >= 30:
        priority = "🟡 MEDIUM"
    else:
        priority = "🟢 LOW"
    
    return {
        'urgency_score': urgency,
        'priority': priority,
        'factors': factors,
        'needs_immediate_action': urgency >= 60
    }

# ============================================================================
# 9:20AM INTRADAY CONFIRMATION MODULE
# ============================================================================

def is_confirmation_window():
    """
    Returns True only between 9:20AM and 9:35AM IST.
    Uses existing get_ist_now() - no pytz dependency needed.
    """
    ist_now = get_ist_now()
    
    # Only on weekdays
    if ist_now.weekday() >= 5:
        return False
    
    hour = ist_now.hour
    minute = ist_now.minute
    
    # 9:20 to 9:35
    if hour == 9 and 20 <= minute <= 35:
        return True
    
    return False


def is_past_entry_window():
    """After 9:35AM IST - entry window closed for the day"""
    ist_now = get_ist_now()
    
    if ist_now.weekday() >= 5:
        return True
    
    hour = ist_now.hour
    minute = ist_now.minute
    
    # Past 9:35
    if hour > 9 or (hour == 9 and minute > 35):
        return True
    
    return False


def get_15min_confirmation(ticker, side, daily_entry_price):
    """
    Check 15-min chart at 9:20AM to confirm daily signal is still valid.
    
    Args:
        ticker: Stock symbol (e.g., "RELIANCE")
        side: "LONG" or "SHORT"
        daily_entry_price: The planned entry price from daily scanner
    
    Returns:
        dict with:
            'confirmed': bool - True if setup is still valid
            'reason': str - Human readable explanation
            'skip': bool - True if trade should be skipped
            'reduce_size': bool - True if should enter with smaller size
            'gap_pct': float - Gap percentage from entry
            'vol_ratio': float - Volume ratio vs average
            'checks_passed': int - Number of checks passed (0-4)
            'checks_total': int - Total checks run
    """
    symbol = ticker if '.NS' in str(ticker) else f"{ticker}.NS"
    checks_passed = 0
    checks_total = 0
    details = []
    
    try:
        # ================================================================
        # GET 15-MINUTE DATA
        # ================================================================
        
        # Method 1: Try Kite API first (more accurate)
        df_15 = None
        
        if (HAS_KITE and kite_session and kite_session.is_connected):
            try:
                ticker_clean = (
                    ticker.replace('.NS', '').replace('.BO', '')
                    .upper().strip()
                )
                
                # Get 15-min data via Kite quote (simplified approach)
                # Full historical requires instrument tokens
                quote = kite_session.kite.quote(f"NSE:{ticker_clean}")
                ohlc = quote.get(f"NSE:{ticker_clean}", {}).get("ohlc", {})
                
                if ohlc:
                    current_price = quote[f"NSE:{ticker_clean}"]["last_price"]
                    day_open = ohlc.get("open", 0)
                    day_high = ohlc.get("high", 0)
                    day_low = ohlc.get("low", 0)
                    volume = quote[f"NSE:{ticker_clean}"].get(
                        "volume", 0
                    )
                    avg_volume = quote[f"NSE:{ticker_clean}"].get(
                        "average_price", 0
                    )
                    
                    logger.info(
                        f"Kite data for {ticker_clean}: "
                        f"Open={day_open}, CMP={current_price}"
                    )
            except Exception as e:
                logger.warning(f"Kite 15min failed for {ticker}: {e}")
        
        # Method 2: Fallback to yfinance
        if df_15 is None:
            try:
                rate_limited_api_call(symbol, min_interval=0.5)
                stock = yf.Ticker(symbol)
                df_15 = stock.history(period="2d", interval="15m")
            except Exception as e:
                logger.error(f"yfinance 15min failed for {ticker}: {e}")
        
        # If we got yfinance data, extract what we need
        if df_15 is not None and len(df_15) >= 5:
            # Get today's candles
            if hasattr(df_15.index, 'date'):
                today_date = get_ist_now().date()
                today_mask = df_15.index.date == today_date
                today_candles = df_15[today_mask]
            else:
                # Fallback: use last few candles
                today_candles = df_15.tail(5)
            
            if len(today_candles) == 0:
                return {
                    'confirmed': False,
                    'reason': 'No today data - market may not be open',
                    'skip': True,
                    'reduce_size': False,
                    'gap_pct': 0,
                    'vol_ratio': 0,
                    'checks_passed': 0,
                    'checks_total': 0
                }
            
            day_open = float(today_candles.iloc[0]['Open'])
            current_price = float(today_candles.iloc[-1]['Close'])
            day_high = float(today_candles['High'].max())
            day_low = float(today_candles['Low'].min())
            
            # Volume from today's candles
            today_vol = float(today_candles['Volume'].sum())
            avg_vol = float(df_15['Volume'].mean()) if len(df_15) > 0 else 1
            
            # Close series for EMA/MACD
            close_15 = df_15['Close'].astype(float)
        
        elif 'day_open' in dir() and day_open > 0:
            # We got data from Kite quote
            today_candles = None
            close_15 = None
            today_vol = volume if 'volume' in dir() else 0
            avg_vol = 1
        
        else:
            return {
                'confirmed': True,
                'reason': (
                    'No 15-min data available - '
                    'proceeding with daily signal only'
                ),
                'skip': False,
                'reduce_size': False,
                'gap_pct': 0,
                'vol_ratio': 1,
                'checks_passed': 0,
                'checks_total': 0
            }
        
        # ================================================================
        # CHECK 1: GAP CHECK
        # ================================================================
        checks_total += 1
        
        gap_pct = (
            (day_open - daily_entry_price) / daily_entry_price
        ) * 100
        
        if side == 'LONG':
            if gap_pct < -0.5:
                details.append(
                    f"❌ Gap down {gap_pct:.2f}% from entry"
                )
                return {
                    'confirmed': False,
                    'reason': (
                        f'Gap down {gap_pct:.2f}% from entry '
                        f'₹{daily_entry_price:.2f} - setup broken'
                    ),
                    'skip': True,
                    'reduce_size': False,
                    'gap_pct': gap_pct,
                    'vol_ratio': 0,
                    'checks_passed': checks_passed,
                    'checks_total': checks_total
                }
            elif gap_pct > 1.5:
                details.append(
                    f"❌ Gap up {gap_pct:.2f}% - entry too expensive"
                )
                return {
                    'confirmed': False,
                    'reason': (
                        f'Gap up {gap_pct:.2f}% - '
                        f'entry price too far from plan'
                    ),
                    'skip': True,
                    'reduce_size': False,
                    'gap_pct': gap_pct,
                    'vol_ratio': 0,
                    'checks_passed': checks_passed,
                    'checks_total': checks_total
                }
            else:
                checks_passed += 1
                details.append(f"✅ Gap OK: {gap_pct:+.2f}%")
        
        else:  # SHORT
            if gap_pct > 0.5:
                details.append(
                    f"❌ Gap up {gap_pct:.2f}% from entry"
                )
                return {
                    'confirmed': False,
                    'reason': (
                        f'Gap up {gap_pct:.2f}% from entry - '
                        f'short setup broken'
                    ),
                    'skip': True,
                    'reduce_size': False,
                    'gap_pct': gap_pct,
                    'vol_ratio': 0,
                    'checks_passed': checks_passed,
                    'checks_total': checks_total
                }
            elif gap_pct < -1.5:
                details.append(
                    f"❌ Gap down {gap_pct:.2f}% - entry too far"
                )
                return {
                    'confirmed': False,
                    'reason': (
                        f'Gap down {gap_pct:.2f}% - '
                        f'short entry too far from plan'
                    ),
                    'skip': True,
                    'reduce_size': False,
                    'gap_pct': gap_pct,
                    'vol_ratio': 0,
                    'checks_passed': checks_passed,
                    'checks_total': checks_total
                }
            else:
                checks_passed += 1
                details.append(f"✅ Gap OK: {gap_pct:+.2f}%")
        
        # ================================================================
        # CHECK 2: EMA CHECK ON 15-MIN
        # ================================================================
        checks_total += 1
        
        ema_ok = True
        ema_reason = ""
        
        if close_15 is not None and len(close_15) >= 20:
            ema20_15 = close_15.ewm(span=20, adjust=False).mean()
            current_close = float(close_15.iloc[-1])
            current_ema20 = float(ema20_15.iloc[-1])
            
            if side == 'LONG' and current_close < current_ema20 * 0.998:
                ema_ok = False
                ema_reason = (
                    f'Price ₹{current_close:.2f} below '
                    f'15-min EMA20 ₹{current_ema20:.2f}'
                )
                details.append(f"❌ {ema_reason}")
            elif side == 'SHORT' and current_close > current_ema20 * 1.002:
                ema_ok = False
                ema_reason = (
                    f'Price ₹{current_close:.2f} above '
                    f'15-min EMA20 ₹{current_ema20:.2f}'
                )
                details.append(f"❌ {ema_reason}")
            else:
                checks_passed += 1
                details.append(
                    f"✅ EMA OK: Price ₹{current_close:.2f} "
                    f"vs EMA ₹{current_ema20:.2f}"
                )
        else:
            # Can't check EMA - give benefit of doubt
            checks_passed += 1
            details.append("✅ EMA check skipped (insufficient data)")
        
        if not ema_ok:
            return {
                'confirmed': False,
                'reason': ema_reason,
                'skip': True,
                'reduce_size': False,
                'gap_pct': gap_pct,
                'vol_ratio': 0,
                'checks_passed': checks_passed,
                'checks_total': checks_total
            }
        
        # ================================================================
        # CHECK 3: VOLUME CHECK
        # ================================================================
        checks_total += 1
        
        if today_candles is not None and len(today_candles) >= 2:
            first_2_vol = float(
                today_candles.iloc[:2]['Volume'].mean()
            )
            avg_vol_15 = float(df_15['Volume'].mean()) if avg_vol > 0 else 1
            vol_ratio = first_2_vol / avg_vol_15 if avg_vol_15 > 0 else 1.0
        elif today_vol > 0 and avg_vol > 0:
            vol_ratio = today_vol / avg_vol
        else:
            vol_ratio = 1.0
        
        if vol_ratio < 0.4:
            details.append(
                f"❌ Volume very weak ({vol_ratio:.1f}x avg)"
            )
            return {
                'confirmed': False,
                'reason': (
                    f'Opening volume very weak '
                    f'({vol_ratio:.1f}x) - no interest'
                ),
                'skip': True,
                'reduce_size': False,
                'gap_pct': gap_pct,
                'vol_ratio': vol_ratio,
                'checks_passed': checks_passed,
                'checks_total': checks_total
            }
        else:
            checks_passed += 1
            details.append(f"✅ Volume OK: {vol_ratio:.1f}x avg")
        
        # ================================================================
        # CHECK 4: MACD ON 15-MIN
        # ================================================================
        checks_total += 1
        reduce_size = False
        
        if close_15 is not None and len(close_15) >= 26:
            exp12 = close_15.ewm(span=12, adjust=False).mean()
            exp26 = close_15.ewm(span=26, adjust=False).mean()
            macd_line = exp12 - exp26
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            
            macd_val = float(macd_line.iloc[-1])
            signal_val = float(macd_signal.iloc[-1])
            
            if side == 'LONG' and macd_val < signal_val:
                reduce_size = True
                details.append(
                    "⚠️ MACD bearish on 15-min - reduce size"
                )
                # Don't skip, just reduce size
            elif side == 'SHORT' and macd_val > signal_val:
                reduce_size = True
                details.append(
                    "⚠️ MACD bullish on 15-min - reduce size"
                )
            else:
                checks_passed += 1
                details.append("✅ MACD confirms direction")
        else:
            checks_passed += 1
            details.append("✅ MACD check skipped (insufficient data)")
        
        # ================================================================
        # ALL CHECKS PASSED
        # ================================================================
        
        reason = (
            f"Passed {checks_passed}/{checks_total} checks | "
            f"Gap: {gap_pct:+.2f}% | Vol: {vol_ratio:.1f}x"
        )
        
        if reduce_size:
            reason += " | ⚠️ Reduce position size"
        
        return {
            'confirmed': True,
            'reason': reason,
            'skip': False,
            'reduce_size': reduce_size,
            'gap_pct': gap_pct,
            'vol_ratio': vol_ratio,
            'checks_passed': checks_passed,
            'checks_total': checks_total,
            'details': details
        }
    
    except Exception as e:
        logger.error(f"15min confirmation error for {ticker}: {e}")
        return {
            'confirmed': True,
            'reason': (
                f'Confirmation error: {str(e)[:50]} - '
                f'proceeding with caution'
            ),
            'skip': False,
            'reduce_size': False,
            'gap_pct': 0,
            'vol_ratio': 1,
            'checks_passed': 0,
            'checks_total': 0
        }


def update_sheet_status(ticker, new_status, reason=""):
    """
    Update the Status column in Google Sheet.
    Used to mark PENDING as ACTIVE, SKIPPED, etc.
    """
    sheet, status = get_google_sheet_connection()
    
    if sheet is None:
        return False, status
    
    try:
        cell = sheet.find(ticker)
        if not cell:
            return False, f"Ticker {ticker} not found"
        
        row = cell.row
        headers = sheet.row_values(1)
        
        try:
            status_col = headers.index('Status') + 1
        except ValueError:
            status_col = 9
        
        sheet.update_cell(row, status_col, new_status)
        
        # Also update a notes column if exists
        try:
            if 'Notes' in headers:
                notes_col = headers.index('Notes') + 1
            elif 'Last_Auto_Action' in headers:
                notes_col = headers.index('Last_Auto_Action') + 1
            else:
                notes_col = None
            
            if notes_col:
                note = (
                    f"{new_status}: {reason} "
                    f"({get_ist_now().strftime('%H:%M')})"
                )
                sheet.update_cell(row, notes_col, note)
        except Exception:
            pass
        
        log_email(
            f"📋 Status updated: {ticker} → {new_status} ({reason})"
        )
        return True, f"{ticker} marked as {new_status}"
    
    except Exception as e:
        return False, f"Error updating status: {str(e)}"


# ============================================================================
# NEW: AUTOMATION SAFETY FUNCTIONS
# ============================================================================

def get_daily_update_count(ticker, update_type="sl"):
    """
    Count how many times we've updated this stock today.
    Prevents excessive API calls to Google Sheets.
    """
    today = get_ist_now().strftime('%Y%m%d')
    count = 0
    
    for key in st.session_state:
        if isinstance(key, str) and key.startswith(f"{update_type}_updated_{ticker}_{today}"):
            count += 1
        if isinstance(key, str) and key.startswith(f"market_sl_{ticker}_{today}"):
            count += 1
    
    return count


def is_sl_improvement(current_sl, new_sl, position_type):
    """
    Check if the new SL is BETTER (tighter/closer to price) than the old SL.
    We NEVER move SL backwards (away from price).
    
    For LONG: New SL must be HIGHER than old SL (closer to price from below)
    For SHORT: New SL must be LOWER than old SL (closer to price from above)
    """
    if position_type == 'LONG':
        return new_sl > current_sl  # Higher SL = better for LONG
    else:
        return new_sl < current_sl  # Lower SL = better for SHORT


def verify_price_before_exit(ticker, expected_price, tolerance_pct=3.0):
    """
    SAFETY: Before auto-exiting, verify the price with a fresh API call.
    This prevents selling due to stale/glitched data.
    
    Returns: (is_valid, actual_price, reason)
    """
    try:
        realtime = get_realtime_price(ticker)
        
        if realtime is None:
            # Can't verify - DON'T auto-exit
            return False, 0, "Cannot verify price - skipping auto-exit"
        
        actual_price = realtime['price']
        
        # Check if actual price is within tolerance of expected price
        price_diff_pct = abs(actual_price - expected_price) / expected_price * 100
        
        if price_diff_pct > tolerance_pct:
            return False, actual_price, (
                f"Price mismatch: Expected ₹{expected_price:.2f}, "
                f"Got ₹{actual_price:.2f} ({price_diff_pct:.1f}% diff)"
            )
        
        return True, actual_price, "Price verified OK"
    
    except Exception as e:
        return False, 0, f"Verification failed: {str(e)}"


MAX_SL_UPDATES_PER_DAY = 3  # Maximum SL updates per stock per day
MAX_AUTO_EXITS_PER_DAY = 2  # Maximum auto-exits per day (across all stocks)


def get_daily_auto_exit_count():
    """Count total auto-exits today"""
    today = get_ist_now().strftime('%Y%m%d')
    count = 0
    for key in st.session_state:
        if isinstance(key, str) and key.startswith(f"auto_exit_") and today in key:
            count += 1
    return count



# ============================================================================
# PARTIAL PROFIT BOOKING TRACKER
# ============================================================================

def calculate_partial_exit_levels(entry_price, target1, target2, position_type):
    """
    Calculate recommended partial exit levels
    """
    if position_type == "LONG":
        move = target1 - entry_price
        levels = [
            {'level': entry_price + move * 0.5, 'exit_pct': 25, 'reason': '50% to T1'},
            {'level': target1, 'exit_pct': 25, 'reason': 'Target 1'},
            {'level': entry_price + (target2 - entry_price) * 0.75, 'exit_pct': 25, 'reason': '75% to T2'},
            {'level': target2, 'exit_pct': 25, 'reason': 'Target 2'},
        ]
    else:
        move = entry_price - target1
        levels = [
            {'level': entry_price - move * 0.5, 'exit_pct': 25, 'reason': '50% to T1'},
            {'level': target1, 'exit_pct': 25, 'reason': 'Target 1'},
            {'level': entry_price - (entry_price - target2) * 0.75, 'exit_pct': 25, 'reason': '75% to T2'},
            {'level': target2, 'exit_pct': 25, 'reason': 'Target 2'},
        ]
    
    return levels

def track_partial_exit(ticker, current_price, entry_price, quantity, position_type, target1, target2):
    """
    Track partial exit recommendations based on current price
    """
    levels = calculate_partial_exit_levels(entry_price, target1, target2, position_type)
    
    recommendations = []
    remaining_qty = quantity
    
    for level in levels:
        if position_type == "LONG":
            if current_price >= level['level']:
                exit_qty = int(quantity * level['exit_pct'] / 100)
                if exit_qty > 0:
                    recommendations.append({
                        'level': level['level'],
                        'exit_pct': level['exit_pct'],
                        'exit_qty': exit_qty,
                        'reason': level['reason'],
                        'status': 'TRIGGERED',
                        'pnl': (level['level'] - entry_price) * exit_qty
                    })
                    remaining_qty -= exit_qty
        else:
            if current_price <= level['level']:
                exit_qty = int(quantity * level['exit_pct'] / 100)
                if exit_qty > 0:
                    recommendations.append({
                        'level': level['level'],
                        'exit_pct': level['exit_pct'],
                        'exit_qty': exit_qty,
                        'reason': level['reason'],
                        'status': 'TRIGGERED',
                        'pnl': (entry_price - level['level']) * exit_qty
                    })
                    remaining_qty -= exit_qty
    
    # Add pending levels
    for level in levels:
        already_added = any(r['level'] == level['level'] for r in recommendations)
        if not already_added:
            exit_qty = int(quantity * level['exit_pct'] / 100)
            recommendations.append({
                'level': level['level'],
                'exit_pct': level['exit_pct'],
                'exit_qty': exit_qty,
                'reason': level['reason'],
                'status': 'PENDING',
                'pnl': 0
            })
    
    return {
        'recommendations': recommendations,
        'remaining_qty': max(0, remaining_qty),
        'triggered_count': sum(1 for r in recommendations if r['status'] == 'TRIGGERED'),
        'total_booked_pnl': sum(r['pnl'] for r in recommendations if r['status'] == 'TRIGGERED')
    }

# ============================================================================
# COMPLETE SMART ANALYSIS FUNCTION
# ============================================================================

@st.cache_data(ttl=15)  # 15 second cache
def smart_analyze_position(ticker, position_type, entry_price, quantity, stop_loss,
                          target1, target2, trail_threshold=2.0, sl_alert_threshold=50,
                          sl_approach_threshold=2.0, enable_mtf=True, entry_date=None):
    """
    Complete smart analysis with all features
    Accepts sidebar parameters for dynamic thresholds
    """
    df = get_stock_data_safe(ticker, period="6mo")
    if df is None or df.empty:
        return None
    
    try:
        current_price = float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
        day_change = ((current_price - prev_close) / prev_close) * 100
        day_high = float(df['High'].iloc[-1])
        day_low = float(df['Low'].iloc[-1])
    except Exception as e:
        return None
    
    # Basic P&L
    if position_type == "LONG":
        pnl_percent = ((current_price - entry_price) / entry_price) * 100
        pnl_amount = (current_price - entry_price) * quantity
    else:
        pnl_percent = ((entry_price - current_price) / entry_price) * 100
        pnl_amount = (entry_price - current_price) * quantity
    
    # Technical Indicators
    rsi = float(calculate_rsi(df['Close']).iloc[-1])
    if pd.isna(rsi):
        rsi = 50.0
    
    macd, signal, histogram = calculate_macd(df['Close'])
    macd_hist = float(histogram.iloc[-1]) if len(histogram) > 0 else 0
    if pd.isna(macd_hist):
        macd_hist = 0
    macd_signal = "BULLISH" if macd_hist > 0 else "BEARISH"
    
    # Stochastic
    stoch_k, stoch_d = calculate_stochastic(df['High'], df['Low'], df['Close'])
    stoch_k_val = float(stoch_k.iloc[-1]) if not pd.isna(stoch_k.iloc[-1]) else 50
    stoch_d_val = float(stoch_d.iloc[-1]) if not pd.isna(stoch_d.iloc[-1]) else 50
    
    # Momentum Score
    momentum_score, momentum_trend, momentum_components = calculate_momentum_score(df)
    
    # Volume Analysis
    volume_signal, volume_ratio, volume_desc, volume_trend = analyze_volume(df)
    
    # Support/Resistance
    sr_levels = find_support_resistance(df)
    
    # SL Risk Prediction
    sl_risk, sl_reasons, sl_recommendation, sl_priority = predict_sl_risk(
        df, current_price, stop_loss, position_type, entry_price, sl_alert_threshold
    )
    
    # Multi-Timeframe Analysis
    if enable_mtf:
        mtf_result = multi_timeframe_analysis(ticker, position_type)
    else:
        mtf_result = {
            'signals': {},
            'details': {},
            'alignment_score': 50,
            'recommendation': "MTF disabled",
            'aligned_count': 0,
            'against_count': 0,
            'total_timeframes': 0,
            'trend_strength': 'UNKNOWN'
        }
    
    # Check if target hit
    if position_type == "LONG":
        target1_hit = current_price >= target1
        target2_hit = current_price >= target2
        sl_hit = current_price <= stop_loss
    else:
        target1_hit = current_price <= target1
        target2_hit = current_price <= target2
        sl_hit = current_price >= stop_loss
    
    # Upside prediction (if target hit)
    if target1_hit and not sl_hit:
        upside_score, new_target, upside_reasons, upside_rec, upside_action = predict_upside_potential(
            df, current_price, target1, target2, position_type
        )
    else:
        upside_score = 0
        new_target = target2
        upside_reasons = []
        upside_rec = ""
        upside_action = ""
    
    # Dynamic Levels
    dynamic_levels = calculate_dynamic_levels(
        df, entry_price, current_price, stop_loss, position_type, pnl_percent, trail_threshold
    )
    
    # NEW: Smart Stop Loss (volatility-adjusted)
    smart_sl_data = calculate_smart_stop_loss(
        df, entry_price, current_price, position_type,
        risk_percent=2.0, market_health=None  # market_health passed later in main()
    )
    
    # NEW: Intraday SL monitoring (only during market hours)
    is_open_now, _, _, _ = is_market_hours()
    intraday_sl_data = None
    if is_open_now:
        intraday_sl_data = monitor_intraday_sl(ticker, stop_loss, position_type)
    
    # Partial Exit Tracking
    partial_exits = track_partial_exit(
        ticker, current_price, entry_price, quantity, position_type, target1, target2
    )
    
    # Holding Period & Tax
    if entry_date:
        holding_days = calculate_holding_period(entry_date)
        tax_implication, tax_color = get_tax_implication(holding_days, pnl_amount)
    else:
        holding_days = 0
        tax_implication = "Entry date not provided"
        tax_color = "⚪"
    
    # Breakeven check
    breakeven_distance = abs(pnl_percent)
    at_breakeven = breakeven_distance < 0.5 and pnl_percent >= 0
    
    # Distance to SL (for approach warning)
    if position_type == "LONG":
        distance_to_sl = ((current_price - stop_loss) / current_price) * 100
    else:
        distance_to_sl = ((stop_loss - current_price) / current_price) * 100
    
    approaching_sl = distance_to_sl > 0 and distance_to_sl <= sl_approach_threshold
    
    # =========================================================================
    # GENERATE ALERTS AND DETERMINE OVERALL STATUS
    # =========================================================================
    alerts = []
    overall_status = 'OK'
    overall_action = 'HOLD'
    
    # Priority 1: SL Hit
    if sl_hit:
        alerts.append({
            'priority': 'CRITICAL',
            'type': '🚨 STOP LOSS HIT',
            'message': f'Price ₹{current_price:.2f} breached SL ₹{stop_loss:.2f}',
            'action': 'EXIT IMMEDIATELY',
            'email_type': 'critical'
        })
        overall_status = 'CRITICAL'
        overall_action = 'EXIT'
    
    # Priority 2: High SL Risk (Early Exit Warning)
    elif sl_risk >= sl_alert_threshold + 20:
        alerts.append({
            'priority': 'CRITICAL',
            'type': '⚠️ HIGH SL RISK',
            'message': f'Risk Score: {sl_risk}% - {", ".join(sl_reasons[:2])}',
            'action': sl_recommendation,
            'email_type': 'critical'
        })
        overall_status = 'CRITICAL'
        overall_action = 'EXIT_EARLY'
    
    # Priority 3: Approaching SL
    elif approaching_sl:
        alerts.append({
            'priority': 'HIGH',
            'type': '⚠️ APPROACHING SL',
            'message': f'Only {distance_to_sl:.1f}% away from Stop Loss!',
            'action': 'Review position - consider early exit',
            'email_type': 'sl_approach'
        })
        if overall_status == 'OK':
            overall_status = 'WARNING'
            overall_action = 'WATCH'
    
    # Priority 4: Moderate SL Risk
    elif sl_risk >= sl_alert_threshold:
        alerts.append({
            'priority': 'HIGH',
            'type': '⚠️ MODERATE SL RISK',
            'message': f'Risk Score: {sl_risk}% - {", ".join(sl_reasons[:2])}',
            'action': sl_recommendation,
            'email_type': 'important'
        })
        overall_status = 'WARNING'
        overall_action = 'WATCH'
    
    # Priority 5: Target 2 Hit
    elif target2_hit:
        alerts.append({
            'priority': 'HIGH',
            'type': '🎯 TARGET 2 HIT',
            'message': f'Both targets achieved! P&L: {pnl_percent:+.2f}%',
            'action': 'BOOK FULL PROFITS',
            'email_type': 'target'
        })
        overall_status = 'SUCCESS'
        overall_action = 'BOOK_PROFITS'
    
    # Priority 6: Target 1 Hit with Upside Analysis
    elif target1_hit:
        if upside_score >= 60:
            alerts.append({
                'priority': 'INFO',
                'type': '🎯 TARGET HIT - HOLD',
                'message': f'Upside Score: {upside_score}% - {", ".join(upside_reasons[:2])}',
                'action': f'{upside_action}',
                'email_type': 'target'
            })
            overall_status = 'OPPORTUNITY'
            overall_action = 'HOLD_EXTEND'
        else:
            alerts.append({
                'priority': 'HIGH',
                'type': '🎯 TARGET HIT - EXIT',
                'message': f'Limited upside ({upside_score}%). Book profits.',
                'action': 'BOOK PROFITS',
                'email_type': 'target'
            })
            overall_status = 'SUCCESS'
            overall_action = 'BOOK_PROFITS'
    
    # Priority 7: Trail Stop Recommendation
    elif dynamic_levels['should_trail'] and pnl_percent >= trail_threshold:
        alerts.append({
            'priority': 'MEDIUM',
            'type': '📈 TRAIL STOP LOSS',
            'message': f'{dynamic_levels.get("trail_reason", "Lock profits!")} Move SL from ₹{stop_loss:.2f} to ₹{dynamic_levels["trail_stop"]:.2f}',
            'action': f'New SL: ₹{dynamic_levels["trail_stop"]:.2f}',
            'email_type': 'sl_change'
        })
        overall_status = 'GOOD'
        overall_action = 'TRAIL_SL'
    
    # Priority 8: MTF Warning
    elif enable_mtf and mtf_result['alignment_score'] < 40 and pnl_percent < 0:
        alerts.append({
            'priority': 'MEDIUM',
            'type': '📊 MTF WARNING',
            'message': f'Timeframes against position ({mtf_result["alignment_score"]}% aligned)',
            'action': mtf_result['recommendation'],
            'email_type': 'important'
        })
        overall_status = 'WARNING'
        overall_action = 'WATCH'
    
    # Priority 9: Breakeven Alert
    elif at_breakeven:
        alerts.append({
            'priority': 'LOW',
            'type': '🔔 BREAKEVEN REACHED',
            'message': f'Position at breakeven. Consider moving SL to entry (₹{entry_price:.2f})',
            'action': f'Move SL to ₹{entry_price:.2f} (breakeven)',
            'email_type': 'important'
        })
        if overall_status == 'OK':
            overall_status = 'GOOD'
            overall_action = 'MOVE_SL_BREAKEVEN'
    
    # Priority 10: Partial Exit Alert
    if partial_exits['triggered_count'] > 0 and not target2_hit:
        triggered = [r for r in partial_exits['recommendations'] if r['status'] == 'TRIGGERED']
        if triggered:
            latest = triggered[-1]
            alerts.append({
                'priority': 'LOW',
                'type': '📊 PARTIAL EXIT',
                'message': f'Level ₹{latest["level"]:.2f} triggered - Book {latest["exit_pct"]}% ({latest["exit_qty"]} shares)',
                'action': f'Exit {latest["exit_qty"]} shares at ₹{current_price:.2f}',
                'email_type': 'important'
            })
    
    # Volume Warning
    if position_type == "LONG" and volume_signal == "STRONG_SELLING" and sl_risk < sl_alert_threshold:
        alerts.append({
            'priority': 'LOW',
            'type': '📊 VOLUME WARNING',
            'message': volume_desc,
            'action': 'Monitor closely',
            'email_type': 'important'
        })
    elif position_type == "SHORT" and volume_signal == "STRONG_BUYING" and sl_risk < sl_alert_threshold:
        alerts.append({
            'priority': 'LOW',
            'type': '📊 VOLUME WARNING',
            'message': volume_desc,
            'action': 'Monitor closely',
            'email_type': 'important'
        })
    
    # Calculate Risk-Reward Ratio
    if position_type == "LONG":
        risk = entry_price - stop_loss
        reward = target1 - entry_price
    else:
        risk = stop_loss - entry_price
        reward = entry_price - target1
    
    risk_reward_ratio = safe_divide(reward, risk, default=0.0)
    
    # =========================================================================
    # ⚡ UNIFIED DECISION: Smart Exit overrides basic logic when needed
    # =========================================================================
    
    # Build a temporary result dict for smart_exit_decision to use
    temp_result = {
        'current_price': current_price,
        'entry_price': entry_price,
        'position_type': position_type,
        'pnl_percent': pnl_percent,
        'rsi': rsi,
        'macd_hist': macd_hist,
        'momentum_score': momentum_score,
        'volume_signal': volume_signal,
        'volume_ratio': volume_ratio,
        'sl_risk': sl_risk,
        'distance_to_support': sr_levels['distance_to_support'],
        'distance_to_resistance': sr_levels['distance_to_resistance'],
        'support': sr_levels['nearest_support'],
        'resistance': sr_levels['nearest_resistance'],
        'holding_days': holding_days,
        'mtf_alignment': mtf_result['alignment_score'],
        'sl_hit': sl_hit,
        'target1_hit': target1_hit,
        'target2_hit': target2_hit,
    }
    
    # Get market health for smart exit (may be None, that's OK)
    # We pass None here - market_health is applied in main() display
    exit_decision = smart_exit_decision(temp_result, None, df)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # OVERRIDE RULES: Smart Exit can override basic logic
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # Rule 1: If SL is already hit, NEVER override (highest priority)
    # Rule 2: If Target 2 hit, keep SUCCESS (book profits)
    # Rule 3: For everything else, Smart Exit can override
    
    if not sl_hit and not target2_hit:
        
        # CRITICAL OVERRIDE: Smart Exit says EXIT_NOW
        if exit_decision['action'] == 'EXIT_NOW' and exit_decision['score'] <= 15:
            overall_status = 'CRITICAL'
            overall_action = 'EXIT'
            alerts.insert(0, {
                'priority': 'CRITICAL',
                'type': '🧠 SMART EXIT - EXIT NOW',
                'message': f"Score: {exit_decision['score']}/100 - {exit_decision['summary']}",
                'action': 'EXIT IMMEDIATELY - Momentum collapse detected',
                'email_type': 'critical'
            })
        
        # HIGH OVERRIDE: Smart Exit says EXIT_TODAY
        elif exit_decision['action'] == 'EXIT_TODAY' and exit_decision['score'] <= 25:
            if overall_status not in ['CRITICAL']:
                overall_status = 'WARNING'
                overall_action = 'EXIT_TODAY'
                alerts.insert(0, {
                    'priority': 'HIGH',
                    'type': '🧠 SMART EXIT - EXIT TODAY',
                    'message': f"Score: {exit_decision['score']}/100 - {exit_decision['summary']}",
                    'action': 'Plan exit before market close',
                    'email_type': 'critical'
                })
        
        # MEDIUM OVERRIDE: Smart Exit says PARTIAL_EXIT
        elif exit_decision['action'] == 'PARTIAL_EXIT' and exit_decision['score'] <= 35:
            if overall_status not in ['CRITICAL', 'WARNING']:
                overall_status = 'WARNING'
                overall_action = 'PARTIAL_EXIT'
                alerts.insert(0, {
                    'priority': 'MEDIUM',
                    'type': '🧠 SMART EXIT - PARTIAL EXIT',
                    'message': f"Score: {exit_decision['score']}/100 - {exit_decision['summary']}",
                    'action': 'Book 50% profits, hold rest with tight SL',
                    'email_type': 'important'
                })
        
        # TIGHTEN SL suggestion
        elif exit_decision['action'] == 'TIGHTEN_SL' and exit_decision['score'] <= 45:
            if overall_status in ['OK', 'GOOD']:
                overall_action = 'TIGHTEN_SL'
                alerts.append({
                    'priority': 'MEDIUM',
                    'type': '🧠 SMART EXIT - TIGHTEN SL',
                    'message': f"Score: {exit_decision['score']}/100 - Caution warranted",
                    'action': 'Move SL closer to protect capital',
                    'email_type': 'important'
                })
        
        # POSITIVE: Smart Exit says ADD_MORE or HOLD_STRONG
        elif exit_decision['action'] == 'ADD_MORE' and exit_decision['score'] >= 80:
            if overall_status in ['OK', 'GOOD']:
                overall_status = 'OPPORTUNITY'
                overall_action = 'ADD_MORE'
                alerts.append({
                    'priority': 'LOW',
                    'type': '🧠 SMART EXIT - STRONG HOLD',
                    'message': f"Score: {exit_decision['score']}/100 - All factors aligned",
                    'action': 'Consider adding to position',
                    'email_type': 'important'
                })
    
    # =========================================================================
    # UNIFIED DECISION LABEL (Single source of truth)
    # =========================================================================
    
    # Map the final overall_action to a human-readable unified label
    unified_action_labels = {
        'EXIT': '🚨 EXIT NOW',
        'EXIT_EARLY': '🚨 EXIT EARLY',
        'EXIT_TODAY': '⚠️ EXIT TODAY',
        'PARTIAL_EXIT': '📊 PARTIAL EXIT',
        'TIGHTEN_SL': '🔧 TIGHTEN SL',
        'BOOK_PROFITS': '💰 BOOK PROFITS',
        'HOLD_EXTEND': '💪 HOLD & EXTEND',
        'TRAIL_SL': '📈 TRAIL SL',
        'MOVE_SL_BREAKEVEN': '🔔 MOVE SL TO BREAKEVEN',
        'WATCH': '👀 WATCH CLOSELY',
        'ADD_MORE': '🚀 ADD MORE',
        'HOLD': '✅ HOLD',
    }
    
    unified_label = unified_action_labels.get(overall_action, f"ℹ️ {overall_action}")
    
    return {
        # Basic Info
        'ticker': ticker,
        'position_type': position_type,
        'entry_price': entry_price,
        'current_price': current_price,
        'quantity': quantity,
        'pnl_percent': pnl_percent,
        'pnl_amount': pnl_amount,
        'day_change': day_change,
        'day_high': day_high,
        'day_low': day_low,
        
        # Original Levels
        'stop_loss': stop_loss,
        'target1': target1,
        'target2': target2,
        
        # Technical Indicators
        'rsi': rsi,
        'macd_hist': macd_hist,
        'macd_signal': macd_signal,
        'stoch_k': stoch_k_val,
        'stoch_d': stoch_d_val,
        
        # Momentum
        'momentum_score': momentum_score,
        'momentum_trend': momentum_trend,
        'momentum_components': momentum_components,
        
        # Volume
        'volume_signal': volume_signal,
        'volume_ratio': volume_ratio,
        'volume_desc': volume_desc,
        'volume_trend': volume_trend,
        
        # Support/Resistance
        'support': sr_levels['nearest_support'],
        'resistance': sr_levels['nearest_resistance'],
        'distance_to_support': sr_levels['distance_to_support'],
        'distance_to_resistance': sr_levels['distance_to_resistance'],
        'support_strength': sr_levels['support_strength'],
        'resistance_strength': sr_levels['resistance_strength'],
        
        # SL Risk
        'sl_risk': sl_risk,
        'sl_reasons': sl_reasons,
        'sl_recommendation': sl_recommendation,
        'sl_priority': sl_priority,
        'distance_to_sl': distance_to_sl,
        'approaching_sl': approaching_sl,
        
        # Upside
        'upside_score': upside_score,
        'upside_reasons': upside_reasons,
        'new_target': new_target,
        
        # Dynamic Levels
        'trail_stop': dynamic_levels['trail_stop'],
        'should_trail': dynamic_levels['should_trail'],
        'trail_reason': dynamic_levels.get('trail_reason', ''),
        'trail_action': dynamic_levels.get('trail_action', ''),
        'dynamic_target1': dynamic_levels['target1'],
        'dynamic_target2': dynamic_levels['target2'],
        'atr': dynamic_levels['atr'],
        
        # Targets Status
        'target1_hit': target1_hit,
        'target2_hit': target2_hit,
        'sl_hit': sl_hit,
        'at_breakeven': at_breakeven,
        
        # Multi-Timeframe
        'mtf_signals': mtf_result['signals'],
        'mtf_details': mtf_result.get('details', {}),
        'mtf_alignment': mtf_result['alignment_score'],
        'mtf_recommendation': mtf_result['recommendation'],
        'mtf_trend_strength': mtf_result.get('trend_strength', 'UNKNOWN'),
        
        # Partial Exits
        'partial_exits': partial_exits,
        
        # Holding Period
        'holding_days': holding_days,
        'tax_implication': tax_implication,
        'tax_color': tax_color,
        
        # Risk-Reward
        'risk_reward_ratio': risk_reward_ratio,
        
        # Alerts & Status (UNIFIED - single source of truth)
        'alerts': alerts,
        'overall_status': overall_status,
        'overall_action': overall_action,
        'unified_label': unified_label,
        'exit_decision': exit_decision,
        
        # NEW: Smart SL Data
        'smart_sl_data': smart_sl_data,
        
        # NEW: Intraday SL Data
        'intraday_sl_data': intraday_sl_data,
        
        # Chart Data (excluded from main cache - fetched separately for charts)
        'df': None  # Set to None to reduce cache size
    }

# ============================================================================
# LOAD PORTFOLIO FROM GOOGLE SHEETS
# ============================================================================

def load_portfolio():
    """Load portfolio from Google Sheets using gspread"""
    
    if not HAS_GSPREAD:
        st.error("❌ gspread not installed. Run: pip install gspread oauth2client")
        return None
    
    try:
        # Use the service account connection (already defined in your code)
        sheet, status = get_google_sheet_connection()
        
        if sheet is None:
            st.error(f"❌ Connection failed: {status}")
            return None
        
        # Get all records from the sheet
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        
        if df.empty:
            st.warning("⚠️ No data found in sheet")
            return None
        
        # Filter active positions
        if 'Status' in df.columns:
            df = df[df['Status'].str.upper().isin(['ACTIVE', 'PENDING'])]
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Validate required columns
        required_cols = ['Ticker', 'Position', 'Entry_Price', 'Stop_Loss', 'Target_1']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.warning(f"⚠️ Missing columns: {missing_cols}")
            # Try alternative column names
            alt_names = {
                'Ticker': ['Symbol', 'Stock', 'Name'],
                'Position': ['Type', 'Side', 'Direction'],
                'Entry_Price': ['Entry', 'Buy_Price', 'Price'],
                'Stop_Loss': ['SL', 'Stoploss'],
                'Target_1': ['Target', 'T1', 'Target1']
            }
            for col, alts in alt_names.items():
                if col not in df.columns:
                    for alt in alts:
                        if alt in df.columns:
                            df[col] = df[alt]
                            break
        
        # Set defaults for optional columns
        if 'Quantity' not in df.columns:
            df['Quantity'] = 1
        if 'Target_2' not in df.columns:
            df['Target_2'] = df['Target_1'] * 1.1
        if 'Entry_Date' not in df.columns:
            df['Entry_Date'] = None
        
        # ✅ NEW: Ensure Realized_PnL column exists
        if 'Realized_PnL' not in df.columns:
            df['Realized_PnL'] = 0.0
        
        # ✅ NEW: Ensure Zerodha_Order_ID column exists
        if 'Zerodha_Order_ID' not in df.columns:
            df['Zerodha_Order_ID'] = ''
        
        st.success(f"✅ Loaded {len(df)} active positions from Google Sheets")
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error loading from Google Sheets: {e}")
        st.info("💡 Make sure the Google Sheet is set to 'Anyone with the link can view'")
        
        # Return sample data as fallback
        st.warning("⚠️ Using sample data as fallback")
        return pd.DataFrame({
            'Ticker': ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK'],
            'Position': ['LONG', 'LONG', 'SHORT', 'LONG', 'LONG'],
            'Entry_Price': [2450.00, 3580.00, 1520.00, 1650.00, 1050.00],
            'Quantity': [10, 5, 8, 12, 20],
            'Stop_Loss': [2380.00, 3480.00, 1580.00, 1600.00, 1010.00],
            'Target_1': [2550.00, 3720.00, 1420.00, 1750.00, 1120.00],
            'Target_2': [2650.00, 3850.00, 1350.00, 1850.00, 1180.00],
            'Entry_Date': ['2024-01-15', '2024-01-20', '2024-02-01', '2024-01-10', '2024-02-05'],
            'Status': ['ACTIVE', 'ACTIVE', 'ACTIVE', 'ACTIVE', 'ACTIVE'],
            'Realized_PnL': [0.0, 0.0, 0.0, 0.0, 0.0]
        })
# ============================================================================
# PORTFOLIO VALIDATION
# ============================================================================

def validate_portfolio(df):
    """
    Validate portfolio data and return errors
    Returns: (is_valid, errors_list)
    """
    errors = []
    warnings = []
    
    # Check required columns
    required_cols = ['Ticker', 'Position', 'Entry_Price', 'Stop_Loss', 'Target_1']
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"❌ Missing required column: {col}")
    
    if errors:
        return False, errors, warnings
    
    # Validate each row
    for idx, row in df.iterrows():
        ticker = str(row.get('Ticker', f'Row {idx}')).strip()
        
        try:
            entry = float(row['Entry_Price'])
            sl = float(row['Stop_Loss'])
            target = float(row['Target_1'])
            position = str(row['Position']).upper().strip()
            status = str(row['Status']).upper().strip()
        except (ValueError, TypeError) as e:
            errors.append(f"❌ {ticker}: Invalid number format - {e}")
            continue
        
        # Check positive values
        if entry <= 0:
            errors.append(f"❌ {ticker}: Entry price must be positive")
        if sl <= 0:
            errors.append(f"❌ {ticker}: Stop loss must be positive")
        if target <= 0:
            errors.append(f"❌ {ticker}: Target must be positive")
        
        # Check position type
        if position not in ['LONG', 'SHORT']:
            errors.append(f"❌ {ticker}: Position must be 'LONG' or 'SHORT', got '{position}'")
            continue
        
        # Validate levels based on position type
        # Validate levels based on position type (for ALL statuses)
        if position == 'LONG':
            if entry <= sl:
                errors.append(
                    f"❌ {ticker} (LONG): Entry (₹{entry}) must be > "
                    f"Stop Loss (₹{sl})"
                )
            if target <= entry:
                warnings.append(
                    f"⚠️ {ticker} (LONG): Target (₹{target}) should be > "
                    f"Entry (₹{entry})"
                )
        else:  # SHORT
            if entry >= sl:
                errors.append(
                    f"❌ {ticker} (SHORT): Entry (₹{entry}) must be < "
                    f"Stop Loss (₹{sl})"
                )
            if target >= entry:
                warnings.append(
                    f"⚠️ {ticker} (SHORT): Target (₹{target}) should be < "
                    f"Entry (₹{entry})"
                )
        
        # Check quantity if present
        if 'Quantity' in df.columns:
            try:
                qty = int(row['Quantity'])
                if qty <= 0:
                    errors.append(f"❌ {ticker}: Quantity must be positive")
            except:
                warnings.append(f"⚠️ {ticker}: Invalid quantity, using default (1)")
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings

# ============================================================================
# GOOGLE SHEETS UPDATE FUNCTIONS
# ============================================================================

def get_google_sheet_connection():
    try:
        if credentials is None:
            return None, "Google credentials not configured"
        gc = gspread.authorize(credentials)
        sh = gc.open("my_portfolio")
        return sh.sheet1, "success"
    except Exception as e:
        return None, str(e)

def get_persistent_guard(ticker, action_type):
    """
    Check if an auto-trade action was already performed today.
    Uses Google Sheets as persistent storage (survives page refresh).
    
    Args:
        ticker: Stock symbol
        action_type: "sl_trail", "market_sl", "auto_exit", "target_extend"
    
    Returns: (already_done, timestamp_string)
    """
    today = get_ist_now().strftime('%Y-%m-%d')
    guard_key = f"{action_type}_{ticker}_{today}"
    
    # Check session state first (fast path)
    if guard_key in st.session_state:
        return True, st.session_state[guard_key]
    
    # Check Google Sheet for persistent record
    try:
        sheet, status = get_google_sheet_connection()
        if sheet is None:
            return False, ""
        
        cell = sheet.find(ticker)
        if not cell:
            return False, ""
        
        row = cell.row
        headers = sheet.row_values(1)
        
        # Look for Last_Auto_Action column
        col_name = 'Last_Auto_Action'
        if col_name in headers:
            col_idx = headers.index(col_name) + 1
            cell_value = sheet.cell(row, col_idx).value
            
            if cell_value and today in str(cell_value):
                if action_type in str(cell_value):
                    st.session_state[guard_key] = cell_value
                    return True, cell_value
        
        return False, ""
    except Exception as e:
        logger.warning(f"Guard check failed: {e}")
        return False, ""


def set_persistent_guard(ticker, action_type, details=""):
    """
    Record that an auto-trade action was performed.
    Writes to both session state AND Google Sheet.
    """
    today = get_ist_now().strftime('%Y-%m-%d')
    now_str = get_ist_now().strftime('%Y-%m-%d %H:%M:%S')
    guard_key = f"{action_type}_{ticker}_{today}"
    guard_value = f"{action_type}|{now_str}|{details}"
    
    # Set session state
    st.session_state[guard_key] = guard_value
    
    # Set in Google Sheet (persistent)
    try:
        sheet, status = get_google_sheet_connection()
        if sheet is None:
            return
        
        cell = sheet.find(ticker)
        if not cell:
            return
        
        row = cell.row
        headers = sheet.row_values(1)
        
        col_name = 'Last_Auto_Action'
        if col_name in headers:
            col_idx = headers.index(col_name) + 1
        else:
            # Create column if it doesn't exist
            col_idx = len(headers) + 1
            sheet.update_cell(1, col_idx, col_name)
        
        sheet.update_cell(row, col_idx, guard_value)
        logger.info(f"Guard set: {guard_key} = {guard_value}")
    except Exception as e:
        logger.warning(f"Guard write failed: {e}")

def update_sheet_stop_loss(ticker, new_sl, reason, should_send_email=True, email_settings=None, result=None):
    """
    Update Stop Loss in Google Sheet and Zerodha.
    
    CRITICAL RULE: In LIVE mode, Sheet is ONLY updated if Zerodha succeeds.
    In PAPER mode, Sheet is updated directly (no Zerodha call).
    
    Returns: (success, message)
    """
    sheet, status = get_google_sheet_connection()
    
    if sheet is None:
        logger.warning(f"Cannot update sheet: {status}")
        return False, status
    
    try:
        # Find the row with this ticker
        cell = sheet.find(ticker)
        if not cell:
            return False, f"Ticker {ticker} not found in sheet"
        
        row = cell.row
        
        # Find Stop_Loss column
        headers = sheet.row_values(1)
        
        try:
            sl_col = headers.index('Stop_Loss') + 1
        except ValueError:
            sl_col = 4  # Default to column D
        
        # Get old SL value before updating
        old_sl = sheet.cell(row, sl_col).value
        
        # Round to NSE tick size
        new_sl = round_to_tick_size(new_sl)
        
        # ================================================================
        # ZERODHA INTEGRATION (MUST succeed before Sheet update)
        # ================================================================
        kite_mode = st.session_state.get('kite_mode', 'PAPER')
        zerodha_success = True  # Default True for PAPER mode
        zerodha_msg = "Paper mode - no Zerodha call"
        
        if (HAS_KITE and kite_session and kite_session.is_connected and
                kite_mode == 'LIVE'):
            
            # Find Zerodha_Order_ID from sheet
            zerodha_order_id = None
            try:
                if 'Zerodha_Order_ID' in headers:
                    order_id_col = headers.index('Zerodha_Order_ID') + 1
                    zerodha_order_id = sheet.cell(row, order_id_col).value
            except Exception as e:
                logger.warning(f"Could not read Zerodha_Order_ID: {e}")
            
            if zerodha_order_id and str(zerodha_order_id).strip():
                # Modify existing SL order on Zerodha
                zerodha_success, zerodha_msg = kite_session.modify_stoploss(
                    str(zerodha_order_id).strip(),
                    new_sl
                )
                
                if zerodha_success:
                    log_email(
                        f"🔗 Zerodha SL modified: {ticker} → ₹{new_sl:.2f}"
                    )
                else:
                    log_email(
                        f"❌ Zerodha SL modify FAILED: {ticker} - {zerodha_msg}"
                    )
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # CRITICAL: DO NOT update Sheet if Zerodha failed!
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    error_msg = (
                        f"⚠️ ZERODHA FAILED for {ticker}: {zerodha_msg}. "
                        f"Sheet NOT updated. "
                        f"Real SL is still ₹{old_sl} on Zerodha!"
                    )
                    logger.error(error_msg)
                    log_email(error_msg)
                    
                    # Send ALERT email about the failure
                    if should_send_email and email_settings and result:
                        sender = email_settings.get('sender_email')
                        password = email_settings.get('sender_password')
                        recipient = email_settings.get('recipient_email')
                        
                        if sender and password and recipient:
                            fail_subject = (
                                f"🚨 ZERODHA SL UPDATE FAILED - {ticker}"
                            )
                            fail_html = f"""
                            <html>
                            <body style="font-family: Arial; padding: 20px;">
                                <div style="background:#dc3545; color:white; 
                                            padding:20px; border-radius:10px; 
                                            text-align:center;">
                                    <h1>🚨 SL Update Failed!</h1>
                                    <p>{ticker}</p>
                                </div>
                                <div style="padding:20px;">
                                    <p><strong>What happened:</strong> 
                                       Bot tried to update SL from ₹{old_sl} to 
                                       ₹{new_sl:.2f} but Zerodha API failed.</p>
                                    <p><strong>Error:</strong> {zerodha_msg}</p>
                                    <p style="color:#dc3545; font-weight:bold;">
                                       ⚠️ Your REAL SL on Zerodha is still 
                                       ₹{old_sl}!
                                    </p>
                                    <p><strong>Action Required:</strong> 
                                       Manually update SL on Zerodha/Kite app.
                                    </p>
                                    <p>Time: {get_ist_now().strftime(
                                        '%Y-%m-%d %H:%M:%S'
                                    )} IST</p>
                                </div>
                            </body>
                            </html>
                            """
                            send_email_alert(
                                fail_subject, fail_html,
                                sender, password, recipient
                            )
                    
                    return False, error_msg
            
            else:
                # No Zerodha Order ID - just update sheet
                zerodha_msg = "No Zerodha_Order_ID found - sheet only update"
                log_email(f"ℹ️ {ticker}: {zerodha_msg}")
        
        # ================================================================
        # SHEET UPDATE (Only reaches here if Zerodha succeeded or PAPER mode)
        # ================================================================
        sheet.update_cell(row, sl_col, new_sl)
        
        # Build log message
        mode_tag = "🔗 LIVE" if kite_mode == 'LIVE' else "📝 PAPER"
        log_message = (
            f"🔄 {mode_tag} SL Updated {ticker}: "
            f"₹{old_sl} → ₹{new_sl:.2f} - {reason}"
        )
        
        if kite_mode == 'LIVE' and zerodha_success:
            log_message += " | ✅ Zerodha synced"
        
        logger.info(log_message)
        log_email(log_message)
        
        # Send email notification
        if should_send_email and email_settings and result:
            if email_settings.get('email_on_sl_change', True):
                sender = email_settings.get('sender_email')
                password = email_settings.get('sender_password')
                recipient = email_settings.get('recipient_email')
                
                if sender and password and recipient:
                    subject = f"🔄 Stop Loss Updated - {ticker}"
                    
                    sync_status = (
                        "✅ Zerodha Synced" if (
                            kite_mode == 'LIVE' and zerodha_success
                        )
                        else "📝 Paper Mode" if kite_mode == 'PAPER'
                        else "⚠️ Zerodha Not Connected"
                    )
                    
                    html_content = f"""
                    <html>
                    <body style="font-family: Arial, sans-serif; padding: 20px; 
                                 background: #f8f9fa;">
                        <div style="max-width: 600px; margin: 0 auto; 
                                    background: white; border-radius: 10px; 
                                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                            <div style="background: #17a2b8; color: white; 
                                        padding: 20px; text-align: center;">
                                <h1 style="margin: 0;">
                                    🔄 Stop Loss Updated
                                </h1>
                                <p style="margin: 10px 0 0 0; font-size: 1.2em;">
                                    {ticker}
                                </p>
                            </div>
                            <div style="padding: 20px;">
                                <div style="background: #e7f3ff; padding: 15px; 
                                            border-radius: 8px; margin-bottom: 20px; 
                                            border-left: 4px solid #17a2b8;">
                                    <p style="margin: 0; font-size: 1.1em;">
                                        <strong>Old Stop Loss:</strong> ₹{old_sl}
                                    </p>
                                    <p style="margin: 10px 0 0 0; font-size: 1.2em; 
                                              color: #17a2b8;">
                                        <strong>New Stop Loss:</strong> 
                                        ₹{new_sl:.2f}
                                    </p>
                                    <p style="margin: 10px 0 0 0;">
                                        <strong>Reason:</strong> {reason}
                                    </p>
                                    <p style="margin: 10px 0 0 0;">
                                        <strong>Zerodha:</strong> {sync_status}
                                    </p>
                                </div>
                                <table style="width: 100%; border-collapse: collapse;">
                                    <tr>
                                        <td style="padding: 10px; 
                                                   border-bottom: 1px solid #eee;">
                                            <strong>Current Price</strong>
                                        </td>
                                        <td style="padding: 10px; 
                                                   border-bottom: 1px solid #eee; 
                                                   text-align: right;">
                                            ₹{result['current_price']:,.2f}
                                        </td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 10px; 
                                                   border-bottom: 1px solid #eee;">
                                            <strong>Entry Price</strong>
                                        </td>
                                        <td style="padding: 10px; 
                                                   border-bottom: 1px solid #eee; 
                                                   text-align: right;">
                                            ₹{result['entry_price']:,.2f}
                                        </td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 10px; 
                                                   border-bottom: 1px solid #eee;">
                                            <strong>Current P&L</strong>
                                        </td>
                                        <td style="padding: 10px; 
                                                   border-bottom: 1px solid #eee; 
                                                   text-align: right; 
                                                   color: {'#28a745' if result['pnl_percent'] >= 0 else '#dc3545'};">
                                            {result['pnl_percent']:+.2f}% 
                                            (₹{result['pnl_amount']:+,.0f})
                                        </td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 10px; 
                                                   border-bottom: 1px solid #eee;">
                                            <strong>Mode</strong>
                                        </td>
                                        <td style="padding: 10px; 
                                                   border-bottom: 1px solid #eee; 
                                                   text-align: right;">
                                            {mode_tag}
                                        </td>
                                    </tr>
                                </table>
                            </div>
                            <div style="background: #f8f9fa; padding: 15px; 
                                        text-align: center; font-size: 0.9em; 
                                        color: #666;">
                                <p style="margin: 0;">
                                    Smart Portfolio Monitor - {sync_status}
                                </p>
                                <p style="margin: 5px 0 0 0;">
                                    {get_ist_now().strftime(
                                        '%Y-%m-%d %H:%M:%S'
                                    )} IST
                                </p>
                            </div>
                        </div>
                    </body>
                    </html>
                    """
                    
                    send_email_alert(
                        subject, html_content, sender, password, recipient
                    )
                    log_email(f"📧 Email sent for SL update: {ticker}")
        
        return True, log_message
    
    except Exception as e:
        error_msg = f"Error updating {ticker}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def update_sheet_target(ticker, new_target, target_num, reason, should_send_email=True, email_settings=None, result=None):
    """
    Update Target in Google Sheet and send email notification
    target_num: 1 or 2
    Returns: (success, message)
    """
    sheet, status = get_google_sheet_connection()
    
    if sheet is None:
        logger.warning(f"Cannot update sheet: {status}")
        return False, status
    
    try:
        # Find the row
        cell = sheet.find(ticker)
        if not cell:
            return False, f"Ticker {ticker} not found in sheet"
        
        row = cell.row
        
        # Find Target column
        headers = sheet.row_values(1)
        target_col_name = f'Target_{target_num}'
        
        try:
            target_col = headers.index(target_col_name) + 1
        except ValueError:
            target_col = 6 if target_num == 1 else 7  # Default columns
        
        # Get old target value before updating
        old_target = sheet.cell(row, target_col).value
        
        
        # ✅ Round to NSE tick size before updating
        new_target = round_to_tick_size(new_target)
                # ================================================================
        # ZERODHA: No direct target modification needed
        # (Targets are managed by our bot, not by Zerodha orders)
        # But we log it for tracking
        # ================================================================
        kite_mode = st.session_state.get('kite_mode', 'PAPER')
        
        if (HAS_KITE and kite_session and kite_session.is_connected and
                kite_mode == 'LIVE'):
            log_email(
                f"🔗 Target update for {ticker}: "
                f"₹{old_target} → ₹{new_target:.2f} "
                f"(Zerodha: No order modification needed - "
                f"exit managed by bot)"
            )
        

        # Update the cell
        sheet.update_cell(row, target_col, new_target)
        
        log_message = f"🎯 AUTO-UPDATED {ticker} Target {target_num}: ₹{old_target} → ₹{new_target:.2f} - {reason}"
        logger.info(log_message)
        log_email(log_message)
        
        # 🆕 SEND EMAIL NOTIFICATION
        if should_send_email and email_settings and result:
            if email_settings.get('email_on_target_change', True):
                sender = email_settings.get('sender_email')
                password = email_settings.get('sender_password')
                recipient = email_settings.get('recipient_email')
                
                if sender and password and recipient:
                    subject = f"🎯 Target Extended - {ticker}"
                    
                    html_content = f"""
                    <html>
                    <body style="font-family: Arial, sans-serif; padding: 20px; background: #f8f9fa;">
                        <div style="max-width: 600px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                            <div style="background: #28a745; color: white; padding: 20px; text-align: center;">
                                <h1 style="margin: 0;">🎯 Target Extended</h1>
                                <p style="margin: 10px 0 0 0; font-size: 1.2em;">{ticker}</p>
                            </div>
                            <div style="padding: 20px;">
                                <div style="background: #d4edda; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #28a745;">
                                    <p style="margin: 0; font-size: 1.1em;"><strong>Old Target {target_num}:</strong> ₹{old_target}</p>
                                    <p style="margin: 10px 0 0 0; font-size: 1.2em; color: #28a745;"><strong>New Target {target_num}:</strong> ₹{new_target:.2f}</p>
                                    <p style="margin: 10px 0 0 0;"><strong>Reason:</strong> {reason}</p>
                                </div>
                                <table style="width: 100%; border-collapse: collapse;">
                                    <tr>
                                        <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Current Price</strong></td>
                                        <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right;">₹{result['current_price']:,.2f}</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Upside Score</strong></td>
                                        <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right;">{result.get('upside_score', 0):.0f}/100</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Potential Gain</strong></td>
                                        <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right;">
                                            {((new_target - result['current_price']) / result['current_price'] * 100):+.2f}%
                                        </td>
                                    </tr>
                                </table>
                            </div>
                            <div style="background: #f8f9fa; padding: 15px; text-align: center; font-size: 0.9em; color: #666;">
                                <p style="margin: 0;">Smart Portfolio Monitor - Auto Update</p>
                                <p style="margin: 5px 0 0 0;">{get_ist_now().strftime('%Y-%m-%d %H:%M:%S')} IST</p>
                            </div>
                        </div>
                    </body>
                    </html>
                    """
                    
                    send_email_alert(subject, html_content, sender, password, recipient)
                    log_email(f"📧 Email sent for Target update: {ticker}")
        
        return True, log_message
    
    except Exception as e:
        error_msg = f"Error updating {ticker}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def mark_position_inactive(ticker, exit_price, pnl_amount, exit_reason, 
                            should_send_email=True, email_settings=None, 
                            result=None):
    """
    Mark position as INACTIVE, place exit order on Zerodha, and record P&L.
    
    CRITICAL RULE: In LIVE mode, Sheet is ONLY updated if Zerodha order succeeds.
    In PAPER mode, Sheet is updated directly.
    
    Returns: (success, message)
    """
    sheet, status = get_google_sheet_connection()
    
    if sheet is None:
        logger.warning(f"Cannot update sheet: {status}")
        return False, status
    
    try:
        # Find the row
        cell = sheet.find(ticker)
        if not cell:
            return False, f"Ticker {ticker} not found in sheet"
        
        row = cell.row
        headers = sheet.row_values(1)
        
        # ================================================================
        # ZERODHA EXIT ORDER (MUST succeed before Sheet update in LIVE mode)
        # ================================================================
        kite_mode = st.session_state.get('kite_mode', 'PAPER')
        zerodha_success = True  # Default True for PAPER mode
        zerodha_order_id = None
        zerodha_msg = "Paper mode"
        
        if (HAS_KITE and kite_session and kite_session.is_connected and
                kite_mode == 'LIVE'):
            
            # Determine transaction type
            if result and result.get('position_type') == 'LONG':
                txn_type = "SELL"
            else:
                txn_type = "BUY"
            
            qty = result.get('quantity', 0) if result else 0
            
            if qty > 0:
                # SAFETY: Verify position exists in Zerodha
                pos_exists, pos_msg = kite_session.verify_position_exists(
                    ticker, qty,
                    result.get('position_type', 'LONG') if result else 'LONG'
                )
                
                if pos_exists:
                    # SAFETY: Verify price before exit
                    live_price, price_msg = kite_session.get_live_price(
                        ticker.replace('.NS', '').replace('.BO', '')
                    )
                    
                    if live_price > 0:
                        # Check price is reasonable
                        price_diff = (
                            abs(live_price - exit_price) / exit_price * 100
                        )
                        
                        if price_diff > 5.0:
                            error_msg = (
                                f"⚠️ Price mismatch for {ticker}: "
                                f"Expected ₹{exit_price:.2f}, "
                                f"Live ₹{live_price:.2f} "
                                f"({price_diff:.1f}% diff). "
                                f"Sheet NOT updated. Manual exit required."
                            )
                            log_email(error_msg)
                            
                            # Send alert email
                            if (should_send_email and email_settings 
                                    and result):
                                sender = email_settings.get('sender_email')
                                password = email_settings.get(
                                    'sender_password'
                                )
                                recipient = email_settings.get(
                                    'recipient_email'
                                )
                                if sender and password and recipient:
                                    send_email_alert(
                                        f"🚨 EXIT FAILED - {ticker} "
                                        f"Price Mismatch",
                                        f"<h2>Price mismatch detected</h2>"
                                        f"<p>{error_msg}</p>",
                                        sender, password, recipient
                                    )
                            
                            return False, error_msg
                        
                        # Use live price for better execution
                        exit_price = live_price
                    
                    # Place exit order
                    zerodha_success, zerodha_msg, zerodha_order_id = (
                        kite_session.place_exit_order(
                            ticker=ticker,
                            quantity=qty,
                            transaction_type=txn_type,
                            order_type="MARKET",
                            product="CNC",
                            tag=f"Exit_{exit_reason[:15]}"
                        )
                    )
                    
                    if zerodha_success:
                        log_email(
                            f"🔗 Zerodha EXIT: {txn_type} {qty}x {ticker} "
                            f"| ID: {zerodha_order_id}"
                        )
                        
                        # Cancel pending SL orders
                        open_sl_orders = kite_session.get_open_sl_orders(
                            ticker
                        )
                        for sl_order in open_sl_orders:
                            cancel_id = sl_order.get('order_id')
                            if cancel_id:
                                kite_session.cancel_order(str(cancel_id))
                                log_email(
                                    f"🔗 Cancelled SL order: {cancel_id}"
                                )
                    else:
                        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        # CRITICAL: Zerodha failed - DO NOT update sheet!
                        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        error_msg = (
                            f"🚨 ZERODHA EXIT FAILED for {ticker}: "
                            f"{zerodha_msg}. "
                            f"Sheet NOT updated. Position still ACTIVE!"
                        )
                        logger.error(error_msg)
                        log_email(error_msg)
                        
                        # Send CRITICAL alert email
                        if (should_send_email and email_settings 
                                and result):
                            sender = email_settings.get('sender_email')
                            password = email_settings.get('sender_password')
                            recipient = email_settings.get('recipient_email')
                            
                            if sender and password and recipient:
                                fail_html = f"""
                                <html>
                                <body style="font-family: Arial; padding: 20px;">
                                    <div style="background:#dc3545; color:white; 
                                                padding:20px; border-radius:10px; 
                                                text-align:center;">
                                        <h1>🚨 EXIT ORDER FAILED!</h1>
                                        <p style="font-size:1.3em;">{ticker}</p>
                                    </div>
                                    <div style="padding:20px; background:#f8d7da; 
                                                margin:15px 0; border-radius:10px;">
                                        <p><strong>Error:</strong> 
                                           {zerodha_msg}</p>
                                        <p><strong>Attempted:</strong> 
                                           {txn_type} {qty} shares at 
                                           ₹{exit_price:.2f}</p>
                                        <p><strong>Reason:</strong> 
                                           {exit_reason}</p>
                                        <p style="color:#dc3545; 
                                                  font-weight:bold; 
                                                  font-size:1.2em;">
                                            ⚠️ Position is STILL ACTIVE! 
                                            Manual exit required!
                                        </p>
                                    </div>
                                    <div style="padding:15px; background:#fff3cd; 
                                                border-radius:10px;">
                                        <h3>What to do:</h3>
                                        <ol>
                                            <li>Open Kite App immediately</li>
                                            <li>Find {ticker} in your 
                                                positions</li>
                                            <li>Place exit order manually</li>
                                            <li>Update Google Sheet status 
                                                to INACTIVE</li>
                                        </ol>
                                    </div>
                                    <p style="color:#666; font-size:0.9em;">
                                        {get_ist_now().strftime(
                                            '%Y-%m-%d %H:%M:%S'
                                        )} IST
                                    </p>
                                </body>
                                </html>
                                """
                                
                                send_email_alert(
                                    f"🚨 CRITICAL: EXIT FAILED - {ticker}",
                                    fail_html,
                                    sender, password, recipient
                                )
                        
                        return False, error_msg
                
                else:
                    # Position not found in Zerodha
                    log_email(
                        f"⚠️ {ticker} not found in Zerodha: {pos_msg}. "
                        f"Updating sheet only."
                    )
                    zerodha_msg = f"Not in Zerodha: {pos_msg}"
            else:
                zerodha_msg = "Quantity is 0"
        
        # ================================================================
        # SHEET UPDATE (Only reaches here if Zerodha succeeded or PAPER)
        # ================================================================
        
        # Find Status column
        try:
            status_col = headers.index('Status') + 1
        except ValueError:
            status_col = 9
        
        # Find Realized_PnL column
        try:
            pnl_col = headers.index('Realized_PnL') + 1
        except ValueError:
            pnl_col = len(headers) + 1
            sheet.update_cell(1, pnl_col, 'Realized_PnL')
        
        # Find Exit_Date column
        try:
            exit_date_col = headers.index('Exit_Date') + 1
        except ValueError:
            exit_date_col = len(headers) + 2
            sheet.update_cell(1, exit_date_col, 'Exit_Date')
        
        # Find Exit_Reason column
        try:
            exit_reason_col = headers.index('Exit_Reason') + 1
        except ValueError:
            exit_reason_col = len(headers) + 3
            sheet.update_cell(1, exit_reason_col, 'Exit_Reason')
        
        # Update all cells
        sheet.update_cell(row, status_col, 'INACTIVE')
        sheet.update_cell(row, pnl_col, pnl_amount)
        
        exit_date = get_ist_now().strftime('%Y-%m-%d')
        sheet.update_cell(row, exit_date_col, exit_date)
        sheet.update_cell(row, exit_reason_col, exit_reason)
        
        # Build log message
        mode_tag = "🔗 LIVE" if kite_mode == 'LIVE' else "📝 PAPER"
        log_message = (
            f"🚪 {mode_tag} CLOSED: {ticker} | "
            f"Exit: ₹{exit_price:.2f} | "
            f"P&L: ₹{pnl_amount:+,.0f} | "
            f"Reason: {exit_reason}"
        )
        
        if kite_mode == 'LIVE' and zerodha_order_id:
            log_message += f" | Zerodha ID: {zerodha_order_id}"
        
        logger.info(log_message)
        log_email(log_message)
        
        # Send email notification
        if should_send_email and email_settings and result:
            sender = email_settings.get('sender_email')
            password = email_settings.get('sender_password')
            recipient = email_settings.get('recipient_email')
            
            if sender and password and recipient:
                if pnl_amount > 0:
                    header_color = "#28a745"
                    status_emoji = "✅"
                else:
                    header_color = "#dc3545"
                    status_emoji = "❌"
                
                sync_status = (
                    f"✅ Zerodha Order: {zerodha_order_id}" 
                    if (kite_mode == 'LIVE' and zerodha_order_id)
                    else "📝 Paper Mode" if kite_mode == 'PAPER'
                    else f"ℹ️ {zerodha_msg}"
                )
                
                subject = (
                    f"{status_emoji} Position Closed - {ticker} | "
                    f"P&L: ₹{pnl_amount:+,.0f}"
                )
                
                html_content = f"""
                <html>
                <body style="font-family: Arial, sans-serif; padding: 20px; 
                             background: #f8f9fa;">
                    <div style="max-width: 600px; margin: 0 auto; 
                                background: white; border-radius: 10px; 
                                box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                        <div style="background: {header_color}; color: white; 
                                    padding: 20px; text-align: center;">
                            <h1 style="margin: 0;">
                                {status_emoji} Position Closed
                            </h1>
                            <p style="margin: 10px 0 0 0; font-size: 1.2em;">
                                {ticker}
                            </p>
                        </div>
                        <div style="padding: 20px;">
                            <div style="background: {'#d4edda' if pnl_amount > 0 else '#f8d7da'}; 
                                        padding: 15px; border-radius: 8px; 
                                        margin-bottom: 20px; 
                                        border-left: 4px solid {header_color};">
                                <p style="margin: 0; font-size: 1.3em; 
                                          color: {header_color};">
                                    <strong>P&L: ₹{pnl_amount:+,.0f}</strong>
                                </p>
                                <p style="margin: 10px 0 0 0;">
                                    <strong>Reason:</strong> {exit_reason}
                                </p>
                                <p style="margin: 10px 0 0 0;">
                                    <strong>Broker:</strong> {sync_status}
                                </p>
                            </div>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr>
                                    <td style="padding: 10px; 
                                               border-bottom: 1px solid #eee;">
                                        <strong>Position</strong>
                                    </td>
                                    <td style="padding: 10px; 
                                               border-bottom: 1px solid #eee; 
                                               text-align: right;">
                                        {'📈 LONG' if result.get('position_type') == 'LONG' else '📉 SHORT'}
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding: 10px; 
                                               border-bottom: 1px solid #eee;">
                                        <strong>Entry Price</strong>
                                    </td>
                                    <td style="padding: 10px; 
                                               border-bottom: 1px solid #eee; 
                                               text-align: right;">
                                        ₹{result.get('entry_price', 0):,.2f}
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding: 10px; 
                                               border-bottom: 1px solid #eee;">
                                        <strong>Exit Price</strong>
                                    </td>
                                    <td style="padding: 10px; 
                                               border-bottom: 1px solid #eee; 
                                               text-align: right;">
                                        ₹{exit_price:,.2f}
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding: 10px; 
                                               border-bottom: 1px solid #eee;">
                                        <strong>Quantity</strong>
                                    </td>
                                    <td style="padding: 10px; 
                                               border-bottom: 1px solid #eee; 
                                               text-align: right;">
                                        {result.get('quantity', 0)} shares
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding: 10px; 
                                               border-bottom: 1px solid #eee;">
                                        <strong>P&L %</strong>
                                    </td>
                                    <td style="padding: 10px; 
                                               border-bottom: 1px solid #eee; 
                                               text-align: right; 
                                               color: {header_color}; 
                                               font-weight: bold;">
                                        {result.get('pnl_percent', 0):+.2f}%
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding: 10px; 
                                               border-bottom: 1px solid #eee;">
                                        <strong>Holding Period</strong>
                                    </td>
                                    <td style="padding: 10px; 
                                               border-bottom: 1px solid #eee; 
                                               text-align: right;">
                                        {result.get('holding_days', 0)} days
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding: 10px; 
                                               border-bottom: 1px solid #eee;">
                                        <strong>Mode</strong>
                                    </td>
                                    <td style="padding: 10px; 
                                               border-bottom: 1px solid #eee; 
                                               text-align: right;">
                                        {mode_tag}
                                    </td>
                                </tr>
                            </table>
                        </div>
                        <div style="background: #f8f9fa; padding: 15px; 
                                    text-align: center; font-size: 0.9em; 
                                    color: #666;">
                            <p style="margin: 0;">
                                Smart Portfolio Monitor - Position Closed
                            </p>
                            <p style="margin: 5px 0 0 0;">
                                {get_ist_now().strftime(
                                    '%Y-%m-%d %H:%M:%S'
                                )} IST
                            </p>
                        </div>
                    </div>
                </body>
                </html>
                """
                
                send_email_alert(
                    subject, html_content, sender, password, recipient
                )
                log_email(f"📧 Email sent for position closure: {ticker}")
        
        return True, log_message
    
    except Exception as e:
        error_msg = f"Error marking {ticker} inactive: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
    

# ============================================================================
# EMAIL ALERT FUNCTIONS
# ============================================================================

def should_send_alert(alert, email_settings, result):
    """
    Determine if email should be sent for this alert.
    RENAMED from should_send_email to avoid name collision with variables.
    """
    email_type = alert.get('email_type', 'important')
    
    if email_type == 'critical' and email_settings.get('email_on_critical', True):
        return True
    elif email_type == 'target' and email_settings.get('email_on_target', True):
        return True
    elif email_type == 'sl_approach' and email_settings.get('email_on_sl_approach', True):
        return True
    elif email_type == 'sl_change' and email_settings.get('email_on_sl_change', True):
        return True
    elif email_type == 'target_change' and email_settings.get('email_on_target_change', True):
        return True
    elif email_type == 'important' and email_settings.get('email_on_important', True):
        return True
    
    return False

def create_alert_email_html(result, alert):
    """
    Create HTML content for alert email
    """
    status_colors = {
        'CRITICAL': '#dc3545',
        'HIGH': '#ffc107',
        'MEDIUM': '#17a2b8',
        'LOW': '#28a745'
    }
    
    priority_color = status_colors.get(alert['priority'], '#6c757d')
    pnl_color = '#28a745' if result['pnl_percent'] >= 0 else '#dc3545'
    
    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; padding: 20px; background: #f8f9fa;">
        <div style="max-width: 600px; margin: 0 auto; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            
            <!-- Header -->
            <div style="background: {priority_color}; color: white; padding: 20px; text-align: center;">
                <h1 style="margin: 0;">{alert['type']}</h1>
                <p style="margin: 10px 0 0 0; font-size: 1.2em;">{result['ticker']}</p>
            </div>
            
            <!-- Content -->
            <div style="padding: 20px;">
                
                <!-- Alert Message -->
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                    <p style="margin: 0; font-size: 1.1em;"><strong>Message:</strong> {alert['message']}</p>
                    <p style="margin: 10px 0 0 0; font-size: 1.2em; color: {priority_color};"><strong>Action:</strong> {alert['action']}</p>
                </div>
                
                <!-- Position Details -->
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Position Type</strong></td>
                        <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right;">{'📈 LONG' if result['position_type'] == 'LONG' else '📉 SHORT'}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Entry Price</strong></td>
                        <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right;">₹{result['entry_price']:,.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Current Price</strong></td>
                        <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right;">₹{result['current_price']:,.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Stop Loss</strong></td>
                        <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right;">₹{result['stop_loss']:,.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>P&L</strong></td>
                        <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right; color: {pnl_color}; font-weight: bold;">
                            {result['pnl_percent']:+.2f}% (₹{result['pnl_amount']:+,.0f})
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>SL Risk Score</strong></td>
                        <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right;">{result['sl_risk']}%</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Final Decision</strong></td>
                        <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right; font-weight: bold;">
                            {result.get('unified_label', result.get('overall_action', 'N/A'))}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Quantity</strong></td>
                        <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right;">{result['quantity']} shares</td>
                    </tr>
                </table>
                
                <!-- Technical Summary -->
                <div style="margin-top: 20px; padding: 15px; background: #e9ecef; border-radius: 8px;">
                    <h3 style="margin: 0 0 10px 0;">Technical Summary</h3>
                    <p style="margin: 5px 0;">RSI: {result['rsi']:.1f} | MACD: {result['macd_signal']} | Momentum: {result['momentum_score']:.0f}/100</p>
                    <p style="margin: 5px 0;">Volume: {result['volume_signal'].replace('_', ' ')} ({result['volume_ratio']:.1f}x)</p>
                    <p style="margin: 5px 0;">Support: ₹{result['support']:,.2f} | Resistance: ₹{result['resistance']:,.2f}</p>
                </div>
                
            </div>
            
            <!-- Footer -->
            <div style="background: #f8f9fa; padding: 15px; text-align: center; font-size: 0.9em; color: #666;">
                <p style="margin: 0;">Smart Portfolio Monitor v6.0</p>
                <p style="margin: 5px 0 0 0;">{get_ist_now().strftime('%Y-%m-%d %H:%M:%S')} IST</p>
            </div>
            
        </div>
    </body>
    </html>
    """
    
    return html

def create_summary_email_html(results, critical_count, warning_count, portfolio_risk):
    """
    Create HTML content for summary email
    """
    ist_now = get_ist_now()
    
    # Build critical alerts section
    critical_html = ""
    for r in results:
        if r['overall_status'] == 'CRITICAL':
            critical_html += f"""
            <div style="background:#f8d7da; padding:15px; margin:10px 0; border-radius:8px; border-left:4px solid #dc3545;">
                <h3 style="margin:0; color:#721c24;">{r['ticker']} - {r['overall_action'].replace('_', ' ')}</h3>
                <p style="margin:5px 0;">Position: {r['position_type']} | P&L: {r['pnl_percent']:+.2f}%</p>
                <p style="margin:5px 0;">SL Risk: {r['sl_risk']}% | Current: ₹{r['current_price']:,.2f}</p>
                <p style="margin:5px 0; font-weight:bold;">⚡ {r['alerts'][0]['action'] if r['alerts'] else 'Review immediately'}</p>
            </div>
            """
    
    # Build warning alerts section
    warning_html = ""
    for r in results:
        if r['overall_status'] == 'WARNING':
            warning_html += f"""
            <div style="background:#fff3cd; padding:15px; margin:10px 0; border-radius:8px; border-left:4px solid #ffc107;">
                <h3 style="margin:0; color:#856404;">{r['ticker']} - {r['overall_action'].replace('_', ' ')}</h3>
                <p style="margin:5px 0;">Position: {r['position_type']} | P&L: {r['pnl_percent']:+.2f}%</p>
                <p style="margin:5px 0;">SL Risk: {r['sl_risk']}%</p>
            </div>
            """
    
    total_pnl = sum(r['pnl_amount'] for r in results)
    pnl_color = '#28a745' if total_pnl >= 0 else '#dc3545'
    
    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; padding: 20px; background: #f8f9fa;">
        <div style="max-width: 600px; margin: 0 auto; background: white; border-radius: 10px; overflow: hidden;">
            
            <!-- Header -->
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center;">
                <h1 style="margin: 0;">📊 Portfolio Alert Summary</h1>
                <p style="margin: 10px 0 0 0;">{ist_now.strftime('%Y-%m-%d %H:%M:%S')} IST</p>
            </div>
            
            <!-- Summary Stats -->
            <div style="padding: 20px; display: flex; justify-content: space-around; background: #f8f9fa;">
                <div style="text-align: center;">
                    <h2 style="margin: 0; color: #dc3545;">{critical_count}</h2>
                    <p style="margin: 5px 0;">Critical</p>
                </div>
                <div style="text-align: center;">
                    <h2 style="margin: 0; color: #ffc107;">{warning_count}</h2>
                    <p style="margin: 5px 0;">Warning</p>
                </div>
                <div style="text-align: center;">
                    <h2 style="margin: 0; color: {pnl_color};">₹{total_pnl:+,.0f}</h2>
                    <p style="margin: 5px 0;">Total P&L</p>
                </div>
            </div>
            
            <!-- Portfolio Risk -->
            <div style="padding: 15px 20px; background: {portfolio_risk['risk_color']}20; border-left: 4px solid {portfolio_risk['risk_color']};">
                <p style="margin: 0;"><strong>Portfolio Risk:</strong> {portfolio_risk['risk_icon']} {portfolio_risk['risk_status']} ({portfolio_risk['portfolio_risk_pct']:.1f}%)</p>
            </div>
            
            <!-- Critical Alerts -->
            {f'<div style="padding: 20px;"><h2 style="color: #dc3545;">🚨 Critical Alerts</h2>{critical_html}</div>' if critical_html else ''}
            
            <!-- Warning Alerts -->
            {f'<div style="padding: 20px;"><h2 style="color: #ffc107;">⚠️ Warnings</h2>{warning_html}</div>' if warning_html else ''}
            
            <!-- Footer -->
            <div style="background: #f8f9fa; padding: 15px; text-align: center; font-size: 0.9em; color: #666;">
                <p style="margin: 0;">Smart Portfolio Monitor v6.0</p>
            </div>
            
        </div>
    </body>
    </html>
    """
    
    return html

def send_portfolio_alerts(results, email_settings, portfolio_risk):
    """
    Send email alerts for portfolio positions
    """
    if not email_settings.get('enabled', False):
        return
    
    sender = email_settings.get('sender_email', '')
    password = email_settings.get('sender_password', '')
    recipient = email_settings.get('recipient_email', '')
    cooldown = email_settings.get('cooldown', 15)
    
    if not sender or not password or not recipient:
        return
    
    # Count alerts
    critical_count = sum(1 for r in results if r['overall_status'] == 'CRITICAL')
    warning_count = sum(1 for r in results if r['overall_status'] == 'WARNING')
    
    # Send summary email for critical alerts
    if critical_count > 0:
        alert_hash = generate_alert_hash("PORTFOLIO", "SUMMARY_CRITICAL", str(critical_count))
        
        if can_send_email(alert_hash, cooldown):
            subject = f"🚨 CRITICAL: {critical_count} positions need attention!"
            html = create_summary_email_html(results, critical_count, warning_count, portfolio_risk)
            
            success, msg = send_email_alert(subject, html, sender, password, recipient)
            if success:
                mark_email_sent(alert_hash)
                log_email(f"Summary email sent: {critical_count} critical, {warning_count} warning")
            else:
                log_email(f"Summary email failed: {msg}")
    
    # Send individual alerts for specific conditions
    for result in results:
        for alert in result['alerts']:
            if should_send_alert(alert, email_settings, result):
                alert_hash = generate_alert_hash(result['ticker'], alert['type'], str(result['current_price']))
                
                if can_send_email(alert_hash, cooldown):
                    subject = f"{alert['type']} - {result['ticker']}"
                    html = create_alert_email_html(result, alert)
                    
                    success, msg = send_email_alert(subject, html, sender, password, recipient)
                    if success:
                        mark_email_sent(alert_hash)
                        log_email(f"Alert sent: {result['ticker']} - {alert['type']}")
                    else:
                        log_email(f"Alert failed for {result['ticker']}: {msg}")
# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

def render_sidebar():
    """
    Render the sidebar with all settings and calculators
    Returns: dictionary with all settings
    """
    
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # =====================================================================
        # EMAIL CONFIGURATION
        # =====================================================================
        st.markdown("### 📧 Email Alerts")
        
        # ╔═══════════════════════════════════════════════════════════════╗
        # ║  CREDENTIALS: Read from environment variables (SECURE)       ║
        # ║  Set these in your terminal or Streamlit secrets:            ║
        # ║    export SMTP_EMAIL="your-email@gmail.com"                  ║
        # ║    export SMTP_APP_PASSWORD="xxxx xxxx xxxx xxxx"            ║
        # ║    export SMTP_RECIPIENT="recipient@gmail.com"               ║
        # ╚═══════════════════════════════════════════════════════════════╝
        YOUR_EMAIL = os.environ.get("SMTP_EMAIL", "")
        YOUR_APP_PASSWORD = os.environ.get("SMTP_APP_PASSWORD", "")
        YOUR_RECIPIENT = os.environ.get(
            "SMTP_RECIPIENT",
            os.environ.get("SMTP_EMAIL", "")
        )
        
        # Check if credentials are configured
        credentials_configured = bool(
            YOUR_EMAIL and
            YOUR_APP_PASSWORD and
            "@" in YOUR_EMAIL and
            YOUR_EMAIL != "your-email@gmail.com" and
            YOUR_APP_PASSWORD != "xxxx xxxx xxxx xxxx"
        )
        
        # ✅ Initialize email state ONCE (stays OFF until you enable it)
        if 'email_alerts_enabled' not in st.session_state:
            st.session_state.email_alerts_enabled = False  # ✅ Default OFF

        email_enabled = st.checkbox(
            "Enable Email Alerts",
            value=st.session_state.email_alerts_enabled,  # ✅ Remember user's choice
            key="email_enabled_checkbox",
            help="Enable/disable all email notifications"
        )

        # ✅ Update session state when checkbox changes
        if email_enabled != st.session_state.email_alerts_enabled:
            st.session_state.email_alerts_enabled = email_enabled
            if email_enabled:
                st.success("✅ Email alerts ENABLED")
                log_email("Email alerts enabled by user")
            else:
                st.info("📧 Email alerts DISABLED")
                log_email("Email alerts disabled by user")
        
        # Email settings dictionary
        email_settings = {
            'enabled': False,
            'sender_email': '',
            'sender_password': '',
            'recipient_email': '',
            'email_on_critical': True,
            'email_on_target': True,
            'email_on_sl_approach': True,
            'email_on_sl_change': True,
            'email_on_target_change': True,
            'email_on_important': True,
            'cooldown': 15
        }
        
        if email_enabled:
            if credentials_configured:
                email_settings['enabled'] = True
                email_settings['sender_email'] = YOUR_EMAIL
                email_settings['sender_password'] = YOUR_APP_PASSWORD
                email_settings['recipient_email'] = YOUR_RECIPIENT if YOUR_RECIPIENT else YOUR_EMAIL
                
                st.success("✅ Email auto-configured!")
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"📤 From: {YOUR_EMAIL[:3]}***@gmail.com")
                with col2:
                    st.caption(f"📥 To: {email_settings['recipient_email'][:3]}***@gmail.com")
                
                # Test email button
                if st.button("📧 Send Test Email", type="secondary", use_container_width=True):
                    test_subject = "🧪 Test Email - Smart Portfolio Monitor"
                    test_html = f"""
                    <html>
                    <body style="font-family: Arial, sans-serif; padding: 20px;">
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    color: white; padding: 20px; border-radius: 10px; text-align: center;">
                            <h1>✅ Test Email Successful!</h1>
                            <p>Your email configuration is working correctly.</p>
                            <p>Time: {get_ist_now().strftime('%Y-%m-%d %H:%M:%S')} IST</p>
                        </div>
                        <div style="padding: 20px; background: #f8f9fa; margin-top: 15px; border-radius: 10px;">
                            <p>You will receive alerts for:</p>
                            <ul>
                                <li>🔴 Critical alerts (SL hit, high risk)</li>
                                <li>🎯 Target achieved</li>
                                <li>⚠️ Approaching stop loss</li>
                                <li>🔄 Trail SL recommendations</li>
                                <li>📈 New target suggestions</li>
                            </ul>
                        </div>
                    </body>
                    </html>
                    """
                    success, msg = send_email_alert(
                        test_subject, test_html,
                        email_settings['sender_email'],
                        email_settings['sender_password'],
                        email_settings['recipient_email']
                    )
                    if success:
                        st.success("✅ Test email sent! Check your inbox.")
                        log_email(f"Test email sent to {email_settings['recipient_email']}")
                    else:
                        st.error(f"❌ Failed: {msg}")
                        log_email(f"Test email FAILED: {msg}")
            else:
                st.warning("⚠️ Configure credentials in code OR enter manually:")
                email_settings['sender_email'] = st.text_input("Your Gmail", placeholder="you@gmail.com")
                email_settings['sender_password'] = st.text_input(
                    "App Password", type="password",
                    help="16-character Gmail App Password"
                )
                email_settings['recipient_email'] = st.text_input(
                    "Send Alerts To", placeholder="recipient@gmail.com"
                )
                
                if email_settings['sender_email'] and email_settings['sender_password']:
                    email_settings['enabled'] = True
                    if not email_settings['recipient_email']:
                        email_settings['recipient_email'] = email_settings['sender_email']
                
                st.info("""
                **To auto-enable emails:**
                1. Open this Python file
                2. Find `YOUR_APP_PASSWORD = "xxxx xxxx xxxx xxxx"`
                3. Replace with your actual App Password
                4. Save and restart the app
                """)
            
            st.divider()
            
            # Alert Types
            st.markdown("#### 📬 Alert Types")
            col1, col2 = st.columns(2)
            with col1:
                email_settings['email_on_critical'] = st.checkbox("🔴 Critical", value=True)
                email_settings['email_on_target'] = st.checkbox("🎯 Target Hit", value=True)
                email_settings['email_on_sl_approach'] = st.checkbox("⚠️ Near SL", value=True)
            with col2:
                email_settings['email_on_sl_change'] = st.checkbox("🔄 Trail SL", value=True)
                email_settings['email_on_target_change'] = st.checkbox("📈 New Target", value=True)
                email_settings['email_on_important'] = st.checkbox("📋 Important", value=True)
            
            email_settings['cooldown'] = st.slider("⏱️ Cooldown (min)", 5, 60, 15)
            
            # Status display
            if email_settings['enabled']:
                enabled_count = sum([
                    email_settings['email_on_critical'],
                    email_settings['email_on_target'],
                    email_settings['email_on_sl_approach'],
                    email_settings['email_on_sl_change'],
                    email_settings['email_on_target_change'],
                    email_settings['email_on_important']
                ])
                st.markdown(f"""
                <div style='background:linear-gradient(135deg, #28a745, #218838); 
                            color:white; padding:10px; border-radius:8px; text-align:center; margin-top:10px;'>
                    📧 <strong>ACTIVE</strong> | {enabled_count}/6 alerts ON
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("☑️ Check 'Enable Email Alerts' above to activate")
        
        st.divider()
        
        # =====================================================================
        # ZERODHA KITE INTEGRATION
        # =====================================================================
        st.markdown("### 🔗 Zerodha Integration")
        
        if not HAS_KITE:
            st.warning(
                "⚠️ Kite not installed.\n"
                "`pip install kiteconnect`\n"
                "Then create `config_kite.py` and `kite_connector.py`"
            )
            kite_enabled = False
            kite_mode = "PAPER"
        else:
            if not kite_is_configured():
                st.error(
                    "❌ API credentials not set.\n"
                    "Edit `config_kite.py` and fill in your API key/secret."
                )
                kite_enabled = False
                kite_mode = "PAPER"
            else:
                # Mode selector
                kite_mode = st.radio(
                    "Trading Mode",
                    ["PAPER", "LIVE"],
                    index=0,
                    help=(
                        "PAPER: Simulates orders (safe for testing)\n"
                        "LIVE: Places REAL orders on Zerodha"
                    ),
                    key="kite_mode_radio"
                )
                
                if kite_mode == "LIVE":
                    st.warning("⚠️ LIVE MODE - Real orders will be placed!")
                
                # Connection status
                token, is_valid, token_msg = load_token()
                
                if is_valid:
                    st.success(f"✅ {token_msg}")
                    kite_enabled = True
                    
                    # Initialize session if not done
                    if not kite_session.is_connected:
                        with st.spinner("Connecting to Zerodha..."):
                            success, msg = kite_session.initialize()
                            if success:
                                st.success(f"🟢 {msg}")
                                st.session_state.kite_connected = True
                            else:
                                st.error(f"🔴 {msg}")
                                kite_enabled = False
                    
                    if st.button("🔓 Logout", key="kite_logout"):
                        kite_session.logout()
                        st.session_state.kite_connected = False
                        st.success("Logged out")
                        st.rerun()
                else:
                    kite_enabled = False
                    st.warning(f"🔑 {token_msg}")
                    
                    # Login flow
                    login_url, url_status = get_login_url()
                    
                    if login_url:
                        st.markdown(
                            f"**Step 1:** [Click here to login to Kite]({login_url})"
                        )
                        st.caption(
                            "After login, you'll be redirected. "
                            "Copy the `request_token` from the URL."
                        )
                        
                        request_token = st.text_input(
                            "Step 2: Paste Request Token",
                            type="password",
                            key="kite_request_token",
                            help="Found in the redirect URL after ?request_token="
                        )
                        
                        if st.button(
                            "🔑 Connect",
                            key="kite_connect_btn",
                            use_container_width=True
                        ):
                            if request_token:
                                with st.spinner("Authenticating..."):
                                    success, msg = (
                                        kite_session.login_with_request_token(
                                            request_token
                                        )
                                    )
                                    if success:
                                        st.success(f"✅ {msg}")
                                        st.session_state.kite_connected = True
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error(f"❌ {msg}")
                            else:
                                st.error("Paste the request token first")
                    else:
                        st.error(f"Cannot generate login URL: {url_status}")
                
                # Show order log
                if kite_session and kite_session.order_log:
                    with st.expander(
                        f"📋 Order Log ({len(kite_session.order_log)})",
                        expanded=False
                    ):
                        for order in reversed(kite_session.order_log[-5:]):
                            st.caption(
                                f"{order.get('timestamp', '')[:19]} | "
                                f"{order.get('ticker', '')} | "
                                f"{order.get('type', '')} | "
                                f"ID: {order.get('order_id', 'N/A')}"
                            )
        
        st.session_state.kite_mode = kite_mode
        
        st.divider()
        

        # =====================================================================
        # AUTO-REFRESH
        # =====================================================================
        st.markdown("### 🔄 Auto-Refresh")
        auto_refresh = st.checkbox("Enable Auto-Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (seconds)", 30, 300, 60)
        
        if not HAS_AUTOREFRESH:
            st.warning("⚠️ Install streamlit-autorefresh:\n`pip install streamlit-autorefresh`")
        
        st.divider()
        
        # =====================================================================
        # ALERT THRESHOLDS
        # =====================================================================
        st.markdown("### 🎯 Alert Thresholds")
        loss_threshold = st.slider("Alert on Loss %", -10.0, 0.0, -2.0, step=0.5)
        profit_threshold = st.slider("Alert on Profit %", 0.0, 20.0, 5.0, step=0.5)
        trail_sl_trigger = st.slider("Trail SL after Profit %", 0.5, 10.0, 2.0, step=0.5)
        sl_risk_threshold = st.slider("SL Risk Alert Threshold", 30, 90, 50)
        sl_approach_threshold = st.slider("SL Approach Warning %", 1.0, 5.0, 2.0, step=0.5)
        
        st.divider()
        
        # =====================================================================
        # ANALYSIS SETTINGS
        # =====================================================================
        st.markdown("### 📊 Analysis Settings")
        enable_volume_analysis = st.checkbox("Volume Confirmation", value=True)
        enable_sr_detection = st.checkbox("Support/Resistance", value=True)
        enable_multi_timeframe = st.checkbox("Multi-Timeframe Analysis", value=True)
        enable_correlation = st.checkbox("Correlation Analysis", value=False,
                                        help="May slow down loading")
        
        st.divider()
        
        # =====================================================================
        # POSITION SIZING CALCULATOR
        # =====================================================================
        st.markdown("### 💰 Position Sizing Calculator")
        
        with st.expander("Calculate Optimal Position Size", expanded=False):
            st.markdown("**Based on Risk Percentage**")
            
            calc_capital = st.number_input(
                "Total Capital (₹)",
                min_value=10000.0,
                value=100000.0,
                step=10000.0,
                key="pos_calc_capital"
            )
            
            calc_risk_pct = st.slider(
                "Risk per Trade (%)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                key="pos_calc_risk"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                calc_entry = st.number_input(
                    "Entry Price (₹)",
                    min_value=1.0,
                    value=1500.0,
                    step=10.0,
                    key="pos_calc_entry"
                )
            with col2:
                calc_sl = st.number_input(
                    "Stop Loss (₹)",
                    min_value=1.0,
                    value=1450.0,
                    step=10.0,
                    key="pos_calc_sl"
                )
            
            if calc_entry > calc_sl:
                risk_amount = calc_capital * (calc_risk_pct / 100)
                risk_per_share = abs(calc_entry - calc_sl)
                position_size = int(risk_amount / risk_per_share)
                investment = position_size * calc_entry
                investment_pct = (investment / calc_capital) * 100
                
                st.success(f"**Buy {position_size} shares**")
                st.info(f"Investment: ₹{investment:,.0f} ({investment_pct:.1f}% of capital)")
                st.caption(f"Risk Amount: ₹{risk_amount:,.0f} | Risk/Share: ₹{risk_per_share:.2f}")
                
                # Additional info
                if investment_pct > 30:
                    st.warning("⚠️ Position is > 30% of capital. Consider reducing.")
            elif calc_entry < calc_sl:
                st.error("For LONG: Entry must be > Stop Loss")
                st.info("For SHORT: Use Entry < SL calculator below")
            else:
                st.info("Enter different Entry and Stop Loss prices")
        
        st.divider()
        
        # =====================================================================
        # RISK-REWARD CALCULATOR
        # =====================================================================
        st.markdown("### ⚖️ Risk-Reward Calculator")
        
        with st.expander("Check Trade Quality", expanded=False):
            rr_col1, rr_col2 = st.columns(2)
            
            with rr_col1:
                rr_entry = st.number_input("Entry (₹)", min_value=1.0, value=1500.0, step=10.0, key="rr_entry")
                rr_sl = st.number_input("SL (₹)", min_value=1.0, value=1450.0, step=10.0, key="rr_sl")
            
            with rr_col2:
                rr_target = st.number_input("Target (₹)", min_value=1.0, value=1600.0, step=10.0, key="rr_target")
                rr_type = st.selectbox("Type", ["LONG", "SHORT"], key="rr_type")
            
            if rr_type == "LONG":
                risk = rr_entry - rr_sl
                reward = rr_target - rr_entry
            else:
                risk = rr_sl - rr_entry
                reward = rr_entry - rr_target
            
            if risk > 0 and reward > 0:
                ratio = reward / risk
                
                if ratio >= 3:
                    quality = "🟢 EXCELLENT"
                    color = "green"
                elif ratio >= 2:
                    quality = "🟢 GOOD"
                    color = "green"
                elif ratio >= 1.5:
                    quality = "🟡 ACCEPTABLE"
                    color = "orange"
                else:
                    quality = "🔴 POOR"
                    color = "red"
                
                st.markdown(f"**Risk-Reward Ratio:** <span style='color:{color};font-size:1.5em;'>1:{ratio:.2f}</span>",
                           unsafe_allow_html=True)
                st.markdown(f"**Quality:** {quality}")
                st.caption(f"Risk: ₹{risk:.2f} | Reward: ₹{reward:.2f}")
                
                if ratio < 2:
                    st.warning("⚠️ Minimum recommended: 1:2")
                
                # Win rate needed to be profitable
                breakeven_winrate = (1 / (1 + ratio)) * 100
                st.caption(f"Breakeven Win Rate: {breakeven_winrate:.1f}%")
            elif risk <= 0:
                st.error("Invalid: Risk must be positive!")
            else:
                st.error("Invalid: Reward must be positive!")
        
        st.divider()
        
        # =====================================================================
        # QUICK TRADE LOGGER
        # =====================================================================
        st.markdown("### 📝 Log Closed Trade")
        
        with st.expander("Record Trade Result", expanded=False):
            log_ticker = st.text_input("Ticker", placeholder="RELIANCE", key="log_ticker")
            log_type = st.selectbox("Position", ["LONG", "SHORT"], key="log_type")
            
            log_col1, log_col2 = st.columns(2)
            with log_col1:
                log_entry = st.number_input("Entry ₹", min_value=1.0, value=100.0, step=1.0, key="log_entry")
                log_qty = st.number_input("Quantity", min_value=1, value=10, step=1, key="log_qty")
            with log_col2:
                log_exit = st.number_input("Exit ₹", min_value=1.0, value=110.0, step=1.0, key="log_exit")
                log_reason = st.selectbox("Exit Reason", [
                    "Target Hit", "Stop Loss", "Trail SL", "Manual Exit", "Partial Exit"
                ], key="log_reason")
            
            if st.button("📊 Log Trade", use_container_width=True, key="log_trade_btn"):
                if log_ticker:
                    log_trade(log_ticker.upper(), log_entry, log_exit, log_qty, log_type, log_reason)
                    st.success(f"✅ Trade logged: {log_ticker.upper()}")
                    
                    # Show result
                    if log_type == "LONG":
                        pnl = (log_exit - log_entry) * log_qty
                    else:
                        pnl = (log_entry - log_exit) * log_qty
                    
                    if pnl >= 0:
                        st.success(f"Profit: ₹{pnl:+,.0f}")
                    else:
                        st.error(f"Loss: ₹{pnl:+,.0f}")
                else:
                    st.error("Enter ticker symbol")
        
        st.divider()
        
        # =====================================================================
        # RESET STATS
        # =====================================================================
        st.markdown("### 🔄 Reset Data")
        
        with st.expander("Reset Options", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Reset Stats", use_container_width=True, key="reset_stats"):
                    st.session_state.performance_stats = {
                        'total_trades': 0,
                        'wins': 0,
                        'losses': 0,
                        'total_profit': 0,
                        'total_loss': 0
                    }
                    st.session_state.trade_history = []
                    save_trade_history()  # ✅ Clear saved file too
                    st.session_state.max_drawdown = 0
                    st.session_state.current_drawdown = 0
                    st.session_state.peak_portfolio_value = 0
                    st.success("✅ Stats reset!")
                    time.sleep(1)
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Clear Cache", use_container_width=True, key="clear_cache"):
                    st.cache_data.clear()
                    st.success("✅ Cache cleared!")
                    time.sleep(1)
                    st.rerun()
            
            if st.button("🗑️ Reset Email Log", use_container_width=True, key="reset_email"):
                st.session_state.email_log = []
                st.session_state.email_sent_alerts = {}
                st.session_state.last_email_time = {}
                st.success("✅ Email log reset!")
        # =====================================================================
        # DEBUG INFO
        # =====================================================================
        with st.expander("🔧 Debug Info"):
            st.write(f"Email configured: {'✅ Yes' if credentials_configured else '❌ No'}")
            st.write(f"Email enabled: {'✅ Yes' if email_settings['enabled'] else '❌ No'}")
            st.write(f"Auto-refresh: {'✅ Installed' if HAS_AUTOREFRESH else '❌ Not installed'}")
            st.write(f"Refresh interval: {refresh_interval}s")
            st.write(f"Trail SL trigger: {trail_sl_trigger}%")
            st.write(f"SL Risk threshold: {sl_risk_threshold}%")
            st.write(f"SL Approach threshold: {sl_approach_threshold}%")
            st.write(f"API calls this session: {st.session_state.api_call_count}")
            
            if email_settings['enabled']:
                st.write(f"Email cooldown: {email_settings['cooldown']} min")
            
            # Email log
                        # Email log
            if st.session_state.email_log:
                st.markdown("**Recent Email Log:**")
                for log_entry in st.session_state.email_log[-5:]:
                    st.caption(log_entry)
                
                # Download button for full log
                full_log = "\n".join(st.session_state.email_log)
                st.download_button(
                    "📥 Download Full Log",
                    full_log,
                    file_name=f"email_log_{get_ist_now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    key="download_email_log"
                )
        
        # Return all settings
        return {
            'email_settings': email_settings,
            'auto_refresh': auto_refresh,
            'refresh_interval': refresh_interval,
            'loss_threshold': loss_threshold,
            'profit_threshold': profit_threshold,
            'trail_sl_trigger': trail_sl_trigger,
            'sl_risk_threshold': sl_risk_threshold,
            'sl_approach_threshold': sl_approach_threshold,
            'enable_volume_analysis': enable_volume_analysis,
            'enable_sr_detection': enable_sr_detection,
            'enable_multi_timeframe': enable_multi_timeframe,
            'enable_correlation': enable_correlation,
            'kite_mode': locals().get('kite_mode', 'PAPER'),
            'kite_enabled': locals().get('kite_enabled', False)
        }

# ============================================================================
# DISPLAY COMPONENTS
# ============================================================================

def display_portfolio_risk_dashboard(portfolio_risk, sector_analysis):
    """
    Display the portfolio risk dashboard
    """
    st.markdown("### 🛡️ Portfolio Risk Analysis")
    
    # Main metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💰 Total Capital",
            f"₹{portfolio_risk['total_capital']:,.0f}"
        )
    
    with col2:
        st.metric(
            "📈 Current Value",
            f"₹{portfolio_risk['current_value']:,.0f}",
            f"{portfolio_risk['total_pnl_pct']:+.2f}%"
        )
    
    with col3:
        st.metric(
            "🛡️ Total Risk",
            f"₹{portfolio_risk['total_risk_amount']:,.0f}",
            f"{portfolio_risk['portfolio_risk_pct']:.1f}%"
        )
    
    with col4:
        st.markdown(f"""
        <div style='text-align:center; padding:10px; background:linear-gradient(135deg, {portfolio_risk['risk_color']}, {portfolio_risk['risk_color']}90); 
                    border-radius:10px; color:white;'>
            <h3 style='margin:0;'>{portfolio_risk['risk_icon']} {portfolio_risk['risk_status']}</h3>
            <p style='margin:5px 0; font-size:0.8em;'>Risk Status</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.metric(
            "⚠️ Risky Positions",
            portfolio_risk['risky_positions'],
            f"of {portfolio_risk['total_positions']}"
        )
    
    # Risk recommendation
    if portfolio_risk['portfolio_risk_pct'] > 10:
        st.error(f"🚨 Portfolio risk is HIGH ({portfolio_risk['portfolio_risk_pct']:.1f}%). Consider reducing exposure!")
    elif portfolio_risk['portfolio_risk_pct'] > 5:
        st.warning(f"⚠️ Portfolio risk is MEDIUM ({portfolio_risk['portfolio_risk_pct']:.1f}%). Monitor closely.")
    else:
        st.success(f"✅ Portfolio risk is SAFE ({portfolio_risk['portfolio_risk_pct']:.1f}%).")
    
    # Drawdown info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📉 Current Drawdown", f"{st.session_state.current_drawdown:.2f}%")
    with col2:
        st.metric("📊 Max Drawdown", f"{st.session_state.max_drawdown:.2f}%")
    with col3:
        st.metric("🎯 Avg SL Risk", f"{portfolio_risk['avg_sl_risk']:.1f}%")
    
    # Sector exposure warnings
    if sector_analysis['warnings']:
        st.markdown("#### ⚠️ Concentration Warnings")
        for warning in sector_analysis['warnings']:
            st.warning(warning)

def display_performance_dashboard():
    """
    Display performance statistics dashboard
    """
    st.markdown("### 📈 Performance Dashboard")
    
    perf_stats = get_performance_stats()
    
    if perf_stats:
        # Main stats row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📊 Total Trades", perf_stats['total_trades'])
        
        with col2:
            win_color = "normal" if perf_stats['win_rate'] >= 50 else "inverse"
            st.metric("🎯 Win Rate", f"{perf_stats['win_rate']:.1f}%")
        
        with col3:
            st.metric("✅ Wins / ❌ Losses", f"{perf_stats['wins']} / {perf_stats['losses']}")
        
        with col4:
            exp_color = "normal" if perf_stats['expectancy'] >= 0 else "inverse"
            st.metric("📈 Expectancy", f"₹{perf_stats['expectancy']:,.0f}")
        
        with col5:
            pf_color = "normal" if perf_stats['profit_factor'] >= 1 else "inverse"
            pf_display = f"{perf_stats['profit_factor']:.2f}" if perf_stats['profit_factor'] < 100 else "∞"
            st.metric("⚖️ Profit Factor", pf_display)
        
        # Second row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "💰 Net Profit", 
                f"₹{perf_stats['net_profit']:+,.0f}"
            )
        
        with col2:
            st.metric("📈 Avg Win", f"₹{perf_stats['avg_win']:,.0f}")
        
        with col3:
            st.metric("📉 Avg Loss", f"₹{perf_stats['avg_loss']:,.0f}")
        
        with col4:
            if perf_stats['avg_loss'] > 0:
                rr_actual = perf_stats['avg_win'] / perf_stats['avg_loss']
                st.metric("⚖️ Actual R:R", f"1:{rr_actual:.2f}")
            else:
                st.metric("⚖️ Actual R:R", "N/A")
        
        with col5:
            total_costs = st.session_state.performance_stats.get(
                'total_costs', 0
            )
            st.metric(
                "💸 Total Costs", 
                f"₹{total_costs:,.0f}",
                help="Brokerage + STT + GST + Exchange fees"
            )
        
        # Performance assessment
        if perf_stats['win_rate'] >= 60 and perf_stats['profit_factor'] >= 1.5:
            st.success("🌟 Excellent performance! Keep up the good work.")
        elif perf_stats['win_rate'] >= 50 and perf_stats['profit_factor'] >= 1.2:
            st.info("👍 Good performance. Room for improvement.")
        elif perf_stats['win_rate'] >= 40 and perf_stats['profit_factor'] >= 1.0:
            st.warning("⚠️ Marginal performance. Review your strategy.")
        else:
            st.error("🚨 Poor performance. Strategy review needed.")
        
        # Trade history
        if st.session_state.trade_history:
            st.markdown("#### 📋 Recent Trades")
            
            history_data = []
            for trade in reversed(
                st.session_state.trade_history[-10:]
            ):
                # Handle both old format (no costs) and new format
                timestamp = trade.get('timestamp', datetime.now())
                if isinstance(timestamp, str):
                    date_str = timestamp[:16]
                elif isinstance(timestamp, datetime):
                    date_str = timestamp.strftime('%Y-%m-%d %H:%M')
                else:
                    date_str = str(timestamp)[:16]
                
                gross_pnl = trade.get(
                    'gross_pnl', trade.get('pnl', 0)
                )
                net_pnl = trade.get('pnl', 0)
                costs = trade.get('transaction_costs', 0)
                net_pct = trade.get(
                    'pnl_pct', trade.get('pnl_pct', 0)
                )
                
                history_data.append({
                    'Date': date_str,
                    'Ticker': trade.get('ticker', ''),
                    'Type': trade.get('type', ''),
                    'Entry': f"₹{trade.get('entry', 0):.2f}",
                    'Exit': f"₹{trade.get('exit', 0):.2f}",
                    'Qty': trade.get('quantity', 0),
                    'Gross': f"₹{gross_pnl:+,.0f}",
                    'Costs': f"₹{costs:.0f}",
                    'Net P&L': f"₹{net_pnl:+,.0f}",
                    'Net %': f"{net_pct:+.2f}%",
                    'Result': '✅' if trade.get('win', False) else '❌',
                    'Reason': trade.get('reason', '')
                })
            
            df_history = pd.DataFrame(history_data)
            st.dataframe(df_history, use_container_width=True, hide_index=True)
            
            # Export button
            csv = df_history.to_csv(index=False)
            st.download_button(
                "📥 Download Trade History",
                csv,
                file_name=f"trade_history_{get_ist_now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    else:
        st.info("📊 No trades logged yet. Use the 'Log Closed Trade' feature in the sidebar to record trades.")
        
        # Sample explanation
        st.markdown("""
        **How to track performance:**
        1. When you close a trade, go to sidebar → 'Log Closed Trade'
        2. Enter the trade details (entry, exit, quantity)
        3. Click 'Log Trade'
        4. View your statistics here!
        """)

def display_sector_analysis(sector_analysis):
    """
    Display sector exposure analysis
    """
    st.markdown("### 🏢 Sector Exposure")
    
    if not sector_analysis['sectors']:
        st.info("No sector data available")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Pie chart
        sector_data = []
        for sector, data in sector_analysis['sectors'].items():
            sector_data.append({
                'Sector': sector,
                'Percentage': data['percentage'],
                'Value': data['value']
            })
        
        df_sector = pd.DataFrame(sector_data)
        
        fig = px.pie(
            df_sector,
            values='Percentage',
            names='Sector',
            title='Sector Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Sector Breakdown")
        
        for sector, data in list(sector_analysis['sectors'].items())[:8]:
            pnl_color = "green" if data['pnl'] >= 0 else "red"
            st.markdown(f"""
            **{sector}** ({data['percentage']:.1f}%)
            - Stocks: {data['count']}
            - Value: ₹{data['value']:,.0f}
            - P&L: <span style='color:{pnl_color}'>₹{data['pnl']:+,.0f}</span>
            """, unsafe_allow_html=True)
            st.divider()
        
        # Diversification score
        score = sector_analysis['diversification_score']
        if score >= 70:
            score_color = "#28a745"
            score_text = "Well Diversified"
        elif score >= 50:
            score_color = "#ffc107"
            score_text = "Moderately Diversified"
        else:
            score_color = "#dc3545"
            score_text = "Poorly Diversified"
        
        st.markdown(f"""
        <div style='text-align:center; padding:15px; background:{score_color}20; border-radius:10px; border-left:4px solid {score_color};'>
            <h2 style='margin:0; color:{score_color};'>{score}/100</h2>
            <p style='margin:5px 0;'>{score_text}</p>
        </div>
        """, unsafe_allow_html=True)

def display_correlation_analysis(results, enable_correlation):
    """
    Display correlation analysis
    """
    st.markdown("### 🔗 Correlation Analysis")
    
    if not enable_correlation:
        st.info("Correlation analysis is disabled. Enable it in sidebar settings.")
        return
    
    tickers = [r['ticker'] for r in results]
    
    if len(tickers) < 2:
        st.warning("Need at least 2 positions for correlation analysis")
        return
    
    # Check cache
    cache_valid = (
        st.session_state.correlation_matrix is not None and
        st.session_state.last_correlation_calc is not None and
        (datetime.now() - st.session_state.last_correlation_calc).seconds < 300
    )
    
    if not cache_valid:
        with st.spinner("Calculating correlations..."):
            corr_matrix, status = calculate_correlation_matrix(tickers)
            if corr_matrix is not None:
                st.session_state.correlation_matrix = corr_matrix
                st.session_state.last_correlation_calc = datetime.now()
    else:
        corr_matrix = st.session_state.correlation_matrix
        status = "Cached"
    
    if corr_matrix is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Heatmap
            fig = px.imshow(
                corr_matrix,
                labels=dict(x="Stock", y="Stock", color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.index,
                color_continuous_scale="RdYlGn",
                zmin=-1, zmax=1
            )
            fig.update_layout(title="Correlation Matrix", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            high_corr, avg_corr, corr_status = analyze_correlation_risk(corr_matrix)
            
            st.markdown(f"**Average Correlation:** {avg_corr:.2f}")
            st.markdown(f"**Status:** {corr_status}")
            
            if high_corr:
                st.markdown("#### ⚠️ High Correlations")
                for hc in high_corr[:5]:
                    risk_color = "red" if hc['risk'] == 'HIGH' else "orange"
                    st.markdown(f"""
                    <div style='background:{risk_color}20; padding:8px; margin:5px 0; border-radius:5px;'>
                        <strong>{hc['pair']}</strong>: {hc['correlation']:.2f}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No highly correlated pairs found")
    else:
        st.error(f"Could not calculate correlations: {status}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application entry point
    """
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Smart Portfolio Monitor v6.0</h1>', unsafe_allow_html=True)
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Market Status
    is_open, market_status, market_msg, market_icon = is_market_hours()
    ist_now = get_ist_now()
    
    # =========================================================================
    # HEADER ROW (Market Status + Time + Refresh Button)
    # =========================================================================
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown(f"### {market_icon} {market_status}")
        st.caption(market_msg)
    
    with col2:
        st.markdown(f"### 🕐 {ist_now.strftime('%H:%M:%S')} IST")
        st.caption(ist_now.strftime('%A, %B %d, %Y'))
    
    with col3:
        if st.button("🔄 Refresh", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Kite Connection Status Bar
    if HAS_KITE and settings.get('kite_enabled', False):
        kite_mode = settings.get('kite_mode', 'PAPER')
        
        if kite_mode == 'LIVE' and kite_session and kite_session.is_connected:
            st.markdown("""
            <div style='background:#28a745; color:white; padding:8px 15px; 
                        border-radius:8px; text-align:center; margin:5px 0;'>
                🔗 <strong>ZERODHA LIVE</strong> - Orders will be executed automatically
            </div>
            """, unsafe_allow_html=True)
        elif kite_mode == 'PAPER':
            st.markdown("""
            <div style='background:#17a2b8; color:white; padding:8px 15px; 
                        border-radius:8px; text-align:center; margin:5px 0;'>
                📝 <strong>PAPER MODE</strong> - Simulating orders (no real trades)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#ffc107; color:black; padding:8px 15px; 
                        border-radius:8px; text-align:center; margin:5px 0;'>
                ⚠️ <strong>ZERODHA DISCONNECTED</strong> - Login required in sidebar
            </div>
            """, unsafe_allow_html=True)
    
    # =========================================================================
    # ✅ GAP 1: MARKET HEALTH DISPLAY (OUTSIDE COLUMNS - FULL WIDTH!)
    # =========================================================================
    st.divider()
    
    market_health = get_market_health()
    
    if market_health:
        st.markdown(f"""
        <div style='background:{market_health['color']}20; padding:20px; border-radius:12px; 
                    border-left:5px solid {market_health['color']}; margin:15px 0;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <h2 style='margin:0; color:{market_health['color']};'>
                        {market_health['icon']} Market Health: {market_health['status']}
                    </h2>
                    <p style='margin:8px 0; font-size:1.1em;'>{market_health['message']}</p>
                    <p style='margin:8px 0; font-weight:bold; font-size:1.05em;'>{market_health['action']}</p>
                </div>
                <div style='text-align:center; min-width:100px;'>
                    <h1 style='margin:0; color:{market_health['color']};'>{market_health['health_score']}</h1>
                    <p style='margin:0; font-size:0.9em;'>Health Score</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show detailed metrics in expander
        with st.expander("📊 Market Details", expanded=False):
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            
            with m_col1:
                st.metric("NIFTY 50", f"₹{market_health['nifty_price']:,.0f}", 
                         f"{market_health['nifty_change']:+.2f}%")
            
            with m_col2:
                st.metric("RSI", f"{market_health['nifty_rsi']:.1f}")
            
            with m_col3:
                st.metric("India VIX", f"{market_health['vix']:.1f}")
            
            with m_col4:
                st.metric("Trend", "Bullish" if market_health['above_sma20'] else "Bearish")
            
            # Technical levels
            st.caption(f"NIFTY SMA20: ₹{market_health['nifty_sma20']:,.0f} | SMA50: ₹{market_health['nifty_sma50']:,.0f}")
        
        # ✅ AUTO-ADJUST SL THRESHOLDS BASED ON MARKET
        if market_health['sl_adjustment'] == 'AGGRESSIVE':
            settings['sl_risk_threshold'] = max(30, settings['sl_risk_threshold'] - 20)
            st.warning(f"⚠️ SL Risk threshold auto-adjusted to {settings['sl_risk_threshold']}% due to weak market")
        elif market_health['sl_adjustment'] == 'TIGHTEN':
            settings['sl_risk_threshold'] = max(35, settings['sl_risk_threshold'] - 10)
            st.info(f"ℹ️ SL Risk threshold adjusted to {settings['sl_risk_threshold']}% (cautious mode)")
    
    else:
        market_health = None  # Set to None if fetch failed
        st.warning("⚠️ Unable to fetch market health data")
    
    st.divider()
    
    # =========================================================================
    # SETTINGS SUMMARY
    # =========================================================================
    with st.expander("⚙️ Current Settings", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Trail SL Trigger", f"{settings['trail_sl_trigger']}%")
        with col2:
            st.metric("SL Risk Alert", f"{settings['sl_risk_threshold']}%")
        with col3:
            st.metric("Refresh Interval", f"{settings['refresh_interval']}s")
        with col4:
            st.metric("MTF Analysis", "✅ On" if settings['enable_multi_timeframe'] else "❌ Off")
        with col5:
            st.metric("Email Alerts", "✅ On" if settings['email_settings']['enabled'] else "❌ Off")
    
    st.divider()
    
    # Load Portfolio
        # Load Portfolio
    portfolio = load_portfolio()
    
    if portfolio is None or len(portfolio) == 0:
        st.warning("⚠️ No positions found!")
        
        # Show sample format
        st.markdown("### 📋 Expected Google Sheets Format:")
        sample_df = pd.DataFrame({
            'Ticker': ['RELIANCE', 'TCS', 'INFY'],
            'Position': ['LONG', 'LONG', 'SHORT'],
            'Entry_Price': [2450.00, 3580.00, 1520.00],
            'Quantity': [10, 5, 8],
            'Stop_Loss': [2380.00, 3480.00, 1580.00],
            'Target_1': [2550.00, 3720.00, 1420.00],
            'Target_2': [2650.00, 3850.00, 1350.00],
            'Entry_Date': ['2024-01-15', '2024-01-20', '2024-02-01'],
            'Status': ['ACTIVE', 'ACTIVE', 'ACTIVE']
        })
        st.dataframe(sample_df, use_container_width=True)
        return
    
    # Validate Portfolio Data
    is_valid, errors, warnings = validate_portfolio(portfolio)
    
    if errors:
        st.error("❌ Portfolio Validation Failed!")
        for error in errors:
            st.error(error)
        st.stop()
    
    if warnings:
        with st.expander("⚠️ Validation Warnings", expanded=False):
            for warning in warnings:
                st.warning(warning)
        
        # Show sample format
        st.markdown("### 📋 Expected Google Sheets Format:")
        sample_df = pd.DataFrame({
            'Ticker': ['RELIANCE', 'TCS', 'INFY'],
            'Position': ['LONG', 'LONG', 'SHORT'],
            'Entry_Price': [2450.00, 3580.00, 1520.00],
            'Quantity': [10, 5, 8],
            'Stop_Loss': [2380.00, 3480.00, 1580.00],
            'Target_1': [2550.00, 3720.00, 1420.00],
            'Target_2': [2650.00, 3850.00, 1350.00],
            'Entry_Date': ['2024-01-15', '2024-01-20', '2024-02-01'],
            'Status': ['ACTIVE', 'ACTIVE', 'ACTIVE']
        })
        st.dataframe(sample_df, use_container_width=True)
    if warnings:
        with st.expander("⚠️ Validation Warnings", expanded=False):
            for warning in warnings:
                st.warning(warning)
    
    # =========================================================================
    # ANALYZE ALL POSITIONS
    # =========================================================================
    results = []
    progress_bar = st.progress(0, text="Analyzing positions...")
    
    for i, (_, row) in enumerate(portfolio.iterrows()):
        ticker = str(row['Ticker']).strip()
        progress_bar.progress((i + 0.5) / len(portfolio), text=f"Analyzing {ticker}...")
        
        # Get entry date if available
        entry_date = row.get('Entry_Date', None)
        
        result = smart_analyze_position(
            ticker,
            str(row['Position']).upper().strip(),
            float(row['Entry_Price']),
            int(row.get('Quantity', 1)),
            float(row['Stop_Loss']),
            float(row['Target_1']),
            float(row.get('Target_2', row['Target_1'] * 1.1)),
            settings['trail_sl_trigger'],
            settings['sl_risk_threshold'],
            settings['sl_approach_threshold'],
            settings['enable_multi_timeframe'],
            entry_date
        )
        
        if result:
            results.append(result)
        
        progress_bar.progress((i + 1) / len(portfolio), text=f"Completed {ticker}")
    
    progress_bar.empty()
    
    if not results:
        st.error("❌ Could not fetch stock data. Check internet connection and try again.")
        return
    
    # =========================================================================
    # CALCULATE PORTFOLIO METRICS
    # =========================================================================
    
    # Portfolio risk
    portfolio_risk = calculate_portfolio_risk(results)
    
    # Sector analysis
    sector_analysis = analyze_sector_exposure(results)
    
    # Update drawdown
    update_drawdown(portfolio_risk['current_value'])
    
    # Summary counts
    total_pnl = sum(r['pnl_amount'] for r in results)
    total_invested = sum(r['entry_price'] * r['quantity'] for r in results)
    pnl_percent_total = (total_pnl / total_invested * 100) if total_invested > 0 else 0
    
    critical_count = sum(1 for r in results if r['overall_status'] == 'CRITICAL')
    warning_count = sum(1 for r in results if r['overall_status'] == 'WARNING')
    opportunity_count = sum(1 for r in results if r['overall_status'] == 'OPPORTUNITY')
    success_count = sum(1 for r in results if r['overall_status'] == 'SUCCESS')
    good_count = sum(1 for r in results if r['overall_status'] == 'GOOD')
    
    # =========================================================================
    # SEND EMAIL ALERTS
    # =========================================================================
    if settings['email_settings']['enabled']:
        send_portfolio_alerts(results, settings['email_settings'], portfolio_risk)
    
    # =========================================================================
    # DISPLAY SUMMARY CARDS
    # =========================================================================
    st.markdown("### 📊 Portfolio Summary")
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
    with col1:
        pnl_delta = f"{pnl_percent_total:+.2f}%"
        st.metric("💰 Total P&L", f"₹{total_pnl:+,.0f}", pnl_delta)
    with col2:
        st.metric("📊 Positions", len(results))
    with col3:
        st.metric("🔴 Critical", critical_count)
    with col4:
        st.metric("🟡 Warning", warning_count)
    with col5:
        st.metric("🟢 Good", good_count)
    with col6:
        st.metric("🔵 Opportunity", opportunity_count)
    with col7:
        st.metric("✅ Success", success_count)
    
    st.divider()
    
    # =========================================================================
    # MAIN TABS
    # =========================================================================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7= st.tabs([
        "📊 Dashboard",
        "📈 Charts", 
        "🔔 Alerts",
        "📉 MTF Analysis",
        "🛡️ Portfolio Risk",
        "📈 Performance",
        "📋 Details"
    ])
    
    # =========================================================================
    # TAB 1: DASHBOARD
    # =========================================================================
    with tab1:
        # Sort by status priority
        def get_sort_priority(r):
            priority = calculate_position_priority(r, market_health)
            status_order = {'CRITICAL': 0, 'WARNING': 1, 'OPPORTUNITY': 2, 'SUCCESS': 3, 'GOOD': 4, 'OK': 5}
            return (-priority['urgency_score'], status_order.get(r['overall_status'], 5))
        
        sorted_results = sorted(results, key=get_sort_priority)
        
        for r in sorted_results:
            # Fetch df for this stock
            _stock_df = get_stock_data_safe(r['ticker'], period="6mo")
            r['df'] = _stock_df
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # HANDLE PENDING POSITIONS (9:20AM Confirmation)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            # Check if this is a PENDING position from the portfolio
            is_pending = False
            
            # Find original row in portfolio to check status
            for _, port_row in portfolio.iterrows():
                if (str(port_row['Ticker']).strip() == 
                        str(r['ticker']).replace('.NS', '').replace('.BO', '').strip()):
                    if str(port_row.get('Status', 'ACTIVE')).upper() == 'PENDING':
                        is_pending = True
                    break
            
            if is_pending:
                # Show PENDING card
                with st.expander(
                    f"⏳ **{r['ticker']}** | PENDING | "
                    f"{'📈 LONG' if r['position_type'] == 'LONG' else '📉 SHORT'} | "
                    f"Entry: ₹{r['entry_price']:,.2f} | "
                    f"Waiting for 9:20AM confirmation",
                    expanded=True
                ):
                    # Check if already processed today
                    already_confirmed, conf_time = get_persistent_guard(
                        r['ticker'], "entry_confirmed"
                    )
                    already_skipped, skip_time = get_persistent_guard(
                        r['ticker'], "entry_skipped"
                    )
                    
                    if already_skipped:
                        st.error(
                            f"⏭️ **SKIPPED TODAY** - {skip_time}"
                        )
                        st.caption(
                            "This trade was skipped because 15-min "
                            "confirmation failed."
                        )
                        # Don't render rest of stock card
                        continue
                    
                    if already_confirmed:
                        st.success(
                            f"✅ **CONFIRMED** - {conf_time}"
                        )
                        st.info(
                            "Entry confirmed. Placing order..."
                        )
                        
                        # AUTO-ACTIVATE: Update sheet status
                        activate_guard_key = f"activated_{r['ticker']}"
                        already_activated, _ = get_persistent_guard(
                            r['ticker'], "activated"
                        )
                        
                        if not already_activated:
                            # Place order via Kite if LIVE
                            kite_mode = st.session_state.get(
                                'kite_mode', 'PAPER'
                            )
                            
                            if (kite_mode == 'LIVE' and HAS_KITE and 
                                    kite_session and 
                                    kite_session.is_connected):
                                
                                txn_type = (
                                    "BUY" if r['position_type'] == 'LONG' 
                                    else "SELL"
                                )
                                
                                with st.spinner(
                                    f"Placing {txn_type} order for "
                                    f"{r['ticker']}..."
                                ):
                                    order_success, order_msg, order_id = (
                                        kite_session.place_exit_order(
                                            ticker=r['ticker'],
                                            quantity=r['quantity'],
                                            transaction_type=txn_type,
                                            order_type="MARKET",
                                            product="CNC",
                                            tag="ConfirmedEntry"
                                        )
                                    )
                                    
                                    if order_success:
                                        st.success(
                                            f"✅ Order placed: {order_msg}"
                                        )
                                        
                                        # Update sheet: PENDING → ACTIVE
                                        update_sheet_status(
                                            r['ticker'], 'ACTIVE',
                                            f'Confirmed & entered. '
                                            f'Order: {order_id}'
                                        )
                                        
                                        # Save order ID
                                        try:
                                            sheet, _ = (
                                                get_google_sheet_connection()
                                            )
                                            if sheet:
                                                cell = sheet.find(
                                                    r['ticker']
                                                )
                                                if cell:
                                                    headers = (
                                                        sheet.row_values(1)
                                                    )
                                                    if 'Zerodha_Order_ID' in headers:
                                                        oid_col = (
                                                            headers.index(
                                                                'Zerodha_Order_ID'
                                                            ) + 1
                                                        )
                                                        sheet.update_cell(
                                                            cell.row, 
                                                            oid_col, 
                                                            order_id
                                                        )
                                        except Exception:
                                            pass
                                        
                                        set_persistent_guard(
                                            r['ticker'], "activated",
                                            f"Order {order_id}"
                                        )
                                        
                                        st.balloons()
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error(
                                            f"❌ Order failed: {order_msg}"
                                        )
                            
                            else:
                                # PAPER mode - just activate
                                update_sheet_status(
                                    r['ticker'], 'ACTIVE',
                                    'Confirmed (Paper mode)'
                                )
                                set_persistent_guard(
                                    r['ticker'], "activated",
                                    "Paper mode"
                                )
                                st.info(
                                    "📝 Paper mode - "
                                    "Status updated to ACTIVE"
                                )
                                time.sleep(1)
                                st.rerun()
                        
                        continue  # Skip rest of card
                    
                    # NOT YET CONFIRMED - Show confirmation UI
                    if not is_confirmation_window():
                        if is_past_entry_window():
                            # Past 9:35AM - skip for the day
                            st.warning(
                                "⏰ **Entry window closed (9:35AM)**"
                            )
                            st.info(
                                "Trade will be re-evaluated tomorrow "
                                "at 9:20AM."
                            )
                            
                            # Auto-skip
                            set_persistent_guard(
                                r['ticker'], "entry_skipped",
                                "Past 9:35AM entry window"
                            )
                        else:
                            # Before 9:20AM
                            st.info(
                                "⏳ **Waiting for 9:20AM confirmation "
                                "window...**"
                            )
                            
                            # Show planned trade details
                            p_col1, p_col2, p_col3 = st.columns(3)
                            with p_col1:
                                st.metric(
                                    "Entry", 
                                    f"₹{r['entry_price']:,.2f}"
                                )
                            with p_col2:
                                st.metric(
                                    "Stop Loss", 
                                    f"₹{r['stop_loss']:,.2f}"
                                )
                            with p_col3:
                                st.metric(
                                    "Target", 
                                    f"₹{r['target1']:,.2f}"
                                )
                            
                            st.caption(
                                "At 9:20AM the system will check: "
                                "Gap, EMA, Volume, MACD on 15-min chart"
                            )
                    else:
                        # IN THE WINDOW - Run confirmation
                        st.info(
                            f"🔍 Running 9:20AM confirmation for "
                            f"{r['ticker']}..."
                        )
                        
                        confirmation = get_15min_confirmation(
                            r['ticker'],
                            r['position_type'],
                            r['entry_price']
                        )
                        
                        if confirmation['skip']:
                            st.error(
                                f"❌ **SKIPPED** - "
                                f"{confirmation['reason']}"
                            )
                            
                            # Show details if available
                            if 'details' in confirmation:
                                for detail in confirmation['details']:
                                    st.caption(detail)
                            
                            set_persistent_guard(
                                r['ticker'], "entry_skipped",
                                confirmation['reason']
                            )
                            
                            # Update sheet
                            update_sheet_status(
                                r['ticker'], 'SKIPPED',
                                confirmation['reason']
                            )
                            
                            # Send email
                            if settings['email_settings']['enabled']:
                                sender = settings['email_settings'].get(
                                    'sender_email'
                                )
                                password = settings['email_settings'].get(
                                    'sender_password'
                                )
                                recipient = settings['email_settings'].get(
                                    'recipient_email'
                                )
                                
                                if sender and password and recipient:
                                    skip_html = f"""
                                    <html>
                                    <body style="font-family:Arial; 
                                                 padding:20px;">
                                        <div style="background:#ffc107; 
                                                    padding:20px; 
                                                    border-radius:10px; 
                                                    text-align:center;">
                                            <h1>⏭️ Trade Skipped</h1>
                                            <p style="font-size:1.3em;">
                                                {r['ticker']}
                                            </p>
                                        </div>
                                        <div style="padding:20px;">
                                            <p><strong>Reason:</strong> 
                                               {confirmation['reason']}
                                            </p>
                                            <p><strong>Side:</strong> 
                                               {r['position_type']}
                                            </p>
                                            <p><strong>Planned Entry:</strong>
                                               ₹{r['entry_price']:,.2f}
                                            </p>
                                            <p><strong>Gap:</strong> 
                                               {confirmation.get(
                                                   'gap_pct', 0
                                               ):+.2f}%
                                            </p>
                                        </div>
                                    </body>
                                    </html>
                                    """
                                    
                                    send_email_alert(
                                        f"⏭️ Trade Skipped - "
                                        f"{r['ticker']}",
                                        skip_html,
                                        sender, password, recipient
                                    )
                        
                        elif confirmation['confirmed']:
                            st.success(
                                f"✅ **CONFIRMED** - "
                                f"{confirmation['reason']}"
                            )
                            
                            if 'details' in confirmation:
                                for detail in confirmation['details']:
                                    st.caption(detail)
                            
                            if confirmation.get('reduce_size'):
                                st.warning(
                                    "⚠️ Consider reducing position "
                                    "size - MACD not aligned"
                                )
                            
                            set_persistent_guard(
                                r['ticker'], "entry_confirmed",
                                confirmation['reason']
                            )
                            
                            # Send email
                            if settings['email_settings']['enabled']:
                                sender = settings['email_settings'].get(
                                    'sender_email'
                                )
                                password = settings['email_settings'].get(
                                    'sender_password'
                                )
                                recipient = settings['email_settings'].get(
                                    'recipient_email'
                                )
                                
                                if sender and password and recipient:
                                    conf_html = f"""
                                    <html>
                                    <body style="font-family:Arial; 
                                                 padding:20px;">
                                        <div style="background:#28a745; 
                                                    color:white; 
                                                    padding:20px; 
                                                    border-radius:10px; 
                                                    text-align:center;">
                                            <h1>✅ Trade Confirmed</h1>
                                            <p style="font-size:1.3em;">
                                                {r['ticker']}
                                            </p>
                                        </div>
                                        <div style="padding:20px;">
                                            <p><strong>Result:</strong> 
                                               {confirmation['reason']}
                                            </p>
                                            <p><strong>Side:</strong> 
                                               {r['position_type']}
                                            </p>
                                            <p><strong>Entry:</strong> 
                                               ₹{r['entry_price']:,.2f}
                                            </p>
                                            <p><strong>Checks:</strong> 
                                               {confirmation.get(
                                                   'checks_passed', 0
                                               )}/
                                               {confirmation.get(
                                                   'checks_total', 0
                                               )}
                                            </p>
                                        </div>
                                    </body>
                                    </html>
                                    """
                                    
                                    send_email_alert(
                                        f"✅ Trade Confirmed - "
                                        f"{r['ticker']}",
                                        conf_html,
                                        sender, password, recipient
                                    )
                            
                            time.sleep(1)
                            st.rerun()
                    
                    continue  # Don't render normal card for PENDING
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # NORMAL ACTIVE POSITION (existing code continues here)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            status_icons = {
                'CRITICAL': '🔴', 'WARNING': '🟡', 'OPPORTUNITY': '🔵',
                'SUCCESS': '🟢', 'GOOD': '🟢', 'OK': '⚪'
            }
            status_icon = status_icons.get(r['overall_status'], '⚪')
            pnl_emoji = "📈" if r['pnl_percent'] >= 0 else "📉"
            
            with st.expander(
                f"{status_icon} **{r['ticker']}** | "
                f"{'📈 LONG' if r['position_type'] == 'LONG' else '📉 SHORT'} | "
                f"{pnl_emoji} P&L: **{r['pnl_percent']:+.2f}%** (₹{r['pnl_amount']:+,.0f}) | "
                f"SL Risk: **{r['sl_risk']}%** | "
                f"**{r.get('unified_label', r['overall_action'].replace('_', ' '))}**",
                expanded=(r['overall_status'] in ['CRITICAL', 'WARNING', 'OPPORTUNITY', 'SUCCESS'])
            ):
                # ✅ GAP 2: CHECK FOR EMERGENCY EXIT
                is_emergency, emergency_reasons, urgency_level = detect_emergency_exit(r, market_health)
                
                if is_emergency:
                    if urgency_level == "CRITICAL":
                        st.markdown("""
                        <div style='background:#dc3545; color:white; padding:15px; border-radius:10px; 
                                    text-align:center; font-size:1.2em; font-weight:bold; margin-bottom:15px;
                                    animation: blink 1s infinite;'>
                            🚨 EMERGENCY EXIT REQUIRED 🚨
                        </div>
                        <style>
                        @keyframes blink {
                            0%, 50%, 100% { opacity: 1; }
                            25%, 75% { opacity: 0.5; }
                        }
                        </style>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("⚠️ HIGH URGENCY - Consider immediate exit")
                    
                    st.markdown("**Emergency Conditions:**")
                    for reason in emergency_reasons:
                        st.error(f"• {reason}")
                    
                    st.divider()
                # ✅ GAP 3: STOCK WIN RATE WARNING
                stock_history = get_stock_performance_history(r['ticker'])
                
                if stock_history['has_history']:
                    if stock_history['win_rate'] < 45 or stock_history['expectancy'] < 0:
                        st.markdown(f"""
                        <div style='background:{stock_history['color']}20; padding:12px; border-radius:8px; 
                                    border-left:4px solid {stock_history['color']}; margin-bottom:15px;'>
                            <strong>{stock_history['icon']} Historical Performance: {stock_history['quality']}</strong><br>
                            Win Rate: {stock_history['win_rate']:.1f}% ({stock_history['wins']}/{stock_history['trade_count']}) | 
                            Expectancy: ₹{stock_history['expectancy']:+,.0f}<br>
                            <strong>{stock_history['recommendation']}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        with st.expander(f"{stock_history['icon']} Historical: {stock_history['quality']} ({stock_history['win_rate']:.0f}% win rate)", expanded=False):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Win Rate", f"{stock_history['win_rate']:.1f}%")
                            with col2:
                                st.metric("Trades", f"{stock_history['wins']}/{stock_history['trade_count']}")
                            with col3:
                                st.metric("Expectancy", f"₹{stock_history['expectancy']:+,.0f}")
                
                st.divider()
                # Row 1: Basic Info
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("##### 💰 Position")
                    st.write(f"**Entry:** ₹{r['entry_price']:,.2f}")
                    st.write(f"**Current:** ₹{r['current_price']:,.2f}")
                    st.write(f"**Qty:** {r['quantity']}")
                    pnl_color = "green" if r['pnl_percent'] >= 0 else "red"
                    st.markdown(f"**P&L:** <span style='color:{pnl_color};font-weight:bold;'>"
                               f"₹{r['pnl_amount']:+,.2f} ({r['pnl_percent']:+.2f}%)</span>",
                               unsafe_allow_html=True)
                    if r['holding_days'] > 0:
                        st.caption(f"Holding: {r['holding_days']} days | {r['tax_color']} {r['tax_implication']}")
                
                with col2:
                    st.markdown("##### 🎯 Levels")
                    sl_status = '🔴 HIT!' if r['sl_hit'] else ''
                    t1_status = '✅' if r['target1_hit'] else ''
                    t2_status = '✅' if r['target2_hit'] else ''
                    
                    st.write(f"**Stop Loss:** ₹{r['stop_loss']:,.2f} {sl_status}")
                    st.write(f"**Target 1:** ₹{r['target1']:,.2f} {t1_status}")
                    st.write(f"**Target 2:** ₹{r['target2']:,.2f} {t2_status}")
                    
                    if r['should_trail']:
                        st.success(f"**Trail SL:** ₹{r['trail_stop']:,.2f}")
                        st.caption(r.get('trail_reason', ''))
                    
                    if r['at_breakeven']:
                        st.info("🔔 At Breakeven")
                
                with col3:
                    st.markdown("##### 📊 Indicators")
                    rsi_color = "green" if 40 <= r['rsi'] <= 60 else "orange" if 30 <= r['rsi'] <= 70 else "red"
                    st.markdown(f"**RSI:** <span style='color:{rsi_color};'>{r['rsi']:.1f}</span>", 
                               unsafe_allow_html=True)
                    macd_color = "green" if r['macd_signal'] == "BULLISH" else "red"
                    st.markdown(f"**MACD:** <span style='color:{macd_color};'>{r['macd_signal']}</span>", 
                               unsafe_allow_html=True)
                    st.write(f"**Volume:** {r['volume_signal'].replace('_', ' ')}")
                    st.write(f"**Trend:** {r['momentum_trend']}")
                    st.write(f"**R:R Ratio:** 1:{r['risk_reward_ratio']:.2f}")
                
                with col4:
                    st.markdown("##### 🛡️ Support/Resistance")
                    st.write(f"**Support:** ₹{r['support']:,.2f} ({r['support_strength']})")
                    st.write(f"**Resistance:** ₹{r['resistance']:,.2f} ({r['resistance_strength']})")
                    st.write(f"**ATR:** ₹{r['atr']:,.2f}")
                    st.write(f"**Dist to S:** {r['distance_to_support']:.1f}%")
                    st.write(f"**Dist to R:** {r['distance_to_resistance']:.1f}%")
                
                st.divider()

                                # Check entry trigger status
                
                # Row 2: Smart Scores
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("##### ⚠️ SL Risk Score")
                    risk_color = "#dc3545" if r['sl_risk'] >= 70 else "#ffc107" if r['sl_risk'] >= 50 else "#28a745"
                    st.markdown(f"<h2 style='color:{risk_color};text-align:center;'>{r['sl_risk']}%</h2>",
                               unsafe_allow_html=True)
                    st.progress(r['sl_risk'] / 100)
                    if r['sl_reasons']:
                        for reason in r['sl_reasons'][:3]:
                            st.caption(reason)
                
                with col2:
                    st.markdown("##### 📈 Momentum Score")
                    mom_color = "#28a745" if r['momentum_score'] >= 60 else "#ffc107" if r['momentum_score'] >= 40 else "#dc3545"
                    st.markdown(f"<h2 style='color:{mom_color};text-align:center;'>{r['momentum_score']:.0f}/100</h2>",
                               unsafe_allow_html=True)
                    st.progress(r['momentum_score'] / 100)
                    st.caption(r['momentum_trend'])
                
                with col3:
                    st.markdown("##### 🚀 Upside Score")
                    if r['target1_hit']:
                        up_color = "#28a745" if r['upside_score'] >= 60 else "#ffc107" if r['upside_score'] >= 40 else "#dc3545"
                        st.markdown(f"<h2 style='color:{up_color};text-align:center;'>{r['upside_score']}%</h2>",
                                   unsafe_allow_html=True)
                        st.progress(r['upside_score'] / 100)
                        if r['upside_score'] >= 60:
                            st.success(f"New Target: ₹{r['new_target']:,.2f}")
                    else:
                        st.markdown("<h2 style='color:#6c757d;text-align:center;'>N/A</h2>",
                                   unsafe_allow_html=True)
                        st.caption("Target not yet hit")
                
                with col4:
                    st.markdown("##### 📊 MTF Alignment")
                    if r['mtf_signals']:
                        mtf_color = "#28a745" if r['mtf_alignment'] >= 60 else "#ffc107" if r['mtf_alignment'] >= 40 else "#dc3545"
                        st.markdown(f"<h2 style='color:{mtf_color};text-align:center;'>{r['mtf_alignment']}%</h2>",
                                   unsafe_allow_html=True)
                        st.progress(r['mtf_alignment'] / 100)
                        for tf, signal in r['mtf_signals'].items():
                            sig_emoji = "🟢" if signal == "BULLISH" else "🔴" if signal == "BEARISH" else "⚪"
                            st.caption(f"{tf}: {sig_emoji} {signal}")
                    else:
                        st.markdown("<h2 style='color:#6c757d;text-align:center;'>N/A</h2>",
                                   unsafe_allow_html=True)
                        st.caption("MTF data unavailable")
                                    # ✅ GAP 4: CHART PATTERN DETECTION

                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # UNIFIED: SMART EXIT ANALYSIS (uses pre-calculated decision)
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                exit_decision = r.get('exit_decision', None)
                
                # Re-run with market_health if available (enhances the analysis)
                if 'df' in r and r['df'] is not None and market_health:
                    exit_decision = smart_exit_decision(
                        r, market_health, r['df']
                    )
                    # Store for consistency
                    r['exit_decision'] = exit_decision
                    
                    # Check if market_health changes the overall decision
                    # (because in smart_analyze_position we passed market_health=None)
                    if exit_decision['action'] in ['EXIT_NOW', 'EXIT_TODAY']:
                        if exit_decision['score'] <= 20 and r['overall_status'] not in ['CRITICAL']:
                            # Market health made it worse - update display
                            r['overall_status'] = 'CRITICAL'
                            r['overall_action'] = 'EXIT'
                            r['unified_label'] = '🚨 EXIT NOW'
                
                if exit_decision:
                    st.divider()
                    st.markdown("##### 🧠 Smart Exit Analysis")
                    
                    exit_col1, exit_col2 = st.columns([1, 2])
                    
                    with exit_col1:
                        exit_colors = {
                            'EXIT_NOW': '#dc3545',
                            'EXIT_TODAY': '#fd7e14',
                            'PARTIAL_EXIT': '#ffc107',
                            'TIGHTEN_SL': '#17a2b8',
                            'HOLD': '#28a745',
                            'HOLD_STRONG': '#20c997',
                            'ADD_MORE': '#007bff'
                        }
                        ec = exit_colors.get(exit_decision['action'], '#6c757d')
                        
                        st.markdown(f"""
                        <div style='text-align:center;padding:15px;background:{ec}20;
                                    border-radius:10px;border-left:4px solid {ec};'>
                            <h2 style='margin:0;color:{ec};'>{exit_decision['action'].replace('_', ' ')}</h2>
                            <p style='margin:5px 0;'>Score: {exit_decision['score']}/100</p>
                            <p style='margin:5px 0;'>Confidence: {exit_decision['confidence']}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # SHOW AGREEMENT STATUS
                        basic_action = r['overall_action']
                        smart_action = exit_decision['action']
                        
                        # Check if both brains agree
                        exit_actions = ['EXIT', 'EXIT_EARLY', 'EXIT_NOW', 'EXIT_TODAY']
                        hold_actions = ['HOLD', 'HOLD_STRONG', 'HOLD_EXTEND', 'TRAIL_SL', 'ADD_MORE']
                        
                        basic_is_exit = basic_action in exit_actions
                        smart_is_exit = smart_action in ['EXIT_NOW', 'EXIT_TODAY']
                        basic_is_hold = basic_action in hold_actions
                        smart_is_hold = smart_action in ['HOLD', 'HOLD_STRONG', 'ADD_MORE']
                        
                        if (basic_is_exit and smart_is_exit) or (basic_is_hold and smart_is_hold):
                            st.markdown("""
                            <div style='text-align:center;padding:5px;background:#d4edda;
                                        border-radius:5px;margin-top:10px;'>
                                ✅ <strong>Both analyses AGREE</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style='text-align:center;padding:5px;background:#fff3cd;
                                        border-radius:5px;margin-top:10px;'>
                                ⚠️ <strong>Mixed signals</strong><br>
                                <small>Levels: {basic_action} | Momentum: {smart_action}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with exit_col2:
                        st.markdown(f"**{exit_decision['summary']}**")
                        
                        if exit_decision['reasons']:
                            for reason in exit_decision['reasons'][:5]:
                                st.caption(f"• {reason}")
                        
                        if exit_decision['exit_price']:
                            st.warning(f"💰 Suggested exit price: ₹{exit_decision['exit_price']:,.2f}")
                
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # NEW: INTRADAY SL MONITOR DISPLAY
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                if r.get('intraday_sl_data'):
                    isd = r['intraday_sl_data']
                    
                    if isd['sl_touched']:
                        st.divider()
                        if isd.get('close_below_sl', False):
                            # Closing price below SL = confirmed breach
                            st.error(f"""
                            🚨 **SL BREACHED (Closing Price Confirmed)**
                            - Day Low: ₹{isd['day_low']:,.2f} | Day High: ₹{isd['day_high']:,.2f}
                            - Current: ₹{isd['current']:,.2f} (BELOW SL ₹{r['stop_loss']:,.2f})
                            - ⚡ EXIT REQUIRED for swing trade
                            """)
                        elif isd['recovered']:
                            st.warning(f"""
                            ⚠️ **SL WICK (Touched but Recovered)**
                            - Day Low: ₹{isd['day_low']:,.2f} wicked below SL ₹{r['stop_loss']:,.2f}
                            - Price recovered to: ₹{isd['current']:,.2f}
                            - 📊 For swing trading: This is normal. Watch closing price.
                            - ⚡ Consider tightening SL if this happens repeatedly
                            """)
                        elif isd['sl_breached']:
                            st.error(f"""
                            🚨 **SL CURRENTLY BREACHED**
                            - Current: ₹{isd['current']:,.2f} vs SL ₹{r['stop_loss']:,.2f}
                            - Wait for close or exit now based on conviction
                            """)
                
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # NEW: MARKET-AWARE SL ADJUSTMENT DISPLAY
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                if market_health and 'df' in r:
                    atr_val = r.get('atr', r['current_price'] * 0.02)
                    market_adjusted_sl, sl_adjustments = adjust_sl_for_market_conditions(
                        r['stop_loss'], r['entry_price'], r['current_price'],
                        r['position_type'], market_health, atr_val
                    )
                    
                    if sl_adjustments:
                        st.divider()
                        st.markdown("##### 🌐 Market-Aware SL Adjustment")
                        
                        for adj in sl_adjustments:
                            st.info(f"• {adj}")
                        
                        if market_adjusted_sl != r['stop_loss']:
                            st.warning(
                                f"📌 **Suggested SL:** ₹{r['stop_loss']:,.2f} → "
                                f"₹{market_adjusted_sl:,.2f} (market conditions)"
                            )
                
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # NEW: SMART TARGET DISPLAY
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                if 'df' in r:
                    smart_targets = calculate_smart_target(
                        r['df'], r['entry_price'], r['current_price'],
                        r['position_type'], r['target1'], market_health
                    )
                    
                    with st.expander("🎯 Smart Dynamic Targets", expanded=False):
                        t_col1, t_col2, t_col3 = st.columns(3)
                        
                        with t_col1:
                            st.metric("Smart T1", f"₹{smart_targets['smart_target1']:,.2f}")
                        with t_col2:
                            st.metric("Smart T2", f"₹{smart_targets['smart_target2']:,.2f}")
                        with t_col3:
                            st.metric("Smart T3", f"₹{smart_targets['smart_target3']:,.2f}")
                        
                        st.caption(f"📊 {smart_targets['target_note']} | "
                                  f"Multiplier: {smart_targets['momentum_multiplier']:.2f} | "
                                  f"Confidence: {smart_targets['confidence']:.0f}%")
                        
                        # Fibonacci levels
                        st.markdown("**Fibonacci Extensions:**")
                        fib_cols = st.columns(len(smart_targets['fib_targets']))
                        for i, (level, price) in enumerate(smart_targets['fib_targets'].items()):
                            with fib_cols[i]:
                                st.caption(f"Fib {level}\n₹{price:,.2f}")
                
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # NEW: POSITION PRIORITY DISPLAY
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                priority_data = calculate_position_priority(r, market_health)
                
                if priority_data['urgency_score'] >= 50:
                    st.divider()
                    st.markdown(f"""
                    <div style='background:#f8d7da; padding:10px; border-radius:8px; 
                                border-left:4px solid #dc3545; margin:5px 0;'>
                        <strong>{priority_data['priority']} - Urgency: {priority_data['urgency_score']}/100</strong><br>
                        {'<br>'.join(f'• {f}' for f in priority_data['factors'])}
                    </div>
                    """, unsafe_allow_html=True)

                    # ✅ GAP 4: CHART PATTERN DETECTION                    
                if 'df' in r:
                    detected_patterns = detect_chart_patterns(r['df'], r['current_price'])
                    
                    if detected_patterns:
                        st.divider()
                        st.markdown("##### 📐 Detected Patterns")
                        
                        for pattern in detected_patterns:
                            signal_color = "#28a745" if pattern['signal'] == 'BULLISH' else "#dc3545"
                            
                            st.markdown(f"""
                            <div style='background:{signal_color}20; padding:10px; border-radius:8px; 
                                        border-left:3px solid {signal_color}; margin:5px 0;'>
                                <strong>{pattern['icon']} {pattern['name']}</strong> ({pattern['signal']} - {pattern['strength']})<br>
                                <small>{pattern['description']}</small><br>
                                <em>→ {pattern['action']}</em>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Row 3: Partial Exits
                if r['partial_exits']['triggered_count'] > 0:
                    st.divider()
                    st.markdown("##### 📊 Partial Exit Levels")
                    
                    pe_cols = st.columns(4)
                    for idx, pe in enumerate(r['partial_exits']['recommendations'][:4]):
                        with pe_cols[idx]:
                            status_color = "#28a745" if pe['status'] == 'TRIGGERED' else "#6c757d"
                            st.markdown(f"""
                            <div style='padding:10px;background:{status_color}20;border-radius:8px;text-align:center;border-left:3px solid {status_color};'>
                                <strong>₹{pe['level']:,.2f}</strong><br>
                                <small>{pe['reason']}</small><br>
                                <span style='color:{status_color};'>{pe['status']}</span>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Row 4: Alerts
                if r['alerts']:
                    st.divider()
                    st.markdown("##### ⚠️ Alerts & Recommendations")
                    for alert in r['alerts']:
                        if alert['priority'] == 'CRITICAL':
                            st.error(f"**{alert['type']}**: {alert['message']}\n\n**⚡ Action: {alert['action']}**")
                        elif alert['priority'] == 'HIGH':
                            st.warning(f"**{alert['type']}**: {alert['message']}\n\n**⚡ Action: {alert['action']}**")
                        elif alert['priority'] == 'MEDIUM':
                            st.info(f"**{alert['type']}**: {alert['message']}\n\n**Action: {alert['action']}**")
                        else:
                            st.caption(f"ℹ️ {alert['type']}: {alert['message']}")
                
                # Recommendation Box
                # Recommendation Box (UNIFIED - same signal everywhere)
                rec_colors = {
                    'EXIT': 'critical-box', 'EXIT_EARLY': 'critical-box',
                    'EXIT_TODAY': 'critical-box', 'PARTIAL_EXIT': 'warning-box',
                    'TIGHTEN_SL': 'warning-box', 'WATCH': 'warning-box',
                    'BOOK_PROFITS': 'success-box', 'HOLD_EXTEND': 'info-box',
                    'TRAIL_SL': 'success-box', 'HOLD': 'info-box',
                    'HOLD_STRONG': 'info-box', 'ADD_MORE': 'success-box',
                    'MOVE_SL_BREAKEVEN': 'info-box'
                }
                rec_class = rec_colors.get(r['overall_action'], 'info-box')
                
                # Use unified_label (same label shown in header, here, and everywhere)
                unified = r.get('unified_label', r['overall_action'].replace('_', ' '))
                
                st.markdown(f"""
                <div class="{rec_class}">
                    📌 FINAL RECOMMENDATION: {unified}
                </div>
                """, unsafe_allow_html=True)
                
                # Show what each brain thinks (transparency)
                exit_dec = r.get('exit_decision', {})
                if exit_dec:
                    with st.expander("🔍 Decision Breakdown (How this was decided)", expanded=False):
                        brain_col1, brain_col2, brain_col3 = st.columns(3)
                        
                        with brain_col1:
                            st.markdown("**📊 Level Analysis**")
                            st.caption(f"SL Hit: {'🔴 YES' if r['sl_hit'] else '🟢 No'}")
                            st.caption(f"T1 Hit: {'✅ YES' if r['target1_hit'] else '⬜ No'}")
                            st.caption(f"T2 Hit: {'✅ YES' if r['target2_hit'] else '⬜ No'}")
                            st.caption(f"SL Risk: {r['sl_risk']}%")
                            st.caption(f"Trail Needed: {'✅' if r['should_trail'] else '⬜'}")
                        
                        with brain_col2:
                            st.markdown("**🧠 Momentum Analysis**")
                            st.caption(f"Exit Score: {exit_dec.get('score', 'N/A')}/100")
                            st.caption(f"Exit Action: {exit_dec.get('action', 'N/A')}")
                            st.caption(f"RSI: {r['rsi']:.1f}")
                            st.caption(f"MACD: {r['macd_signal']}")
                            st.caption(f"Momentum: {r['momentum_score']:.0f}/100")
                        
                        with brain_col3:
                            st.markdown("**🎯 Final Decision**")
                            st.caption(f"Status: {r['overall_status']}")
                            st.caption(f"Action: {r['overall_action']}")
                            st.caption(f"Label: {unified}")
                            
                            # Explain WHY
                            if r['sl_hit']:
                                st.caption("📌 Reason: Stop Loss was hit")
                            elif r['target2_hit']:
                                st.caption("📌 Reason: Both targets achieved")
                            elif exit_dec.get('action') in ['EXIT_NOW', 'EXIT_TODAY']:
                                st.caption("📌 Reason: Momentum analysis override")
                            elif r['should_trail']:
                                st.caption("📌 Reason: Trailing stop recommended")
                            elif r['target1_hit']:
                                st.caption("📌 Reason: Target 1 achieved")
                            else:
                                st.caption("📌 Reason: Standard level monitoring")
                # ============================================================================
                # ✅ NEW: AUTO-UPDATE GOOGLE SHEETS SECTION
                # ============================================================================
                                # ============================================================================
                # 🤖 FULLY AUTOMATED TRADING ENGINE
                # ============================================================================
                
                st.divider()
                st.markdown("##### 🤖 Auto-Trading Engine")
                
                auto_col1, auto_col2, auto_col3 = st.columns([2, 2, 1])
                
                # ==================================================================
                # COLUMN 1: AUTOMATED SL MANAGEMENT
                # ==================================================================
                with auto_col1:
                    st.markdown("**🛡️ Stop Loss Management**")
                    
                    daily_sl_count = get_daily_update_count(r['ticker'], "sl")
                    
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # AUTO-TRAIL SL (Profit Protection)
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    if r['should_trail'] and r['trail_stop'] != r['stop_loss']:
                        if is_sl_improvement(r['stop_loss'], r['trail_stop'], r['position_type']):
                            already_trailed, trail_time = get_persistent_guard(
                                r['ticker'], "sl_trail"
                            )
                            
                            if not already_trailed and daily_sl_count < MAX_SL_UPDATES_PER_DAY:
                                    with st.spinner(f"🔄 Auto-Trailing SL for {r['ticker']}..."):
                                        success, msg = update_sheet_stop_loss(
                                            r['ticker'],
                                            r['trail_stop'],
                                            r.get('trail_reason', 'Auto-trail based on profit'),
                                            should_send_email=True,
                                            email_settings=settings["email_settings"],
                                            result=r
                                        )
                                        if success:
                                            st.success(
                                                f"✅ Trail SL: ₹{r['stop_loss']:.2f} "
                                                f"→ ₹{r['trail_stop']:.2f}"
                                            )
                                            set_persistent_guard(
                                                r['ticker'], "sl_trail",
                                                f"₹{r['trail_stop']:.2f}"
                                            )
                                            r['stop_loss'] = r['trail_stop']
                                        else:
                                            st.warning(f"⚠️ {msg}")
                            elif already_trailed:
                                st.info(
                                    f"✅ SL trailed to ₹{r['trail_stop']:.2f} "
                                    f"today ({trail_time})"
                                )
                            else:
                                st.caption(f"⏸️ Max {MAX_SL_UPDATES_PER_DAY} SL updates/day reached")
                        else:
                            st.caption("ℹ️ Trail SL not better than current - skipping")
                    
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # AUTO MARKET-AWARE SL (3-Step Rule)
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    if market_health and 'df' in r:
                        atr_val = r.get('atr', r['current_price'] * 0.02)
                        calculated_sl, sl_adjustments = adjust_sl_for_market_conditions(
                            r['stop_loss'], r['entry_price'], r['current_price'],
                            r['position_type'], market_health, atr_val
                        )
                        
                        should_auto_update_sl = False
                        auto_reason = ""
                        
                        # STEP 1: VIX CHECK
                        if market_health.get('vix', 15) > 25:
                            should_auto_update_sl = True
                            auto_reason = f"⚡ VIX {market_health['vix']:.1f} (>25) - Auto-tightening"
                        
                        # STEP 2: SUPPORT PROXIMITY CHECK (Safety Override)
                        dist_to_support = abs(
                            (r['current_price'] - r['support']) / r['current_price']
                        ) * 100
                        
                        if dist_to_support < 1.0 and r['position_type'] == 'LONG':
                            should_auto_update_sl = False
                            auto_reason = "🛡️ Near support - skipping SL tightening"
                        
                        dist_to_resistance = abs(
                            (r['resistance'] - r['current_price']) / r['current_price']
                        ) * 100
                        
                        if dist_to_resistance < 1.0 and r['position_type'] == 'SHORT':
                            should_auto_update_sl = False
                            auto_reason = "🛡️ Near resistance - skipping SL tightening"
                        
                        # STEP 3: P&L CHECK (Override back to YES for extreme cases)
                        if r['pnl_percent'] < -2:
                            should_auto_update_sl = True
                            auto_reason = f"⚠️ Loss protection ({r['pnl_percent']:.1f}%) - Auto-tightening"
                        elif r['pnl_percent'] > 15:
                            should_auto_update_sl = True
                            auto_reason = f"💰 Big profit ({r['pnl_percent']:.1f}%) - Auto-locking"
                        
                        # SAFETY CHECK 1: New SL must be BETTER than old SL
                        if should_auto_update_sl and not is_sl_improvement(
                            r['stop_loss'], calculated_sl, r['position_type']
                        ):
                            should_auto_update_sl = False
                            auto_reason = "ℹ️ Calculated SL not better than current - no change"
                        
                        # SAFETY CHECK 2: Daily limit not exceeded
                        if should_auto_update_sl and daily_sl_count >= MAX_SL_UPDATES_PER_DAY:
                            should_auto_update_sl = False
                            auto_reason = f"⏸️ Daily limit ({MAX_SL_UPDATES_PER_DAY}) reached"
                        
                        # SAFETY CHECK 3: SL change not too extreme (max 5% move)
                        if should_auto_update_sl:
                            sl_change_pct = abs(calculated_sl - r['stop_loss']) / r['stop_loss'] * 100
                            if sl_change_pct > 5.0:
                                should_auto_update_sl = False
                                auto_reason = f"🚫 SL change too large ({sl_change_pct:.1f}%) - manual review needed"
                        
                        # EXECUTE
                        if should_auto_update_sl and calculated_sl != r['stop_loss']:
                            already_market_sl, market_sl_time = get_persistent_guard(
                                r['ticker'], "market_sl"
                            )
                            if not already_market_sl:
                                with st.spinner(f"🌐 Market-adjusting SL for {r['ticker']}..."):
                                    success, msg = update_sheet_stop_loss(
                                        r['ticker'],
                                        calculated_sl,
                                        f"Market-Aware: {auto_reason}",
                                        should_send_email=True,
                                        email_settings=settings["email_settings"],
                                        result=r
                                    )
                                    
                                    if success:
                                        st.success(
                                            f"✅ Market SL: ₹{r['stop_loss']:.2f} → "
                                            f"₹{calculated_sl:.2f} ({auto_reason})"
                                        )
                                        set_persistent_guard(
                                            r['ticker'], "market_sl",
                                            f"₹{calculated_sl:.2f}"
                                        )
                                        r['stop_loss'] = calculated_sl
                                    else:
                                        st.warning(f"⚠️ {msg}")
                            else:
                                st.caption(f"🤖 Market SL checked this hour: ₹{calculated_sl:.2f}")
                        else:
                            if sl_adjustments:
                                st.caption(f"ℹ️ {auto_reason}")
                    
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # AUTO-UPDATE TARGET
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    if r['target1_hit'] and r['upside_score'] >= 60 and r['new_target'] != r['target2']:
                        update_key = f"target_updated_{r['ticker']}_{get_ist_now().strftime('%Y%m%d')}"
                        
                        if update_key not in st.session_state:
                            with st.spinner(f"🎯 Extending target for {r['ticker']}..."):
                                success, msg = update_sheet_target(
                                    r['ticker'],
                                    r['new_target'],
                                    2,
                                    f"Auto-extended: Upside score {r['upside_score']}%",
                                    should_send_email=True,
                                    email_settings=settings["email_settings"],
                                    result=r
                                )
                                
                                if success:
                                    st.success(f"✅ Target extended to ₹{r['new_target']:.2f}")
                                    set_persistent_guard(
                                        r['ticker'], "target_extend",
                                        f"₹{r['new_target']:.2f}"
                                    )
                                    r['target2'] = r['new_target']
                                else:
                                    st.warning(f"⚠️ {msg}")
                        else:
                            st.info(f"✅ Target extended to ₹{r['new_target']:.2f} today")
                
                # ==================================================================
                # COLUMN 2: AUTOMATED EXIT ENGINE
                # ==================================================================
                with auto_col2:
                    st.markdown("**🚪 Exit Management**")
                    
                    exit_decision = r.get('exit_decision', {})
                    daily_exits = get_daily_auto_exit_count()
                    
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # AUTO-EXIT: SL HIT (Highest Priority)
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    if r['sl_hit']:
                        already_exited, exit_time = get_persistent_guard(
                            r['ticker'], "auto_exit"
                        )
                        if not already_exited:
                            st.error(f"🚨 SL HIT - Auto-exiting {r['ticker']}")
                            
                            # SAFETY: Verify price before exit
                            price_valid, actual_price, verify_msg = verify_price_before_exit(
                                r['ticker'], r['current_price'], tolerance_pct=3.0
                            )
                            
                            if price_valid:
                                with st.spinner(f"🚪 Auto-closing {r['ticker']} (SL Hit)..."):
                                    success, msg = mark_position_inactive(
                                        r['ticker'],
                                        actual_price,
                                        r['pnl_amount'],
                                        "Auto-Exit: Stop Loss Hit",
                                        should_send_email=True,
                                        email_settings=settings["email_settings"],
                                        result=r
                                    )
                                    
                                    if success:
                                        st.success(f"✅ {msg}")
                                        log_trade(
                                            r['ticker'], r['entry_price'], actual_price,
                                            r['quantity'], r['position_type'],
                                            "Auto-Exit: Stop Loss Hit"
                                        )
                                        set_persistent_guard(
                                            r['ticker'], "auto_exit",
                                            f"SL Hit ₹{actual_price:.2f}"
                                        )
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Auto-exit failed: {msg}")
                            else:
                                st.warning(f"⚠️ Price verification failed: {verify_msg}")
                                st.warning("Manual exit required - price data unreliable")
                        else:
                            st.info("✅ SL exit processed today")
                    
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # AUTO-EXIT: TARGET 2 HIT
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    elif r['target2_hit']:
                        exit_hash = f"auto_exit_{r['ticker']}_{get_ist_now().strftime('%Y%m%d')}"
                        
                        if exit_hash not in st.session_state:
                            # Only auto-exit T2 if upside is LIMITED
                            if r['upside_score'] < 50:
                                st.success(f"🎯 T2 HIT + Low upside - Auto-booking {r['ticker']}")
                                
                                price_valid, actual_price, verify_msg = verify_price_before_exit(
                                    r['ticker'], r['current_price'], tolerance_pct=3.0
                                )
                                
                                if price_valid:
                                    with st.spinner(f"💰 Auto-booking profits for {r['ticker']}..."):
                                        success, msg = mark_position_inactive(
                                            r['ticker'],
                                            actual_price,
                                            r['pnl_amount'],
                                            f"Auto-Exit: Target 2 Hit (Upside: {r['upside_score']}%)",
                                            should_send_email=True,
                                            email_settings=settings["email_settings"],
                                            result=r
                                        )
                                        
                                        if success:
                                            st.success(f"✅ {msg}")
                                            log_trade(
                                                r['ticker'], r['entry_price'], actual_price,
                                                r['quantity'], r['position_type'],
                                                "Auto-Exit: Target 2 Hit"
                                            )
                                            st.session_state[exit_hash] = True
                                            st.balloons()
                                            time.sleep(1)
                                            st.rerun()
                                        else:
                                            st.error(f"❌ {msg}")
                                else:
                                    st.warning(f"⚠️ {verify_msg} - Manual exit needed")
                            else:
                                st.info(
                                    f"🎯 T2 hit but upside is {r['upside_score']}% - "
                                    f"HOLDING for ₹{r['new_target']:.2f}"
                                )
                        else:
                            st.info("✅ Target exit processed today")
                    
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # AUTO-EXIT: SMART EXIT OVERRIDE
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    elif (exit_decision and 
                          exit_decision.get('action') == 'EXIT_NOW' and 
                          exit_decision.get('score', 100) <= 15):
                        
                        exit_hash = f"auto_exit_{r['ticker']}_{get_ist_now().strftime('%Y%m%d')}"
                        
                        if exit_hash not in st.session_state and daily_exits < MAX_AUTO_EXITS_PER_DAY:
                            st.error(
                                f"🧠 SMART EXIT triggered for {r['ticker']} "
                                f"(Score: {exit_decision['score']}/100)"
                            )
                            
                            # Show reasons before executing
                            for reason in exit_decision.get('reasons', [])[:3]:
                                st.caption(f"• {reason}")
                            
                            # SAFETY: Double verify price
                            price_valid, actual_price, verify_msg = verify_price_before_exit(
                                r['ticker'], r['current_price'], tolerance_pct=3.0
                            )
                            
                            if price_valid:
                                # SAFETY: Check emergency conditions are still true
                                # (Re-run smart exit with fresh data)
                                fresh_df = get_stock_data_safe(r['ticker'], period="6mo")
                                
                                if fresh_df is not None:
                                    fresh_exit = smart_exit_decision(r, market_health, fresh_df)
                                    
                                    # Only proceed if BOTH checks say EXIT
                                    if (fresh_exit['action'] == 'EXIT_NOW' and 
                                        fresh_exit['score'] <= 20):
                                        
                                        with st.spinner(
                                            f"🚪 Smart Auto-Exit for {r['ticker']}..."
                                        ):
                                            success, msg = mark_position_inactive(
                                                r['ticker'],
                                                actual_price,
                                                r['pnl_amount'],
                                                f"Smart Auto-Exit: {exit_decision['summary']}",
                                                should_send_email=True,
                                                email_settings=settings["email_settings"],
                                                result=r
                                            )
                                            
                                            if success:
                                                st.success(f"✅ {msg}")
                                                log_trade(
                                                    r['ticker'], r['entry_price'],
                                                    actual_price, r['quantity'],
                                                    r['position_type'],
                                                    f"Smart Auto-Exit: Score {exit_decision['score']}"
                                                )
                                                st.session_state[exit_hash] = True
                                                time.sleep(1)
                                                st.rerun()
                                            else:
                                                st.error(f"❌ {msg}")
                                    else:
                                        st.info(
                                            "ℹ️ Re-check passed - conditions improved. "
                                            "Holding position."
                                        )
                                        log_email(
                                            f"Smart exit aborted for {r['ticker']}: "
                                            f"Re-check score {fresh_exit['score']}"
                                        )
                                else:
                                    st.warning("⚠️ Cannot verify - skipping auto-exit")
                            else:
                                st.warning(f"⚠️ {verify_msg}")
                                st.warning("Manual review needed - price unreliable")
                        
                        elif daily_exits >= MAX_AUTO_EXITS_PER_DAY:
                            st.warning(
                                f"⏸️ Daily auto-exit limit ({MAX_AUTO_EXITS_PER_DAY}) reached. "
                                f"Manual exit needed."
                            )
                        else:
                            st.info("✅ Smart exit processed today")
                    
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # STATUS DISPLAY (when no exit needed)
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    else:
                        exit_score = exit_decision.get('score', 50) if exit_decision else 50
                        exit_action = exit_decision.get('action', 'HOLD') if exit_decision else 'HOLD'
                        
                        if exit_score <= 35:
                            st.warning(
                                f"👀 Watching: Exit score {exit_score}/100 "
                                f"({exit_action.replace('_', ' ')})"
                            )
                        elif exit_score >= 70:
                            st.success(
                                f"💪 Strong: Exit score {exit_score}/100 "
                                f"({exit_action.replace('_', ' ')})"
                            )
                        else:
                            st.caption(
                                f"🤖 Monitoring: Exit score {exit_score}/100 "
                                f"({exit_action.replace('_', ' ')})"
                            )
                    
                    st.divider()
                    
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # MANUAL OVERRIDE (Always Available)
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    with st.expander("✋ Manual Override", expanded=False):
                        manual_exit_price = st.number_input(
                            f"Exit Price",
                            min_value=0.01,
                            value=float(round_to_tick_size(r['current_price'])),
                            step=0.05,
                            key=f"manual_price_{r['ticker']}",
                            help="Override auto-exit with custom price"
                        )
                        
                        manual_reason = st.selectbox(
                            "Reason",
                            [
                                "Manual Override - Strategy Change",
                                "Manual Override - Risk Management",
                                "Manual Override - News/Event",
                                "Manual Override - Capital Reallocation",
                                "Manual Override - Other"
                            ],
                            key=f"manual_reason_{r['ticker']}"
                        )
                        
                        # Calculate P&L
                        if r['position_type'] == 'LONG':
                            manual_pnl = (manual_exit_price - r['entry_price']) * r['quantity']
                        else:
                            manual_pnl = (r['entry_price'] - manual_exit_price) * r['quantity']
                        
                        pnl_color = "#28a745" if manual_pnl >= 0 else "#dc3545"
                        st.markdown(
                            f"P&L: <span style='color:{pnl_color};font-weight:bold;'>"
                            f"₹{manual_pnl:+,.2f}</span>",
                            unsafe_allow_html=True
                        )
                        
                        confirm = st.checkbox(
                            f"Confirm manual exit for {r['ticker']}",
                            key=f"confirm_{r['ticker']}"
                        )
                        
                        if st.button(
                            "✋ Execute Manual Exit",
                            key=f"manual_exit_{r['ticker']}",
                            disabled=not confirm,
                            use_container_width=True
                        ):
                            with st.spinner("Processing..."):
                                manual_result = r.copy()
                                manual_result['current_price'] = manual_exit_price
                                manual_result['pnl_amount'] = manual_pnl
                                
                                success, msg = mark_position_inactive(
                                    r['ticker'], manual_exit_price, manual_pnl,
                                    manual_reason,
                                    should_send_email=True,
                                    email_settings=settings["email_settings"],
                                    result=manual_result
                                )
                                
                                if success:
                                    st.success(f"✅ {msg}")
                                    log_trade(
                                        r['ticker'], r['entry_price'],
                                        manual_exit_price, r['quantity'],
                                        r['position_type'], manual_reason
                                    )
                                    if manual_pnl > 0:
                                        st.balloons()
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(f"❌ {msg}")
                
                # ==================================================================
                # COLUMN 3: ENGINE STATUS
                # ==================================================================
                with auto_col3:
                    st.markdown("**📊 Status**")
                    
                    # Daily counters
                    daily_sl = get_daily_update_count(r['ticker'], "sl")
                    daily_exits_total = get_daily_auto_exit_count()
                    
                    st.caption(f"SL updates: {daily_sl}/{MAX_SL_UPDATES_PER_DAY}")
                    st.caption(f"Auto-exits: {daily_exits_total}/{MAX_AUTO_EXITS_PER_DAY}")
                    
                    # Engine status indicator
                    is_open_now, _, _, _ = is_market_hours()
                    
                    if is_open_now:
                        st.markdown("""
                        <div style='text-align:center;padding:8px;background:#d4edda;
                                    border-radius:8px;'>
                            🟢 <strong>ACTIVE</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style='text-align:center;padding:8px;background:#f8f9fa;
                                    border-radius:8px;'>
                            ⏸️ <strong>PAUSED</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Safety indicators
                    st.caption("---")
                    st.caption("Safety Checks:")
                    st.caption("✅ Price verify")
                    st.caption("✅ SL direction")
                    st.caption("✅ Daily limits")
                    st.caption("✅ Double-check")
                    st.caption("✅ Max change 5%")           
    
    # =========================================================================
    # TAB 2: CHARTS
    # =========================================================================
    with tab2:
        selected_stock = st.selectbox(
            "Select Stock for Chart", [r['ticker'] for r in results]
        )
        selected_result = next(
            (r for r in results if r['ticker'] == selected_stock), None
        )
        
        if selected_result:
            # Fetch chart data separately (not cached with analysis)
            df = get_stock_data_safe(selected_stock, period="6mo")
            
            if df is not None and not df.empty:
            
                # Candlestick Chart
                fig = go.Figure()
                
                fig.add_trace(go.Candlestick(
                    x=df['Date'], open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'], name='Price'
                ))
                
                # Add moving averages
                df['SMA20'] = df['Close'].rolling(20).mean()
                df['EMA9'] = df['Close'].ewm(span=9).mean()
                df['SMA50'] = df['Close'].rolling(50).mean()
                
                fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA20'], mode='lines',
                                        name='SMA 20', line=dict(color='orange', width=1)))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA9'], mode='lines',
                                        name='EMA 9', line=dict(color='purple', width=1)))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], mode='lines',
                                        name='SMA 50', line=dict(color='blue', width=1, dash='dot')))
                
                # Add levels
                fig.add_hline(y=selected_result['entry_price'], line_dash="dash",
                            line_color="blue", annotation_text="Entry")
                fig.add_hline(y=selected_result['stop_loss'], line_dash="dash",
                            line_color="red", annotation_text="Stop Loss")
                fig.add_hline(y=selected_result['target1'], line_dash="dash",
                            line_color="green", annotation_text="Target 1")
                fig.add_hline(y=selected_result['target2'], line_dash="dot",
                            line_color="darkgreen", annotation_text="Target 2")
                fig.add_hline(y=selected_result['support'], line_dash="dot",
                            line_color="orange", annotation_text="Support")
                fig.add_hline(y=selected_result['resistance'], line_dash="dot",
                            line_color="purple", annotation_text="Resistance")
                
                if selected_result['should_trail']:
                    fig.add_hline(y=selected_result['trail_stop'], line_dash="dash",
                                line_color="cyan", annotation_text="Trail SL", line_width=2)
                
                fig.update_layout(
                    title=f"{selected_stock} - Price Chart with Levels",
                    height=500,
                    xaxis_rangeslider_visible=False,
                    xaxis_title="Date",
                    yaxis_title="Price (₹)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # RSI and MACD Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    rsi_series = calculate_rsi(df['Close'])
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=df['Date'], y=rsi_series, mode='lines',
                                                name='RSI', line=dict(color='purple')))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray")
                    fig_rsi.update_layout(title="RSI (14)", height=250, yaxis_range=[0, 100])
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
                with col2:
                    macd, signal, histogram = calculate_macd(df['Close'])
                    colors = ['green' if h >= 0 else 'red' for h in histogram]
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Bar(x=df['Date'], y=histogram, name='Histogram',
                                            marker_color=colors))
                    fig_macd.add_trace(go.Scatter(x=df['Date'], y=macd, mode='lines',
                                                name='MACD', line=dict(color='blue', width=1)))
                    fig_macd.add_trace(go.Scatter(x=df['Date'], y=signal, mode='lines',
                                                name='Signal', line=dict(color='orange', width=1)))
                    fig_macd.update_layout(title="MACD", height=250)
                    st.plotly_chart(fig_macd, use_container_width=True)
                
                # Volume Chart
                fig_vol = go.Figure()
                vol_colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red'
                            for i in range(len(df))]
                fig_vol.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume',
                                        marker_color=vol_colors))
                fig_vol.update_layout(title="Volume", height=200)
                st.plotly_chart(fig_vol, use_container_width=True)
    
    # =========================================================================
    # TAB 3: ALERTS
    # =========================================================================
    with tab3:
        st.subheader("🔔 All Alerts")
        
        all_alerts = []
        for r in results:
            for alert in r['alerts']:
                 all_alerts.append({
                    'Ticker': r['ticker'],
                    'Priority': alert['priority'],
                    'Type': alert['type'],
                    'Message': alert['message'],
                    'Action': alert['action'],
                    'P&L': f"{r['pnl_percent']:+.2f}%",
                    'SL Risk': f"{r['sl_risk']}%",
                    'Final': r.get('unified_label', r['overall_action'].replace('_', ' '))
                })
        
        if all_alerts:
            # Sort by priority
            priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
            all_alerts_sorted = sorted(all_alerts, key=lambda x: priority_order.get(x['Priority'], 4))
            
            df_alerts = pd.DataFrame(all_alerts_sorted)
            
            # Color code by priority
            def highlight_priority(row):
                if row['Priority'] == 'CRITICAL':
                    return ['background-color: #f8d7da'] * len(row)
                elif row['Priority'] == 'HIGH':
                    return ['background-color: #fff3cd'] * len(row)
                elif row['Priority'] == 'MEDIUM':
                    return ['background-color: #d1ecf1'] * len(row)
                return [''] * len(row)
            
            st.dataframe(df_alerts.style.apply(highlight_priority, axis=1),
                        use_container_width=True, hide_index=True)
            
            # Summary by priority
            st.markdown("### Alert Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                critical = sum(1 for a in all_alerts if a['Priority'] == 'CRITICAL')
                st.metric("🔴 Critical", critical)
            with col2:
                high = sum(1 for a in all_alerts if a['Priority'] == 'HIGH')
                st.metric("🟠 High", high)
            with col3:
                medium = sum(1 for a in all_alerts if a['Priority'] == 'MEDIUM')
                st.metric("🟡 Medium", medium)
            with col4:
                low = sum(1 for a in all_alerts if a['Priority'] == 'LOW')
                st.metric("🟢 Low", low)
        else:
            st.success("✅ No alerts! All positions are healthy.")
            st.balloons()
    
    # =========================================================================
    # TAB 4: MTF ANALYSIS
    # =========================================================================
    with tab4:
        st.subheader("📉 Multi-Timeframe Analysis")
        
        if not settings['enable_multi_timeframe']:
            st.warning("⚠️ Multi-Timeframe Analysis is disabled. Enable it in the sidebar settings.")
        else:
            for r in results:
                with st.expander(f"{r['ticker']} - MTF Alignment: {r['mtf_alignment']}%",
                                expanded=(r['mtf_alignment'] < 50)):
                    
                    if r['mtf_signals']:
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            alignment_color = "#28a745" if r['mtf_alignment'] >= 60 else "#ffc107" if r['mtf_alignment'] >= 40 else "#dc3545"
                            st.markdown(f"""
                            <div style='text-align:center;padding:20px;background:#f8f9fa;border-radius:10px;'>
                                <h1 style='color:{alignment_color};margin:0;'>{r['mtf_alignment']}%</h1>
                                <p style='margin:5px 0;'>Timeframe Alignment</p>
                                <p style='font-size:0.8em;color:#666;'>{r['mtf_recommendation']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            for tf, signal in r['mtf_signals'].items():
                                details = r['mtf_details'].get(tf, {})
                                sig_color = "🟢" if signal == "BULLISH" else "🔴" if signal == "BEARISH" else "⚪"
                                
                                strength = details.get('strength', 'Unknown')
                                rsi_tf = details.get('rsi', 0)
                                
                                st.markdown(f"""
                                **{tf}:** {sig_color} {signal} ({strength})
                                - RSI: {rsi_tf:.1f} | Above SMA20: {'✅' if details.get('above_sma20') else '❌'} | 
                                EMA Bullish: {'✅' if details.get('ema_bullish') else '❌'} |
                                MACD: {'📈' if details.get('macd_bullish') else '📉'}
                                """)
                    else:
                        st.warning("MTF data not available for this stock")
    
    # =========================================================================
    # TAB 5: PORTFOLIO RISK
    # =========================================================================
    with tab5:
        display_portfolio_risk_dashboard(portfolio_risk, sector_analysis)
        
        st.divider()
        
        # Sector Analysis
        display_sector_analysis(sector_analysis)
        
        st.divider()
        
        # Correlation Analysis
        display_correlation_analysis(results, settings['enable_correlation'])
    
    # =========================================================================
    # TAB 6: PERFORMANCE
    # =========================================================================
    with tab6:
        display_performance_dashboard()
        # =========================================================================
    # =========================================================================
    # TAB 7: DETAILS
    # =========================================================================
    with tab7:
        st.subheader("📋 Complete Analysis Data")
        
        details_data = []
        for r in results:
            details_data.append({
                'Ticker': r['ticker'],
                'Type': r['position_type'],
                'Entry': f"₹{r['entry_price']:,.2f}",
                'Current': f"₹{r['current_price']:,.2f}",
                'P&L %': f"{r['pnl_percent']:+.2f}%",
                'P&L ₹': f"₹{r['pnl_amount']:+,.0f}",
                'SL': f"₹{r['stop_loss']:,.2f}",
                'SL Risk': f"{r['sl_risk']}%",
                'Momentum': f"{r['momentum_score']:.0f}",
                'RSI': f"{r['rsi']:.1f}",
                'MACD': r['macd_signal'],
                'Volume': r['volume_signal'].replace('_', ' '),
                'Support': f"₹{r['support']:,.2f}",
                'Resistance': f"₹{r['resistance']:,.2f}",
                'Trail SL': f"₹{r['trail_stop']:,.2f}" if r['should_trail'] else '-',
                'MTF Align': f"{r['mtf_alignment']}%" if r['mtf_signals'] else 'N/A',
                'R:R': f"1:{r['risk_reward_ratio']:.2f}",
                'Holding': f"{r['holding_days']}d" if r['holding_days'] > 0 else '-',
                'Status': r['overall_status'],
                'Action': r.get('unified_label', r['overall_action'].replace('_', ' '))
            })
        
        df_details = pd.DataFrame(details_data)
        
        # Color code by status
        def highlight_status(row):
            status = row['Status']
            if status == 'CRITICAL':
                return ['background-color: #f8d7da'] * len(row)
            elif status == 'WARNING':
                return ['background-color: #fff3cd'] * len(row)
            elif status in ['SUCCESS', 'GOOD']:
                return ['background-color: #d4edda'] * len(row)
            elif status == 'OPPORTUNITY':
                return ['background-color: #d1ecf1'] * len(row)
            return [''] * len(row)
        
        st.dataframe(df_details.style.apply(highlight_status, axis=1),
                    use_container_width=True, hide_index=True)
        
        # Export option
        csv_data = df_details.to_csv(index=False)
        st.download_button(
            "📥 Download Analysis as CSV",
            csv_data,
            file_name=f"portfolio_analysis_{ist_now.strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    # =========================================================================
    # AUTO REFRESH
    # =========================================================================
    st.divider()
    
    if settings['auto_refresh']:
        if is_open:
            if HAS_AUTOREFRESH:
                count = st_autorefresh(
                    interval=settings['refresh_interval'] * 1000,
                    limit=None,
                    key="portfolio_autorefresh"
                )
                st.caption(f"🔄 Auto-refresh active | Interval: {settings['refresh_interval']}s | Count: {count}")
            else:
                st.caption(f"⏱️ Auto-refresh requires streamlit-autorefresh package")
                st.caption("💡 Install: `pip install streamlit-autorefresh`")
                
                # Manual refresh button as fallback
                if st.button("🔄 Refresh Now", key="manual_refresh"):
                    st.cache_data.clear()
                    st.rerun()
        else:
            st.caption(f"⏸️ Auto-refresh paused - {market_status}: {market_msg}")
    else:
        st.caption("🔄 Auto-refresh disabled. Click 'Refresh' button to update.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<p style='text-align:center;color:#666;font-size:0.8em;'>"
        f"Smart Portfolio Monitor v6.0 | Last updated: {ist_now.strftime('%H:%M:%S')} IST | "
        f"Positions: {len(results)} | API Calls: {st.session_state.api_call_count}"
        f"</p>",
        unsafe_allow_html=True
    )


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()