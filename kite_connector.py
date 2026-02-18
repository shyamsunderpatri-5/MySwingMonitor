"""
Zerodha Kite Connector
=======================
Bridge between Smart Portfolio Monitor and Zerodha Trading API.

IMPORTANT: Install kiteconnect first:
    pip install kiteconnect

This file handles:
- Session management (login/logout)
- Order placement (exit orders)
- Order modification (trailing stop loss)
- Position verification
- Safety checks (margins, market hours)
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
from zoneinfo import ZoneInfo


logger = logging.getLogger(__name__)

# Try importing kiteconnect
try:
    from kiteconnect import KiteConnect
    from kiteconnect import exceptions as kite_exceptions
    HAS_KITE = True
except ImportError:
    HAS_KITE = False
    logger.warning("kiteconnect not installed. Run: pip install kiteconnect")

# Import our config
try:
    from config_kite import (
        KITE_API_KEY, KITE_API_SECRET,
        save_token, load_token, clear_token, is_configured,
        OrderType, TransactionType, ProductType, Exchange, OrderStatus
    )
except ImportError:
    logger.error("config_kite.py not found! Create it first.")
    HAS_KITE = False


# ============================================================================
# KITE SESSION MANAGER
# ============================================================================

class KiteSession:
    """
    Manages the Kite Connect session lifecycle.
    
    Usage:
        session = KiteSession()
        success, msg = session.initialize()
        
        if success:
            session.place_exit_order("RELIANCE", 10, "SELL")
    """
    
    def __init__(self):
        self.kite = None
        self.is_connected = False
        self.last_error = ""
        self.order_log = []       # Track all orders placed
        self.daily_order_count = 0
        self.max_daily_orders = 20  # Safety limit
    
    def initialize(self) -> Tuple[bool, str]:
        """
        Initialize Kite session using saved token.
        
        Returns: (success, message)
        """
        if not HAS_KITE:
            return False, "kiteconnect not installed. Run: pip install kiteconnect"
        
        if not is_configured():
            return False, "API credentials not configured in config_kite.py"
        
        try:
            self.kite = KiteConnect(api_key=KITE_API_KEY)
            
            # Load saved token
            access_token, is_valid, token_msg = load_token()
            
            if not is_valid or not access_token:
                return False, f"Token issue: {token_msg}"
            
            # Set the access token
            self.kite.set_access_token(access_token)
            
            # Verify connection by fetching profile
            try:
                profile = self.kite.profile()
                user_name = profile.get("user_name", "Unknown")
                self.is_connected = True
                logger.info(f"Kite connected: {user_name}")
                return True, f"Connected as {user_name}. {token_msg}"
            except kite_exceptions.TokenException:
                clear_token()
                self.is_connected = False
                return False, "Token expired or invalid. Please login again."
            except Exception as e:
                self.is_connected = False
                return False, f"Connection failed: {str(e)}"
        
        except Exception as e:
            self.is_connected = False
            self.last_error = str(e)
            return False, f"Initialization failed: {str(e)}"
    
    def login_with_request_token(self, request_token: str) -> Tuple[bool, str]:
        """
        Exchange request token for access token.
        Called once per day after user logs in via Kite URL.
        
        Returns: (success, message)
        """
        if not HAS_KITE:
            return False, "kiteconnect not installed"
        
        if not is_configured():
            return False, "API credentials not configured"
        
        try:
            self.kite = KiteConnect(api_key=KITE_API_KEY)
            
            # Generate session (exchange request_token for access_token)
            data = self.kite.generate_session(
                request_token,
                api_secret=KITE_API_SECRET
            )
            
            access_token = data["access_token"]
            
            # Save token to file
            save_token(access_token, request_token)
            
            # Set token on kite object
            self.kite.set_access_token(access_token)
            
            # Verify
            profile = self.kite.profile()
            user_name = profile.get("user_name", "Unknown")
            self.is_connected = True
            
            logger.info(f"Login successful: {user_name}")
            return True, f"Login successful! Welcome {user_name}."
        
        except kite_exceptions.TokenException as e:
            return False, f"Invalid request token: {str(e)}"
        except kite_exceptions.NetworkException as e:
            return False, f"Network error: {str(e)}"
        except Exception as e:
            return False, f"Login failed: {str(e)}"
    
    def logout(self) -> Tuple[bool, str]:
        """Logout and clear saved token"""
        try:
            if self.kite and self.is_connected:
                self.kite.invalidate_access_token()
            clear_token()
            self.is_connected = False
            self.kite = None
            return True, "Logged out successfully"
        except Exception as e:
            clear_token()
            self.is_connected = False
            return False, f"Logout error (token cleared anyway): {str(e)}"

    # ========================================================================
    # SAFETY CHECKS
    # ========================================================================
    
    def pre_flight_check(self, ticker: str, quantity: int, 
                         transaction_type: str) -> Tuple[bool, str]:
        """
        Run all safety checks before placing any order.
        
        Returns: (is_safe, reason)
        """
        if not self.is_connected or not self.kite:
            return False, "Not connected to Kite"
        
        errors = []
        
        # CHECK 1: Daily order limit
        if self.daily_order_count >= self.max_daily_orders:
            return False, f"Daily order limit ({self.max_daily_orders}) reached"
        
        # CHECK 2: Market hours
        now = datetime.now() + timedelta(hours=5, minutes=30)  # IST
        current_time = now.time()
        
        from datetime import time as dt_time
        market_open = dt_time(9, 15)
        market_close = dt_time(15, 30)
        
        if now.weekday() >= 5:
            return False, "Market closed (Weekend)"
        
        if current_time < market_open or current_time > market_close:
            return False, f"Market closed (Hours: 9:15-15:30 IST, Current: {current_time})"
        
        # CHECK 3: Valid quantity
        if quantity <= 0:
            return False, f"Invalid quantity: {quantity}"
        
        # CHECK 4: Check margins (for BUY orders)
        if transaction_type == TransactionType.BUY:
            try:
                margins = self.kite.margins()
                available = margins.get("equity", {}).get("available", {}).get("live_balance", 0)
                
                # Get approximate price
                quote = self.kite.quote(f"NSE:{ticker}")
                price = quote.get(f"NSE:{ticker}", {}).get("last_price", 0)
                
                required = price * quantity
                
                if available < required:
                    return False, (
                        f"Insufficient funds. "
                        f"Need: ₹{required:,.0f}, Available: ₹{available:,.0f}"
                    )
            except Exception as e:
                errors.append(f"Margin check warning: {str(e)}")
        
        # CHECK 5: Verify ticker exists
        try:
            quote = self.kite.quote(f"NSE:{ticker}")
            if not quote:
                return False, f"Ticker {ticker} not found on NSE"
            
            last_price = quote.get(f"NSE:{ticker}", {}).get("last_price", 0)
            if last_price <= 0:
                return False, f"Invalid price for {ticker}: ₹{last_price}"
        except Exception as e:
            return False, f"Cannot verify ticker {ticker}: {str(e)}"
        
        if errors:
            logger.warning(f"Pre-flight warnings: {errors}")
        
        return True, "All checks passed"
    
    def get_live_price(self, ticker: str) -> Tuple[float, str]:
        """
        Get live price from Zerodha (more reliable than yfinance).
        
        Returns: (price, status_message)
        """
        if not self.is_connected or not self.kite:
            return 0.0, "Not connected"
        
        try:
            quote = self.kite.quote(f"NSE:{ticker}")
            data = quote.get(f"NSE:{ticker}", {})
            
            price = data.get("last_price", 0)
            
            if price > 0:
                return float(price), "OK"
            else:
                return 0.0, "Invalid price received"
        except Exception as e:
            return 0.0, f"Error: {str(e)}"

    # ========================================================================
    # ORDER PLACEMENT
    # ========================================================================
    
    def place_exit_order(
        self,
        ticker: str,
        quantity: int,
        transaction_type: str = "SELL",
        order_type: str = "MARKET",
        price: float = 0,
        product: str = "CNC",
        tag: str = "SmartMonitor"
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Place an exit order on Zerodha.
        
        Args:
            ticker: Stock symbol (e.g., "RELIANCE")
            quantity: Number of shares
            transaction_type: "BUY" or "SELL"
            order_type: "MARKET" or "LIMIT"
            price: Required if order_type is "LIMIT"
            product: "CNC" (delivery) or "MIS" (intraday)
            tag: Order tag for identification
        
        Returns: (success, message, order_id)
        """
        if not self.is_connected or not self.kite:
            return False, "Not connected to Kite", None
        
        # Clean ticker
        ticker = ticker.replace(".NS", "").replace(".BO", "").upper().strip()
        
        # Pre-flight check
        is_safe, safety_msg = self.pre_flight_check(
            ticker, quantity, transaction_type
        )
        
        if not is_safe:
            logger.error(f"Pre-flight failed for {ticker}: {safety_msg}")
            return False, f"Safety check failed: {safety_msg}", None
        
        try:
            # Build order params
            order_params = {
                "exchange": Exchange.NSE,
                "tradingsymbol": ticker,
                "transaction_type": transaction_type,
                "quantity": quantity,
                "product": product,
                "order_type": order_type,
                "tag": tag
            }
            
            # Add price for LIMIT orders
            if order_type == OrderType.LIMIT and price > 0:
                order_params["price"] = price
            
            # Place order
            order_id = self.kite.place_order(
                variety="regular",
                **order_params
            )
            
            # Log the order
            self.daily_order_count += 1
            order_record = {
                "timestamp": datetime.now().isoformat(),
                "ticker": ticker,
                "type": transaction_type,
                "quantity": quantity,
                "order_type": order_type,
                "price": price,
                "order_id": str(order_id),
                "status": "PLACED"
            }
            self.order_log.append(order_record)
            
            logger.info(
                f"Order placed: {transaction_type} {quantity}x {ticker} "
                f"({order_type}) | ID: {order_id}"
            )
            
            return True, f"Order placed successfully. ID: {order_id}", str(order_id)
        
        except kite_exceptions.OrderException as e:
            error_msg = f"Order rejected: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, None
        
        except kite_exceptions.TokenException:
            self.is_connected = False
            clear_token()
            return False, "Session expired. Please login again.", None
        
        except kite_exceptions.NetworkException as e:
            return False, f"Network error: {str(e)}", None
        
        except kite_exceptions.InputException as e:
            return False, f"Invalid input: {str(e)}", None
        
        except Exception as e:
            return False, f"Unexpected error: {str(e)}", None
    
    def place_sl_order(
        self,
        ticker: str,
        quantity: int,
        trigger_price: float,
        transaction_type: str = "SELL",
        limit_price: float = 0,
        product: str = "CNC",
        tag: str = "SmartMonitor_SL"
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Place a Stop Loss order on Zerodha.
        
        For LONG positions: SELL SL order (triggers when price falls to trigger_price)
        For SHORT positions: BUY SL order (triggers when price rises to trigger_price)
        
        Args:
            ticker: Stock symbol
            quantity: Number of shares
            trigger_price: Price at which SL triggers
            transaction_type: "SELL" for long exits, "BUY" for short exits
            limit_price: If 0, uses SL-M (market). If set, uses SL (limit).
            product: "CNC" or "MIS"
            tag: Order tag
        
        Returns: (success, message, order_id)
        """
        if not self.is_connected or not self.kite:
            return False, "Not connected to Kite", None
        
        ticker = ticker.replace(".NS", "").replace(".BO", "").upper().strip()
        
        is_safe, safety_msg = self.pre_flight_check(
            ticker, quantity, transaction_type
        )
        
        if not is_safe:
            return False, f"Safety check failed: {safety_msg}", None
        
        try:
            order_params = {
                "exchange": Exchange.NSE,
                "tradingsymbol": ticker,
                "transaction_type": transaction_type,
                "quantity": quantity,
                "product": product,
                "trigger_price": trigger_price,
                "tag": tag
            }
            
            if limit_price > 0:
                # SL order (with limit price)
                order_params["order_type"] = OrderType.SL
                order_params["price"] = limit_price
            else:
                # SL-M order (market price on trigger)
                order_params["order_type"] = OrderType.SL_M
            
            order_id = self.kite.place_order(
                variety="regular",
                **order_params
            )
            
            self.daily_order_count += 1
            self.order_log.append({
                "timestamp": datetime.now().isoformat(),
                "ticker": ticker,
                "type": f"SL_{transaction_type}",
                "quantity": quantity,
                "trigger_price": trigger_price,
                "order_id": str(order_id),
                "status": "PLACED"
            })
            
            logger.info(
                f"SL Order placed: {ticker} trigger@₹{trigger_price} "
                f"qty:{quantity} | ID: {order_id}"
            )
            
            return True, f"SL Order placed. ID: {order_id}", str(order_id)
        
        except kite_exceptions.OrderException as e:
            return False, f"SL Order rejected: {str(e)}", None
        except kite_exceptions.TokenException:
            self.is_connected = False
            clear_token()
            return False, "Session expired. Login again.", None
        except Exception as e:
            return False, f"SL Order error: {str(e)}", None

    # ========================================================================
    # ORDER MODIFICATION (For Trailing SL)
    # ========================================================================
    
    def modify_stoploss(
        self,
        order_id: str,
        new_trigger_price: float,
        new_limit_price: float = 0,
        new_quantity: int = 0
    ) -> Tuple[bool, str]:
        """
        Modify an existing Stop Loss order (for trailing).
        
        Args:
            order_id: The Zerodha order ID to modify
            new_trigger_price: New trigger price for the SL
            new_limit_price: New limit price (if SL-L order)
            new_quantity: New quantity (if partial exit)
        
        Returns: (success, message)
        """
        if not self.is_connected or not self.kite:
            return False, "Not connected to Kite"
        
        try:
            modify_params = {
                "trigger_price": new_trigger_price
            }
            
            if new_limit_price > 0:
                modify_params["price"] = new_limit_price
                modify_params["order_type"] = OrderType.SL
            
            if new_quantity > 0:
                modify_params["quantity"] = new_quantity
            
            self.kite.modify_order(
                variety="regular",
                order_id=order_id,
                **modify_params
            )
            
            self.order_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "MODIFY_SL",
                "order_id": order_id,
                "new_trigger": new_trigger_price,
                "status": "MODIFIED"
            })
            
            logger.info(
                f"SL Modified: Order {order_id} → "
                f"trigger@₹{new_trigger_price}"
            )
            
            return True, f"SL updated to ₹{new_trigger_price:.2f}"
        
        except kite_exceptions.OrderException as e:
            return False, f"Modify failed: {str(e)}"
        except kite_exceptions.TokenException:
            self.is_connected = False
            return False, "Session expired"
        except Exception as e:
            return False, f"Modify error: {str(e)}"
    
    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """Cancel an open order"""
        if not self.is_connected or not self.kite:
            return False, "Not connected"
        
        try:
            self.kite.cancel_order(variety="regular", order_id=order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True, f"Order {order_id} cancelled"
        except Exception as e:
            return False, f"Cancel failed: {str(e)}"

    # ========================================================================
    # POSITION & ORDER QUERIES
    # ========================================================================
    
    def get_positions(self) -> Tuple[bool, Any, str]:
        """
        Get all current positions from Zerodha.
        
        Returns: (success, positions_list, message)
        """
        if not self.is_connected or not self.kite:
            return False, [], "Not connected"
        
        try:
            positions = self.kite.positions()
            net_positions = positions.get("net", [])
            
            # Filter to only stocks with quantity
            active = [
                p for p in net_positions
                if p.get("quantity", 0) != 0
            ]
            
            return True, active, f"Found {len(active)} active positions"
        except Exception as e:
            return False, [], f"Error fetching positions: {str(e)}"
    
    def get_orders(self) -> Tuple[bool, Any, str]:
        """Get all orders for today"""
        if not self.is_connected or not self.kite:
            return False, [], "Not connected"
        
        try:
            orders = self.kite.orders()
            return True, orders, f"Found {len(orders)} orders today"
        except Exception as e:
            return False, [], f"Error: {str(e)}"
    
    def get_order_status(self, order_id: str) -> Tuple[str, Dict]:
        """
        Get status of a specific order.
        
        Returns: (status_string, order_details_dict)
        """
        if not self.is_connected or not self.kite:
            return "ERROR", {"error": "Not connected"}
        
        try:
            order_history = self.kite.order_history(order_id)
            
            if order_history:
                latest = order_history[-1]  # Last update
                status = latest.get("status", "UNKNOWN")
                return status, latest
            
            return "NOT_FOUND", {}
        except Exception as e:
            return "ERROR", {"error": str(e)}
    
    def get_open_sl_orders(self, ticker: str = None) -> list:
        """
        Get all open SL orders, optionally filtered by ticker.
        
        Returns: list of open SL orders
        """
        if not self.is_connected or not self.kite:
            return []
        
        try:
            orders = self.kite.orders()
            
            sl_orders = [
                o for o in orders
                if o.get("order_type") in ["SL", "SL-M"]
                and o.get("status") in ["OPEN", "TRIGGER PENDING"]
            ]
            
            if ticker:
                ticker_clean = (
                    ticker.replace(".NS", "")
                    .replace(".BO", "")
                    .upper()
                    .strip()
                )
                sl_orders = [
                    o for o in sl_orders
                    if o.get("tradingsymbol") == ticker_clean
                ]
            
            return sl_orders
        except Exception as e:
            logger.error(f"Error fetching SL orders: {e}")
            return []

    # ========================================================================
    # PORTFOLIO VERIFICATION
    # ========================================================================
    
    def verify_position_exists(
        self, ticker: str, quantity: int, position_type: str
    ) -> Tuple[bool, str]:
        """
        Verify that a position exists in Zerodha before placing exit order.
        
        Returns: (exists, message)
        """
        if not self.is_connected:
            return False, "Not connected"
        
        ticker_clean = (
            ticker.replace(".NS", "")
            .replace(".BO", "")
            .upper()
            .strip()
        )
        
        try:
            success, positions, msg = self.get_positions()
            
            if not success:
                return False, msg
            
            for pos in positions:
                if pos.get("tradingsymbol") == ticker_clean:
                    actual_qty = pos.get("quantity", 0)
                    
                    if position_type == "LONG" and actual_qty > 0:
                        if actual_qty >= quantity:
                            return True, (
                                f"Position verified: {actual_qty} shares "
                                f"of {ticker_clean}"
                            )
                        else:
                            return False, (
                                f"Quantity mismatch: Sheet says {quantity}, "
                                f"Zerodha has {actual_qty}"
                            )
                    elif position_type == "SHORT" and actual_qty < 0:
                        if abs(actual_qty) >= quantity:
                            return True, (
                                f"Short position verified: "
                                f"{abs(actual_qty)} shares"
                            )
                        else:
                            return False, (
                                f"Quantity mismatch: Sheet says {quantity}, "
                                f"Zerodha has {abs(actual_qty)}"
                            )
            
            return False, f"No position found for {ticker_clean} in Zerodha"
        except Exception as e:
            return False, f"Verification error: {str(e)}"

    # ========================================================================
    # UTILITY
    # ========================================================================
    
    def get_order_log(self) -> list:
        """Get all orders placed by this session"""
        return self.order_log
    
    def reset_daily_count(self):
        """Reset daily order counter (call at start of each day)"""
        self.daily_order_count = 0
        self.order_log = []

    # ========================================================================
    # MARKET DATA - HISTORICAL & LIVE (replaces yfinance)
    # ========================================================================

    # ── Well-known NSE instrument tokens ─────────────────────────────────────
    KNOWN_TOKENS = {
        "NIFTY 50":  256265,   # ^NSEI equivalent
        "INDIA VIX": 264969,   # ^INDIAVIX equivalent
    }

    # ── Interval mapping: yfinance-style → Kite interval string ──────────────
    _KITE_INTERVAL = {
        "1m": "minute",  "5m":  "5minute",  "15m": "15minute",
        "30m": "30minute", "60m": "60minute", "1h":  "60minute",
        "1d": "day",     "1wk": "week",      "1mo": "month",
    }

    def _period_to_dates(self, period: str):
        """Convert a yfinance-style period string to (from_date, to_date) in IST."""
        _MAP = {
            "1d": 1,  "2d": 2,  "5d": 5,
            "1mo": 30, "3mo": 90, "6mo": 180,
            "1y": 365, "2y": 730,
        }
        days = _MAP.get(period, 90)
        # Use IST so date boundaries align with NSE trading calendar
        to_dt   = datetime.now(ZoneInfo("Asia/Kolkata"))
        from_dt = to_dt - timedelta(days=days)
        return from_dt, to_dt

    def get_instrument_token(self, symbol: str, exchange: str = "NSE") -> Optional[int]:
        """
        Resolve a trading symbol to its Kite instrument token.
        Token results are cached per symbol; full instrument lists are cached per
        exchange so the heavy download (thousands of rows) happens only once per
        session, not once per stock.
        """
        if not hasattr(self, "_token_cache"):
            self._token_cache: Dict[str, int] = {}
        # Cache the full instruments list per exchange to avoid repeated downloads
        if not hasattr(self, "_instruments_cache"):
            self._instruments_cache: Dict[str, list] = {}

        cache_key = f"{exchange}:{symbol}"
        if cache_key in self._token_cache:
            return self._token_cache[cache_key]

        # Return hard-coded tokens for indices immediately
        if symbol in self.KNOWN_TOKENS:
            self._token_cache[cache_key] = self.KNOWN_TOKENS[symbol]
            return self.KNOWN_TOKENS[symbol]

        if not self.is_connected or not self.kite:
            logger.warning("get_instrument_token: Not connected")
            return None

        try:
            # Download full instruments list once per exchange, then reuse
            if exchange not in self._instruments_cache:
                logger.info(f"Downloading instruments list for {exchange} (one-time)...")
                self._instruments_cache[exchange] = self.kite.instruments(exchange)

            for inst in self._instruments_cache[exchange]:
                if inst["tradingsymbol"] == symbol.upper():
                    token = inst["instrument_token"]
                    self._token_cache[cache_key] = token
                    return token
            logger.warning(f"Token not found for {exchange}:{symbol}")
            return None
        except Exception as e:
            logger.error(f"get_instrument_token error for {symbol}: {e}")
            return None

    def get_historical_data(
        self,
        instrument_token: int,
        from_date,
        to_date,
        interval: str = "day",
        continuous: bool = False,
    ) -> Optional[Any]:
        """
        Fetch OHLCV history from Kite and return as a pandas DataFrame
        with columns: Date, Open, High, Low, Close, Volume
        (compatible with the yfinance-style DataFrames used in the app).
        """
        try:
            import pandas as pd
            records = self.kite.historical_data(
                instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                continuous=continuous,
            )
            if not records:
                return None
            df = pd.DataFrame(records)
            # Kite returns: date, open, high, low, close, volume [, oi]
            df.rename(columns={
                "date": "Date", "open": "Open", "high": "High",
                "low": "Low", "close": "Close", "volume": "Volume",
            }, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            return df
        except Exception as e:
            logger.error(f"get_historical_data error (token={instrument_token}): {e}")
            return None

    def get_historical_data_by_symbol(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "day",
        exchange: str = "NSE",
    ) -> Optional[Any]:
        """
        Convenience wrapper: fetch OHLCV by symbol + period string.
        Returns a DataFrame or None on failure.
        """
        token = self.get_instrument_token(symbol, exchange)
        if token is None:
            return None
        from_dt, to_dt = self._period_to_dates(period)
        kite_interval = self._KITE_INTERVAL.get(interval, interval)
        return self.get_historical_data(token, from_dt, to_dt, kite_interval)

    def get_quote(self, instruments: list) -> Dict:
        """
        Fetch full quote for a list of instruments, e.g. ["NSE:RELIANCE", "NSE:TCS"].
        Returns the raw Kite quote dict.
        """
        if not self.is_connected or not self.kite:
            return {}
        try:
            return self.kite.quote(instruments)
        except Exception as e:
            logger.error(f"get_quote error: {e}")
            return {}

    def get_ltp(self, instruments: list) -> Dict:
        """
        Fetch last traded price for a list of instruments.
        Returns {instrument_key: {"instrument_token": .., "last_price": ..}}
        """
        if not self.is_connected or not self.kite:
            return {}
        try:
            return self.kite.ltp(instruments)
        except Exception as e:
            logger.error(f"get_ltp error: {e}")
            return {}

    def get_intraday_data(
        self, symbol: str, interval: str = "5minute", exchange: str = "NSE"
    ) -> Optional[Any]:
        """
        Fetch today's intraday candles for *symbol* at *interval*.
        Returns a DataFrame (same schema as get_historical_data) or None.
        """
        token = self.get_instrument_token(symbol, exchange)
        if token is None:
            return None
        today = datetime.now()
        from_dt = today.replace(hour=9, minute=0, second=0, microsecond=0)
        return self.get_historical_data(token, from_dt, today, interval)

    def get_realtime_price_kite(self, symbol: str, exchange: str = "NSE") -> Optional[Dict]:
        """
        Return real-time price snapshot for *symbol*.
        Returns: {price, high, low, volume, last_update, is_realtime} or None.
        """
        instrument_key = f"{exchange}:{symbol}"
        quote = self.get_quote([instrument_key])
        data = quote.get(instrument_key)
        if not data:
            return None
        ohlc = data.get("ohlc", {})
        return {
            "price": float(data.get("last_price", 0)),
            "high":  float(ohlc.get("high", 0)),
            "low":   float(ohlc.get("low", 0)),
            "volume": int(data.get("volume", 0)),
            "last_update": str(data.get("timestamp", datetime.now())),
            "is_realtime": True,
        }

    def get_market_index_data(self, index: str = "NIFTY 50") -> Optional[Any]:
        """
        Fetch 3-month daily history for a market index (NIFTY 50 or INDIA VIX).
        3 months (~65 trading days) ensures SMA50 calculations are valid.
        Returns a DataFrame or None.
        """
        token = self.KNOWN_TOKENS.get(index)
        if token is None:
            logger.warning(f"Unknown index: {index}")
            return None
        from_dt, to_dt = self._period_to_dates("3mo")   # 3mo → ~65 trading days ≥ 50
        return self.get_historical_data(token, from_dt, to_dt, "day")


# ============================================================================
# SINGLETON INSTANCE (Use this in your main app)
# ============================================================================

# Create a single instance that persists
kite_session = KiteSession()
