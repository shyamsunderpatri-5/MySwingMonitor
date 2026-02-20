"""
get_token.py  —  One-time manual token getter (safe version)
=============================================================
Use this ONLY on first setup to verify your credentials.
For daily use, auto_token_refresh.py handles everything automatically.

Run:  python get_token.py
"""

from config_kite import KITE_API_KEY, KITE_API_SECRET, save_token
from kiteconnect import KiteConnect

if not KITE_API_KEY or not KITE_API_SECRET:
    print("❌ KITE_API_KEY / KITE_API_SECRET not set in secrets or environment.")
    raise SystemExit(1)

kite = KiteConnect(api_key=KITE_API_KEY)

print("Open this URL in browser and login:")
print(kite.login_url())
print()

request_token = input("Paste request_token from URL: ").strip()
data          = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
access_token  = data["access_token"]

save_token(access_token, request_token)

print()
print("✅ Token saved to kite_token.json — auto_token_refresh.py will handle renewals from now on.")
