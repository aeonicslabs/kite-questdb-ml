"""
Kite Authentication — handles login flow and token persistence.

Flow:
  1. On first run: generates login URL, user pastes request_token
  2. Exchanges request_token for access_token via Kite REST
  3. Saves access_token to .env (or prints it for manual save)
  4. On subsequent runs: loads from env, validates by fetching profile

Run standalone:
    python -m ingestion.kite_auth
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from kiteconnect import KiteConnect

from config.settings import get_settings

logger = logging.getLogger(__name__)

ENV_FILE = Path(".env")


def get_authenticated_kite() -> KiteConnect:
    """
    Return an authenticated KiteConnect instance.
    Loads access_token from environment / .env file.
    Raises RuntimeError if not authenticated.
    """
    cfg = get_settings().kite
    kite = KiteConnect(api_key=cfg.api_key)

    if cfg.access_token:
        kite.set_access_token(cfg.access_token)
        try:
            profile = kite.profile()
            logger.info("Authenticated as: %s", profile.get("user_name", "unknown"))
            return kite
        except Exception as e:
            logger.warning("Access token invalid or expired: %s", e)
            raise RuntimeError(
                "Access token is invalid. Run `python -m ingestion.kite_auth` to re-login."
            )

    raise RuntimeError(
        "KITE_ACCESS_TOKEN not set. Run `python -m ingestion.kite_auth` to authenticate."
    )


def login_flow() -> None:
    """
    Interactive login flow. Run this once per day before market open.
    Zerodha access tokens expire daily at 6:00 AM IST.
    """
    cfg = get_settings().kite
    kite = KiteConnect(api_key=cfg.api_key)

    print("\n" + "=" * 60)
    print("Kite Authentication")
    print("=" * 60)
    print(f"\n1. Open this URL in your browser:\n")
    print(f"   {kite.login_url()}\n")
    print("2. Log in with your Zerodha credentials.")
    print("3. After redirect, copy the 'request_token' from the URL bar.")
    print("   URL looks like: http://127.0.0.1/?request_token=XXXXXX&action=login&status=success")
    print()

    request_token = input("Paste the request_token here: ").strip()

    try:
        session = kite.generate_session(
            request_token=request_token,
            api_secret=cfg.api_secret,
        )
        access_token = session["access_token"]
        kite.set_access_token(access_token)

        # Verify
        profile = kite.profile()
        print(f"\n✓ Authenticated as: {profile['user_name']} ({profile['user_id']})")
        print(f"  Access token: {access_token}\n")

        # Save to .env file
        _save_access_token(access_token)
        print(f"✓ Access token saved to {ENV_FILE}")
        print("\nYou can now start the pipeline:\n  python main.py\n")

    except Exception as e:
        print(f"\n✗ Authentication failed: {e}")
        sys.exit(1)


def _save_access_token(access_token: str) -> None:
    """Append/update KITE_ACCESS_TOKEN in .env file."""
    lines = []
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text().splitlines()

    # Remove existing access token line
    lines = [l for l in lines if not l.startswith("KITE_ACCESS_TOKEN=")]
    lines.append(f"KITE_ACCESS_TOKEN={access_token}")

    ENV_FILE.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    login_flow()
