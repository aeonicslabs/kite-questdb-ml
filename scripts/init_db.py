"""
Initialize QuestDB schema.
Run this once, or on startup — QuestDB CREATE TABLE IF NOT EXISTS is idempotent.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of where script is invoked from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2

from config.settings import get_settings

logger = logging.getLogger(__name__)


def init_questdb() -> None:
    cfg = get_settings().questdb
    schema_path = Path(__file__).parent.parent / "storage" / "schema.sql"

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    sql = schema_path.read_text()
    # Split on semicolons and execute each statement
    statements = [s.strip() for s in sql.split(";") if s.strip() and not s.strip().startswith("--")]

    conn = psycopg2.connect(cfg.pg_dsn)
    conn.autocommit = True
    cur = conn.cursor()

    created = 0
    for stmt in statements:
        if not stmt:
            continue
        try:
            cur.execute(stmt)
            if "CREATE TABLE" in stmt.upper():
                created += 1
        except Exception as e:
            # QuestDB may throw on duplicate table in some versions — log and continue
            logger.debug("Schema statement warning: %s", e)

    cur.close()
    conn.close()
    logger.info("QuestDB schema initialized (%d tables checked/created)", created)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_questdb()
    print("QuestDB schema initialized successfully.")
