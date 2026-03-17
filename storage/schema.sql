-- ============================================================
-- QuestDB Schema for Kite ML Trading Pipeline
-- Run this once on a fresh QuestDB instance.
-- QuestDB uses SQL with time-series extensions.
-- ============================================================

-- ----------------------------------------------------------------
-- 1. RAW TICKS TABLE
--    One row per WebSocket tick update from Kite
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ticks (
    ts                      TIMESTAMP,          -- exchange timestamp (nanosecond precision)
    received_at             TIMESTAMP,          -- local receive time
    instrument_token        LONG,               -- Kite instrument token
    trading_symbol          SYMBOL CAPACITY 2048 CACHE,
    exchange                SYMBOL CAPACITY 16 CACHE,
    last_price              DOUBLE,
    last_quantity           LONG,
    average_price           DOUBLE,
    volume                  LONG,
    buy_quantity            LONG,
    sell_quantity           LONG,
    open                    DOUBLE,
    high                    DOUBLE,
    low                     DOUBLE,
    close                   DOUBLE,             -- previous close
    change                  DOUBLE,             -- % change from prev close
    open_interest           LONG,
    oi_day_high             LONG,
    oi_day_low              LONG,
    -- 5-level bid depth (price, qty) stored flat — fastest for QuestDB
    bid1_price              DOUBLE,
    bid1_qty                LONG,
    bid2_price              DOUBLE,
    bid2_qty                LONG,
    bid3_price              DOUBLE,
    bid3_qty                LONG,
    bid4_price              DOUBLE,
    bid4_qty                LONG,
    bid5_price              DOUBLE,
    bid5_qty                LONG,
    ask1_price              DOUBLE,
    ask1_qty                LONG,
    ask2_price              DOUBLE,
    ask2_qty                LONG,
    ask3_price              DOUBLE,
    ask3_qty                LONG,
    ask4_price              DOUBLE,
    ask4_qty                LONG,
    ask5_price              DOUBLE,
    ask5_qty                LONG
) TIMESTAMP(ts)
PARTITION BY DAY
WAL
DEDUP UPSERT KEYS(ts, instrument_token);

-- ----------------------------------------------------------------
-- 2. ENGINEERED FEATURES TABLE
--    One row per feature-computation cycle (default: every 1s per instrument)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS features (
    ts                      TIMESTAMP,
    instrument_token        LONG,
    trading_symbol          SYMBOL CAPACITY 2048 CACHE,
    -- Order book features
    bid_ask_spread          DOUBLE,
    mid_price               DOUBLE,
    order_book_imbalance    DOUBLE,     -- (bid_vol - ask_vol) / (bid_vol + ask_vol)
    depth_imbalance_l3      DOUBLE,     -- top-3 levels OBI
    -- Price momentum
    price_change_1s         DOUBLE,
    price_change_5s         DOUBLE,
    price_change_30s        DOUBLE,
    vwap_deviation          DOUBLE,     -- (last_price - vwap) / vwap
    -- Volume features
    volume_rate             DOUBLE,     -- volume per second in last 30s
    buy_sell_ratio          DOUBLE,     -- buy_qty / sell_qty
    -- OI features (options only)
    oi_change_rate          DOUBLE,     -- delta OI / time
    -- Rolling OHLC
    high_30s                DOUBLE,
    low_30s                 DOUBLE,
    range_30s               DOUBLE,
    -- Technical indicators
    rsi_14                  DOUBLE,
    bb_upper                DOUBLE,
    bb_lower                DOUBLE,
    bb_width                DOUBLE,
    -- Options-specific
    iv_approx               DOUBLE,     -- approximate IV from price
    pcr_oi                  DOUBLE,     -- put/call OI ratio (populated for index)
    atm_distance_pct        DOUBLE      -- distance from ATM strike as % of spot
) TIMESTAMP(ts)
PARTITION BY DAY
WAL;

-- ----------------------------------------------------------------
-- 3. ML SIGNALS TABLE
--    One row per inference event
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS signals (
    ts                      TIMESTAMP,
    instrument_token        LONG,
    trading_symbol          SYMBOL CAPACITY 2048 CACHE,
    model_name              SYMBOL CAPACITY 64 CACHE,
    model_version           STRING,
    signal                  SYMBOL CAPACITY 8 CACHE,   -- BUY_CE | BUY_PE | HOLD
    confidence              DOUBLE,
    predicted_direction     SYMBOL CAPACITY 8 CACHE,   -- UP | DOWN | FLAT
    predicted_magnitude     DOUBLE,
    features_snapshot       STRING                      -- JSON of top features used
) TIMESTAMP(ts)
PARTITION BY DAY
WAL;

-- ----------------------------------------------------------------
-- 4. PAPER TRADES TABLE
--    Simulated order book for paper trading P&L
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS paper_trades (
    ts                      TIMESTAMP,
    trade_id                STRING,
    instrument_token        LONG,
    trading_symbol          SYMBOL CAPACITY 2048 CACHE,
    action                  SYMBOL CAPACITY 8 CACHE,    -- BUY | SELL
    quantity                LONG,
    entry_price             DOUBLE,
    exit_price              DOUBLE,
    pnl                     DOUBLE,
    status                  SYMBOL CAPACITY 16 CACHE,   -- OPEN | CLOSED | CANCELLED
    signal_confidence       DOUBLE,
    model_name              SYMBOL CAPACITY 64 CACHE
) TIMESTAMP(ts)
PARTITION BY DAY
WAL;

-- ----------------------------------------------------------------
-- 5. INSTRUMENT UNIVERSE TABLE
--    Daily snapshot of all subscribed instruments
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS instruments (
    ts                      TIMESTAMP,
    instrument_token        LONG,
    trading_symbol          STRING,
    name                    STRING,
    expiry                  STRING,
    strike                  DOUBLE,
    instrument_type         SYMBOL CAPACITY 8 CACHE,    -- CE | PE | EQ | INDEX
    exchange                SYMBOL CAPACITY 16 CACHE,
    lot_size                LONG,
    tick_size               DOUBLE
) TIMESTAMP(ts)
PARTITION BY DAY
WAL;
