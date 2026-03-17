# Kite ML Trading Pipeline

Real-time options trading system for BankNifty/Nifty using Zerodha Kite WebSocket, QuestDB, and ensemble ML models.

## Architecture

```
Kite WebSocket (full mode)
    ↓ asyncio.Queue
QuestDB (ILP/HTTP) ← raw ticks, features, signals
    ↓
Feature Engine (numpy/polars)
    ↓
Inference Engine (XGBoost + LSTM ensemble)
    ↓
Paper Trading / Signal DB
    ↓
Grafana Dashboard
```

## Setup (Mac Mini M4)

### 1. Prerequisites
```bash
brew install docker
brew install python@3.12
```

### 2. Start infrastructure
```bash
docker compose up -d
# QuestDB console: http://localhost:9000
# Grafana:         http://localhost:3000  (admin/admin)
# MLflow:          http://localhost:5000
```

### 3. Python environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure credentials
```bash
cp .env.example .env
# Edit .env: add KITE_API_KEY and KITE_API_SECRET
```

### 5. Authenticate (once per day — tokens expire at 6 AM IST)
```bash
python -m ingestion.kite_auth
```

### 6. Initialize DB schema
```bash
python scripts/init_db.py
```

### 7. Run the pipeline
```bash
python main.py
```

## Running Tests
```bash
pytest tests/unit/ -v
```

## Project Structure
```
kite-questdb-ml/
├── config/           # Pydantic settings (env-driven)
├── ingestion/        # Kite auth, instrument manager, WebSocket worker
├── storage/          # QuestDB writer (ILP), schema SQL
├── features/         # Feature engineering engine (RingBuffer, indicators)
├── ml/
│   ├── models/       # XGBoost + LSTM classifiers
│   ├── training/     # Data fetch, label gen, MLflow tracking
│   └── inference/    # Live inference + ensemble + signal emission
├── scripts/          # init_db, utilities
├── tests/            # Unit tests (no external deps required)
├── docker/           # Grafana provisioning
├── main.py           # Entry point
└── docker-compose.yml
```

## Important Notes

### 20-depth data
Zerodha Kite Connect API does **not** provide 20-depth data programmatically (ToS restriction).
This pipeline uses 5-level depth from the `full` mode WebSocket, which includes:
- 5 bid/ask levels with price and quantity
- OI, OHLC, buy/sell quantities, exchange timestamps

### Paper trading
`APP_PAPER_TRADING=true` by default. All signals are logged to QuestDB's `signals` table.
No orders are placed until you explicitly implement execution and set `APP_PAPER_TRADING=false`.

### Apple Silicon (M4)
- PyTorch uses MPS (Metal) backend automatically
- QuestDB Docker image is native ARM64
- All dependencies support ARM64 natively

### Access token refresh
Zerodha access tokens expire every day at 6:00 AM IST.
Run `python -m ingestion.kite_auth` before each trading day.
