# claim-verifier
# Python 3.11 FastAPI project

## Project Layout

```
ClaimGraph/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── api/
│   │   └── routes/          # Route handlers (one file per domain)
│   ├── services/            # Business logic layer
│   ├── models/              # Pydantic schemas & ODM models
│   ├── db/                  # Database clients (Mongo, Qdrant, Redis)
│   └── scripts/             # One-off utility / ingestion scripts
├── workers/                 # Celery task definitions
├── frontend/                # Static / SPA frontend (future)
├── tests/                   # Pytest test suite
├── .env.example             # Environment variable template
├── docker-compose.yml       # Infrastructure services
└── requirements.txt         # Python dependencies
```

## Quick Start

### 1. Create & activate a Python 3.11 virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment

```powershell
Copy-Item .env.example .env
# Then edit .env with your actual credentials
```

### 4. Start infrastructure services

```powershell
docker compose up -d
```

### 5. Run the API server

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

## Infrastructure Services

| Service  | Port(s)       | Purpose                          |
|----------|---------------|----------------------------------|
| MongoDB  | 27017         | Document store for claims/papers |
| Redis    | 6379          | Celery broker & result backend   |
| GROBID   | 8070 / 8071   | PDF → structured XML parsing     |
| Qdrant   | 6333 / 6334   | Vector store for embeddings      |

## Celery Worker

```powershell
celery -A workers worker --loglevel=info
```
