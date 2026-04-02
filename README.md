# ML Duplicate Claim Detector

A FastAPI service that detects duplicate insurance claims using sentence embeddings and cosine similarity.

## How It Works

1. Claims are encoded into vector embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`)
2. When a new claim is submitted, its embedding is compared against all existing claims
3. If cosine similarity exceeds 0.75, the claim is flagged as a duplicate

## Project Structure

```
ml_duplicate_claim1/
├── data/
│   └── claims.csv          # Source claims dataset
├── model/
│   └── train_model.py      # Generates embeddings from claims.csv
└── api/
    └── server.py           # FastAPI server
```

## Setup on a New System

**Requirements:** Python 3.9+

### 1. Clone the repo

```bash
git clone <repo-url>
cd ml_duplicate_claim1
```

### 2. Create a virtual environment

**Mac / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the `all-MiniLM-L6-v2` model (~90 MB). This is automatic.

### 4. Generate embeddings

Run once (or any time `claims.csv` changes):

```bash
cd model
python train_model.py
cd ..
```

This creates `model/claims_embeddings.npy`.

### 5. Start the API server

```bash
cd api
uvicorn server:app --reload
```

The server starts at `http://localhost:8000`.

## API Endpoints

### `GET /check?text=<claim text>`

Check if a claim is a duplicate.

#### Example 1 — Duplicate of Claim 1 (Car Accident)

Claim 6 in `claims.csv` is a paraphrase of Claim 1. To verify:

```bash
curl "http://localhost:8000/check?text=My+car+was+hit+on+the+highway+causing+damage+to+the+front+bumper+and+I+sustained+minor+injuries"
```

**Response:**
```json
{
  "is_duplicate": true,
  "similarity_score": 0.8004,
  "matched_claim": {
    "id": 1,
    "title": "Car Accident",
    "description": "Vehicle collision on highway resulting in front bumper damage and minor injuries"
  }
}
```

#### Example 2 — Duplicate of Claim 3 (Flood Damage)

Claim 7 in `claims.csv` is a paraphrase of Claim 3. To verify:

```bash
curl "http://localhost:8000/check?text=Heavy+rain+flooded+our+basement+and+ruined+all+the+furniture+and+home+appliances"
```

**Response:**
```json
{
  "is_duplicate": true,
  "similarity_score": 0.906,
  "matched_claim": {
    "id": 3,
    "title": "Flood Damage",
    "description": "Heavy rainfall caused basement flooding destroying furniture and appliances"
  }
}
```

### `GET /claims`

List all claims in the database.

### `GET /health`

Health check — returns status and total claim count.

## UI

Once the server is running, open `http://localhost:8000` to use the web interface:

- **Claims table** — lists all claims, with new ones highlighted
- **Check panel** — pre-filled with Claim 6 and Claim 7; click **Check** on each to see duplicate detection results live

## Interactive Docs

Once the server is running, visit `http://localhost:8000/docs` for the auto-generated Swagger UI.
