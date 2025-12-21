# Three Database Architecture: How They Work Together

## Overview

The system uses **three different databases** to store different representations of the same message data. Each database is optimized for a specific type of search or query:

1. **PostgreSQL** - Primary database (source of truth)
2. **Elasticsearch** - Keyword/BM25 search index
3. **ChromaDB** - Vector embeddings for semantic search

## Why Three Databases?

Each database serves a different purpose and provides different search capabilities:

```
┌─────────────────────────────────────────────────────────────────┐
│                         INCOMING MESSAGE                         │
│      "Hey team, starting OAuth2 implementation today..."        │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Data Ingestion       │
                    │   (Single Write)       │
                    └────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
    ┌───────────────────────┐   ┌───────────────────────┐
    │  Store in PostgreSQL  │   │  Store in             │
    │  (Source of Truth)    │   │  Elasticsearch        │
    │                       │   │  (Keyword Index)      │
    └───────────────────────┘   └───────────────────────┘
                    │
                    │
                    ▼
    ┌───────────────────────┐
    │  Generate Embedding   │
    │  Store in ChromaDB    │
    │  (Vector Search)      │
    └───────────────────────┘
```

## Database Comparison

| Database       | What It Stores | What It's Used For | Example Query |
|----------------|----------------|-------------------|---------------|
| **PostgreSQL** | Full message data + metadata | Primary storage, filtering, joins | "Get all messages from #platform channel in December" |
| **Elasticsearch** | Content + basic metadata | BM25 keyword search, exact term matching | "Find messages containing 'OAuth' or 'authentication'" |
| **ChromaDB** | Vector embeddings (768-dim) | Semantic similarity search | "Find messages similar to 'user login problems'" |

## Detailed Breakdown

### 1. PostgreSQL (Primary Database)

**Purpose**: Source of truth for all message data

**What it stores**:
```sql
messages:
  - id (integer, primary key)
  - source_id (foreign key to sources table)
  - content (full text content)
  - author (email/username)
  - channel (channel name)
  - timestamp (when message was created)
  - external_id (ID in source system like Slack)
  - thread_id (for threading)
  - message_metadata (JSON: reactions, mentions, etc.)
  - embedding_id (reference to ChromaDB: "msg_123")

sources:
  - id, type, name, config
```

**Use cases**:
- Storing complete message data with all metadata
- Filtering by author, channel, date range
- Joining messages with sources
- Providing the "enrichment" data after search results come back
- Tracking what's been indexed (via embedding_id)

**Example**:
```python
# After search returns message IDs [5, 12, 8], fetch full details
SELECT * FROM messages WHERE id IN (5, 12, 8)
```

---

### 2. Elasticsearch (Keyword Search Index)

**Purpose**: Fast BM25-based keyword matching

**What it stores**:
```json
{
  "message_id": "5",              // STRING (to match ES convention)
  "content": "Starting OAuth2 implementation...",
  "source": "slack",
  "channel": "#auth-team",
  "author": "charlie@company.com",
  "timestamp": "2025-12-14T18:58:32",
  "metadata": {...}
}
```

**Use cases**:
- **BM25 keyword search**: Find documents containing specific terms
- **Exact matching**: "OAuth", "authentication", "bug"
- **Boolean queries**: "OAuth AND security", "bug NOT fixed"
- **Phrase matching**: "authorization code flow"
- **Term frequency analysis**: How often does "OAuth" appear?

**How BM25 works**:
- **Term Frequency (TF)**: How many times does "OAuth" appear in this document?
- **Inverse Document Frequency (IDF)**: How rare is "OAuth" across all documents?
- **Length Normalization**: Penalize very long documents
- **Score = IDF * (TF adjusted for doc length)**

**Example**:
```bash
# User searches for "OAuth"
# Elasticsearch returns: {message_id: "5", score: 2.82}
# Then PostgreSQL fetches full details for message ID 5
```

---

### 3. ChromaDB (Vector Embeddings)

**Purpose**: Semantic similarity search

**What it stores**:
```json
{
  "id": "msg_5",                    // Embedding ID
  "embedding": [0.123, -0.456, ...], // 768-dimensional vector
  "document": "Starting OAuth2 implementation...",
  "metadata": {
    "message_id": 5,                 // PostgreSQL ID
    "source": "slack",
    "channel": "#auth-team",
    "author": "charlie@company.com"
  }
}
```

**Use cases**:
- **Semantic similarity**: Find messages with similar meaning
- **Handles synonyms**: "authentication" matches "login", "sign-in"
- **Conceptual matching**: "OAuth security" matches "authorization vulnerabilities"
- **Paraphrasing**: Different words, same meaning

**How it works**:
```
Query: "user login problems"
  ↓
Generate embedding: [0.234, -0.567, 0.891, ...]  (768 numbers)
  ↓
Compare with all message embeddings using cosine similarity
  ↓
Return top matches:
  - "Authentication failing after password reset" (score: 0.89)
  - "Users can't sign in on mobile" (score: 0.85)
  - "Login screen stuck on loading" (score: 0.81)
```

---

## How They Work Together: Search Flow

### Example Query: "OAuth security"

```
USER QUERY: "OAuth security"
│
├─── HYBRID SEARCH STRATEGY (Default)
│    │
│    ├─── Path 1: BM25 (Elasticsearch)
│    │    1. Search Elasticsearch for "OAuth security"
│    │    2. Get results: [msg_5: score=2.8, msg_12: score=2.1]
│    │    3. Extract message IDs: [5, 12]
│    │
│    ├─── Path 2: Semantic (ChromaDB)
│    │    1. Generate embedding for "OAuth security"
│    │    2. Search ChromaDB for similar embeddings
│    │    3. Get results: [msg_8: score=0.85, msg_5: score=0.79]
│    │    4. Extract message IDs: [8, 5]
│    │
│    ├─── Path 3: Fusion (Combine Results)
│    │    1. Reciprocal Rank Fusion (RRF):
│    │       - msg_5: 1/(60+1) + 1/(60+2) = 0.0323  (in both!)
│    │       - msg_12: 1/(60+2) = 0.0161
│    │       - msg_8: 1/(60+1) = 0.0164
│    │    2. Ranked by fused score: [5, 8, 12]
│    │
│    └─── Path 4: Enrichment (PostgreSQL)
│         1. Fetch full details: SELECT * FROM messages WHERE id IN (5,8,12)
│         2. JOIN with sources table
│         3. Return enriched results with all metadata
│
└─── FINAL RESULT:
     [
       {
         "message_id": 5,
         "content": "Starting OAuth2 implementation...",
         "author": "charlie@company.com",
         "channel": "#auth-team",
         "timestamp": "2025-12-14T18:58:32",
         "score": 0.0323,
         "source": "slack"
       },
       ...
     ]
```

## Data Consistency

### Question: Do they always have the same data?

**Short answer**: They should, but they might temporarily diverge.

**Why they might differ**:
1. **New message arrives**: Written to PostgreSQL first, then indexed to ES + ChromaDB
2. **Indexing delay**: If indexing fails, message exists in Postgres but not in ES/Chroma
3. **Data cleared manually**: Someone might delete ES index but not Postgres
4. **Development/testing**: Different services might be restarted at different times

**How we keep them in sync**:
```python
# In scripts/generate_mock_data.py (lines 310-332)

# Step 1: Insert to PostgreSQL (source of truth)
await db.commit()  # 24 messages saved

# Step 2: Fetch from PostgreSQL
result = await db.execute(select(Message).where(...))
all_messages = result.scalars().all()

# Step 3: Generate embeddings and store in ChromaDB
total_embeddings = populate_vector_store(all_messages)  # 24 embeddings

# Step 4: Index to Elasticsearch
total_indexed = populate_elasticsearch(all_messages)  # 24 documents

# Result: All three databases have 24 messages
```

### Verification Commands

```bash
# Check PostgreSQL
docker compose exec postgres psql -U coordination_user -d coordination \
  -c "SELECT COUNT(*) FROM messages;"
# Output: 24

# Check Elasticsearch
curl -s http://localhost:9200/messages/_count | jq .count
# Output: 24

# Check ChromaDB
docker compose exec api python -c \
  "from src.db.vector_store import get_vector_store; \
   vs = get_vector_store(); \
   print(f'ChromaDB count: {vs.get_collection_count()}')"
# Output: ChromaDB count: 24
```

## Why This Architecture?

### Alternative 1: Just use PostgreSQL

**Problem**:
- No semantic search (can't find synonyms)
- Slow full-text search on large datasets
- Can't do similarity matching

### Alternative 2: Just use Elasticsearch

**Problem**:
- No semantic search
- Harder to model relationships (joins)
- Vector search support is newer/less mature

### Alternative 3: Just use ChromaDB

**Problem**:
- No keyword search (can't search for exact term "OAuth")
- No BM25 scoring
- Not optimized for structured queries (filtering, joins)

### Our Solution: Use All Three! 🎯

**Benefits**:
- **Best of all worlds**: Keyword + semantic + structured queries
- **Specialized tools**: Each database does what it's best at
- **Flexibility**: Can use different strategies for different queries
- **Performance**: Each DB is optimized for its use case

**Tradeoff**:
- More complexity to maintain
- Need to keep them in sync
- More infrastructure to run

But for a coordination gap detection system that needs rich, multi-faceted search, this is the right architecture!

## Summary Table

| Question | Answer |
|----------|--------|
| **Do they contain the same data?** | Same messages, different representations |
| **Which is the source of truth?** | PostgreSQL |
| **Can I delete one?** | No - each serves a specific purpose |
| **How are they kept in sync?** | Write to all three when ingesting data |
| **What if they get out of sync?** | Use indexing scripts to rebuild ES/Chroma from Postgres |
| **Why not just one database?** | No single DB excels at keyword + semantic + structured queries |

## Real-World Example

Imagine searching for messages about authentication issues:

**Query**: "users can't login"

**PostgreSQL alone**:
- Returns: Messages with exact words "users", "can't", "login"
- Misses: "authentication failing", "sign-in broken"

**Elasticsearch alone**:
- Returns: Messages with keywords "users", "login" (BM25 ranked)
- Misses: "password reset not working" (semantically related)

**ChromaDB alone**:
- Returns: Semantically similar messages (including synonyms)
- Misses: Messages with exact term "login" but different context

**All three together (Hybrid)**:
- Returns: Best of all approaches
- Ranks exact keyword matches high (BM25)
- Also finds semantically similar messages
- Can filter by channel, author, date (PostgreSQL)
- **Result**: Comprehensive, relevant results! ✨
