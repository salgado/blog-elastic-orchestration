# Multi-Agent Orchestration with LLMs using Elasticsearch and LangGraph 

How to build AI systems with self-correction and long-term memory using the Reflection Pattern. 

## Prerequisites

- Python 3.10+
- Elasticsearch Serverless with ELSER deployed
- Ollama with llama3.1:8b model

## Setup

### 1. Install Dependencies

```bash
# Clone repository
git clone https://github.com/salgado/blog-elastic-orchestration.git
cd blog-elastic-orchestration

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit with your credentials
nano .env
```

Required variables:
```bash
ELASTIC_ENDPOINT=https://your-deployment.es.region.gcp.elastic-cloud.com:443
ELASTIC_API_KEY=your-base64-api-key
OLLAMA_MODEL=llama3.1:8b
```

## Multi-LLM Support

This POC supports multiple LLM providers:

**Local Development (Ollama):**
```bash
# In .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b

python elastic_reflection_poc.py "database connection timeout"
```

**Production (Anthropic Claude):**
```bash
# In .env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514
ANTHROPIC_API_KEY=sk-ant-your-key

python elastic_reflection_poc.py "database connection timeout"
```

### 3. Setup ELSER (Run Once)

```bash
python setup_elser_serverless.py
```

Expected output:
```
SETUP COMPLETE!
1. Inference Endpoint: 'elser-incident-analysis'
2. Index: 'incident-logs' with semantic_text field
3. Sample data: 15 incident logs
4. Semantic search: Tested and working
5. Hybrid search: Tested and working
```

### 4. Run POC

```bash
# Interactive mode
python elastic_reflection_poc.py

# Or with query
python elastic_reflection_poc.py "database connection timeout"
```

## Configuration

Edit `.env` to customize:

```bash
# Optional (defaults provided)
ELASTIC_INDEX_LOGS=incident-logs
ELASTIC_INDEX_MEMORY=agent-memory
MAX_REFLECTION_ITERATIONS=3
QUALITY_THRESHOLD=0.8

# LLM Configuration
# ==================

# LLM Provider: "ollama" (local) or "anthropic" (cloud)
LLM_PROVIDER=ollama

# Model name (used when LLM_PROVIDER=anthropic)
LLM_MODEL=claude-sonnet-4-20250514

# Ollama Configuration (local development - free)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Anthropic Configuration (production - requires API key)
# Get key from: https://console.anthropic.com/settings/keys
# ANTHROPIC_API_KEY=sk-ant-...
```

## Architecture

The system uses two Elasticsearch indices:

1. **`incident-logs`**: Stores incident logs with semantic search (`semantic_content` field)
   - Created by `setup_elser_serverless.py`
   - Used by SearchAgent for hybrid search (semantic + keyword)

2. **`agent-memory`**: Long-Term Memory (LTM) storing successful analyses
   - Created automatically at runtime by `elastic_config.py`
   - Also uses semantic search (`semantic_content` field) for concept-based retrieval
   - Used by AnalyserAgent to retrieve similar past solutions

Both indices use **hybrid search** (semantic ELSER + keyword BM25) for optimal retrieval.

## Files

```
blog-elastic-reflection/
├── .env.example              # Environment template
├── requirements.txt          # Dependencies
├── .gitignore               # Git ignore
├── README.md                # This file
├── elastic_config.py        # Elasticsearch connection
├── elastic_reflection_poc.py # Main POC
└── setup_elser_serverless.py # ELSER setup
```

## License

MIT License
