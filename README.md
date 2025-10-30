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

### 3. Setup ELSER (Run Once)

```bash
python setup_elser_serverless.py
```

Expected output:
```
SETUP COMPLETE!
1. Inference Endpoint: 'elser-incident-analysis'
2. Index: 'incident-logs' with semantic_text field
3. Sample data: 5 incident logs
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
```

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
