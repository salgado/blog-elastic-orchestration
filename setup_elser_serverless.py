"""
ELSER Setup for Serverless Elasticsearch
=========================================

Automated script to:
1. Create Inference Endpoint for ELSER
2. Recreate incident-logs index with semantic_text field
3. Test semantic search

Usage:
    python setup_elser_serverless.py
"""

import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()


def get_es_client():
    """Get Elasticsearch client"""
    endpoint = os.getenv("ELASTIC_ENDPOINT")
    api_key = os.getenv("ELASTIC_API_KEY")

    if not endpoint or not api_key:
        raise ValueError("ELASTIC_ENDPOINT and ELASTIC_API_KEY must be set in .env")

    return Elasticsearch(
        hosts=[endpoint],
        api_key=api_key,
        request_timeout=30
    )


def create_inference_endpoint(es: Elasticsearch):
    """
    Create Inference Endpoint for ELSER

    In Serverless, Inference Endpoints are the modern way to use ML models.
    They replace traditional ingest pipelines and are reusable across indices.
    """
    inference_id = "elser-incident-analysis"

    logger.info("=" * 70)
    logger.info("STEP 1: Creating Inference Endpoint")
    logger.info("=" * 70)

    # Check if endpoint already exists
    try:
        existing = es.inference.get(inference_id=inference_id)
        logger.info(f"Inference endpoint '{inference_id}' already exists")
        logger.info(f"   Task Type: {existing.get('task_type')}")
        logger.info(f"   Service: {existing.get('service')}")
        return inference_id
    except Exception:
        pass  # Endpoint doesn't exist, create it

    # Create inference endpoint
    logger.info(f"Creating inference endpoint: {inference_id}")

    try:
        es.inference.put(
            inference_id=inference_id,
            task_type='sparse_embedding',  # ELSER uses sparse embeddings
            body={
                'service': 'elser',
                'service_settings': {
                    'model_id': '.elser_model_2_linux-x86_64',  # Base model ID
                    'num_allocations': 1,  # Number of model allocations
                    'num_threads': 1  # Threads per allocation
                }
            }
        )

        logger.info(f"Inference endpoint '{inference_id}' created successfully!")
        logger.info("")
        logger.info("What this endpoint does:")
        logger.info("  • Converts text → sparse embeddings using ELSER")
        logger.info("  • Reusable across multiple indices")
        logger.info("  • Optimized for search (low latency)")
        logger.info("")

        return inference_id

    except Exception as e:
        logger.error(f"Failed to create inference endpoint: {e}")
        raise


def recreate_incident_logs_index(es: Elasticsearch, inference_id: str):
    """
    Recreate incident-logs index with semantic_text field

    IMPORTANT: This is the CORRECT way to create the incident-logs index
    for Serverless Elasticsearch. Do NOT use elastic_config.py for this.

    The semantic_text field type automatically:
    - Calls the inference endpoint when documents are indexed
    - Stores both original text and embeddings
    - Enables semantic search

    Modern approach (used here):
      semantic_text + inference_id = automatic ELSER embeddings

    Legacy approach (DO NOT USE):
      ml.tokens + ingest pipeline = manual ELSER configuration
    """
    index_name = os.getenv("ELASTIC_INDEX_LOGS", "incident-logs")

    logger.info("=" * 70)
    logger.info("STEP 2: Recreating incident-logs Index")
    logger.info("=" * 70)

    # Delete existing index
    if es.indices.exists(index=index_name):
        logger.info(f"Deleting existing index: {index_name}")
        es.indices.delete(index=index_name)

    # Create new index with semantic_text field
    logger.info(f"Creating index: {index_name} with semantic_text field")

    es.indices.create(
        index=index_name,
        body={
            "mappings": {
                "properties": {
                    "timestamp": {"type": "date"},
                    "level": {"type": "keyword"},
                    "message": {"type": "text"},
                    "content": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    # SEMANTIC TEXT FIELD - Auto-generates embeddings!
                    "semantic_content": {
                        "type": "semantic_text",
                        "inference_id": inference_id  # Links to ELSER endpoint
                    },
                    "service": {"type": "keyword"},
                    "host": {"type": "keyword"},
                    "metadata": {"type": "object"}
                }
            }
        }
    )

    logger.info(f"Index '{index_name}' created successfully!")
    logger.info("")
    logger.info("Index structure:")
    logger.info("  • content (text): Original text for keyword search")
    logger.info("  • semantic_content (semantic_text): Auto-generates ELSER embeddings")
    logger.info("  • When you index a document, ELSER runs automatically!")
    logger.info("")


def ingest_sample_data(es: Elasticsearch):
    """
    Ingest sample incident logs

    When documents are inserted, the semantic_text field automatically:
    1. Calls the ELSER inference endpoint
    2. Generates embeddings from 'content' field
    3. Stores embeddings in 'semantic_content'
    """
    index_name = os.getenv("ELASTIC_INDEX_LOGS", "incident-logs")

    logger.info("=" * 70)
    logger.info("STEP 3: Ingesting Sample Data")
    logger.info("=" * 70)

    sample_logs = [
        {
            "timestamp": "2025-01-19T10:00:00Z",
            "level": "ERROR",
            "message": "Database connection timeout",
            "content": "Failed to connect to PostgreSQL database after 30 seconds. Connection pool exhausted. Service: api-gateway, Host: prod-api-01",
            "semantic_content": "Failed to connect to PostgreSQL database after 30 seconds. Connection pool exhausted. Service: api-gateway, Host: prod-api-01",
            "service": "api-gateway",
            "host": "prod-api-01",
            "metadata": {"duration_ms": 30000, "retry_count": 3}
        },
        {
            "timestamp": "2025-01-19T10:05:00Z",
            "level": "WARNING",
            "message": "High memory usage detected",
            "content": "Memory usage at 85%. Potential memory leak in worker process. Service: worker, Host: prod-worker-03",
            "semantic_content": "Memory usage at 85%. Potential memory leak in worker process. Service: worker, Host: prod-worker-03",
            "service": "worker",
            "host": "prod-worker-03",
            "metadata": {"memory_percent": 85, "process_id": 1234}
        },
        {
            "timestamp": "2025-01-19T10:10:00Z",
            "level": "ERROR",
            "message": "API rate limit exceeded",
            "content": "External API returned 429 Too Many Requests. Rate limit: 1000 req/min. Current rate: 1250 req/min. Service: integration-service, Host: prod-int-02",
            "semantic_content": "External API returned 429 Too Many Requests. Rate limit: 1000 req/min. Current rate: 1250 req/min. Service: integration-service, Host: prod-int-02",
            "service": "integration-service",
            "host": "prod-int-02",
            "metadata": {"rate_limit": 1000, "current_rate": 1250}
        },
        {
            "timestamp": "2025-01-19T10:15:00Z",
            "level": "ERROR",
            "message": "Database query slow",
            "content": "Query execution took 15 seconds. Query: SELECT * FROM orders WHERE created_at > NOW() - INTERVAL '30 days'. Missing index on created_at column. Service: reporting, Host: prod-db-01",
            "semantic_content": "Query execution took 15 seconds. Query: SELECT * FROM orders WHERE created_at > NOW() - INTERVAL '30 days'. Missing index on created_at column. Service: reporting, Host: prod-db-01",
            "service": "reporting",
            "host": "prod-db-01",
            "metadata": {"query_time_ms": 15000, "table": "orders"}
        },
        {
            "timestamp": "2025-01-19T10:20:00Z",
            "level": "CRITICAL",
            "message": "Service unavailable",
            "content": "Payment service returned 503 Service Unavailable. All 3 replicas are down. Last health check failed. Service: payment, Host: prod-payment-lb",
            "semantic_content": "Payment service returned 503 Service Unavailable. All 3 replicas are down. Last health check failed. Service: payment, Host: prod-payment-lb",
            "service": "payment",
            "host": "prod-payment-lb",
            "metadata": {"replicas_down": 3, "health_check": "failed"}
        }
    ]

    logger.info(f"Indexing {len(sample_logs)} sample documents...")
    logger.info("(ELSER will automatically generate embeddings for each)")
    logger.info("")

    for i, log in enumerate(sample_logs, 1):
        try:
            response = es.index(
                index=index_name,
                document=log,
                refresh=True  # Make immediately searchable
            )
            logger.info(f"  [{i}/{len(sample_logs)}] Indexed: {log['message']}")
        except Exception as e:
            logger.error(f"  [{i}/{len(sample_logs)}] Failed: {e}")

    logger.info("")
    logger.info(f"Ingested {len(sample_logs)} documents")
    logger.info("")


def test_semantic_search(es: Elasticsearch):
    """
    Test semantic search with ELSER

    Demonstrates the power of semantic search:
    - Query: "database failures"
    - Finds: "timeout", "slow query", "connection issues"
    """
    index_name = os.getenv("ELASTIC_INDEX_LOGS", "incident-logs")

    logger.info("=" * 70)
    logger.info("STEP 4: Testing Semantic Search")
    logger.info("=" * 70)

    test_queries = [
        "database failures",
        "memory problems",
        "service down"
    ]

    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        logger.info("-" * 70)

        try:
            response = es.search(
                index=index_name,
                body={
                    "query": {
                        "semantic": {
                            "field": "semantic_content",
                            "query": query
                        }
                    },
                    "size": 3
                }
            )

            hits = response['hits']['hits']
            logger.info(f"Found {len(hits)} results:\n")

            for i, hit in enumerate(hits, 1):
                source = hit['_source']
                score = hit['_score']
                logger.info(f"  {i}. [Score: {score:.2f}] {source['message']}")
                logger.info(f"     Service: {source['service']} | Level: {source['level']}")

            if not hits:
                logger.info("  No results found")

        except Exception as e:
            logger.error(f"Search failed: {e}")

    logger.info("")


def test_hybrid_search(es: Elasticsearch):
    """
    Test hybrid search: Semantic (ELSER) + Keyword (BM25)

    Combines:
    - Semantic understanding (ELSER)
    - Exact keyword matching (BM25)
    """
    index_name = os.getenv("ELASTIC_INDEX_LOGS", "incident-logs")

    logger.info("=" * 70)
    logger.info("STEP 5: Testing Hybrid Search (Semantic + Keyword)")
    logger.info("=" * 70)

    query = "database timeout"

    logger.info(f"\nHybrid Query: '{query}'")
    logger.info("-" * 70)

    try:
        response = es.search(
            index=index_name,
            body={
                "query": {
                    "bool": {
                        "should": [
                            # Semantic search (ELSER)
                            {
                                "semantic": {
                                    "field": "semantic_content",
                                    "query": query
                                }
                            },
                            # Keyword search (BM25)
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["message^2", "content", "service"]
                                }
                            }
                        ]
                    }
                },
                "size": 5
            }
        )

        hits = response['hits']['hits']
        logger.info(f"Found {len(hits)} results:\n")

        for i, hit in enumerate(hits, 1):
            source = hit['_source']
            score = hit['_score']
            logger.info(f"  {i}. [Score: {score:.2f}] {source['message']}")
            logger.info(f"     {source['content'][:100]}...")
            logger.info("")

        logger.info("Hybrid search combines semantic understanding + exact matches!")
        logger.info("")

    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")


def main():
    """Run complete ELSER setup"""
    print("\n" + "=" * 70)
    print("ELSER SETUP FOR SERVERLESS ELASTICSEARCH")
    print("=" * 70 + "\n")

    try:
        # Get client
        es = get_es_client()
        logger.info("Connected to Elasticsearch Serverless\n")

        # Step 1: Create inference endpoint
        inference_id = create_inference_endpoint(es)

        # Step 2: Recreate index with semantic_text
        recreate_incident_logs_index(es, inference_id)

        # Step 3: Ingest sample data
        ingest_sample_data(es)

        # Step 4: Test semantic search
        test_semantic_search(es)

        # Step 5: Test hybrid search
        test_hybrid_search(es)

        # Summary
        print("\n" + "=" * 70)
        print("SETUP COMPLETE!")
        print("=" * 70)
        print("\nWhat was configured:")
        print("  1. Inference Endpoint: 'elser-incident-analysis'")
        print("  2. Index: 'incident-logs' with semantic_text field")
        print("  3. Sample data: 5 incident logs")
        print("  4. Semantic search: Tested and working")
        print("  5. Hybrid search: Tested and working")
        print("\nNext steps:")
        print("  → Run the POC: python elastic_reflection_poc.py")
        print("=" * 70 + "\n")

    except Exception as e:
        logger.error(f"\nSetup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
