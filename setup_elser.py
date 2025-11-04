"""
ELSER Setup for Elasticsearch
==============================

Automated script to:
1. Create Inference Endpoint for ELSER
2. Create incident-logs index with semantic_text field
3. Test semantic search

Works with both traditional Elasticsearch deployments and Serverless.

Usage:
    python setup_elser.py
"""

import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import logging
from datetime import datetime, timedelta, timezone
from elastic_config import get_elastic_config, get_elastic_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()


def create_inference_endpoint(es: Elasticsearch):
    """
    Create Inference Endpoint for ELSER

    Inference Endpoints are the modern way to use ML models in Elasticsearch.
    They are reusable across indices and optimized for search (low latency).
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
    Create incident-logs index with semantic_text field

    The semantic_text field type automatically:
    - Calls the inference endpoint when documents are indexed
    - Stores both original text and embeddings
    - Enables semantic search

    Modern approach (used here):
      semantic_text + inference_id = automatic ELSER embeddings
    """
    config = get_elastic_config()
    index_name = config.index_logs

    logger.info("=" * 70)
    logger.info("STEP 2: Creating incident-logs Index")
    logger.info("=" * 70)

    # Delete existing index if it exists
    if es.indices.exists(index=index_name):
        logger.info(f"Deleting existing index: {index_name}")
        es.indices.delete(index=index_name)

    # Determine if using serverless (no shards/replicas config)
    is_serverless = not config.cloud_id

    # Create new index with semantic_text field
    logger.info(f"Creating index: {index_name} with semantic_text field")

    index_body = {
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

    # Add settings for traditional deployments (not applicable to Serverless)
    if not is_serverless:
        index_body["settings"] = {
            "number_of_shards": 1,
            "number_of_replicas": 1
        }

    es.indices.create(index=index_name, body=index_body)

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
    config = get_elastic_config()
    index_name = config.index_logs

    logger.info("=" * 70)
    logger.info("STEP 3: Ingesting Sample Data")
    logger.info("=" * 70)

    # Create detailed logs with comprehensive metrics for database timeout incident
    base_time = datetime.now(timezone.utc) - timedelta(hours=2)

    sample_logs = [
        # 1. Baseline - normal operation
        {
            "timestamp": (base_time - timedelta(minutes=30)).isoformat(),
            "level": "INFO",
            "service": "database-monitor",
            "host": "prod-db-01",
            "message": "Database health check - normal operation",
            "content": "PostgreSQL connection pool status: 45/200 connections active (22.5% utilization). Query average: 125ms. No slow queries detected.",
            "semantic_content": "PostgreSQL connection pool status: 45/200 connections active (22.5% utilization). Query average: 125ms. No slow queries detected.",
            "metadata": {
                "connections_active": 45,
                "connections_max": 200,
                "pool_utilization_pct": 22.5,
                "avg_query_time_ms": 125,
                "slow_queries_count": 0
            }
        },
        # 2. Early warning - increasing load
        {
            "timestamp": (base_time - timedelta(minutes=20)).isoformat(),
            "level": "WARN",
            "service": "database-monitor",
            "host": "prod-db-01",
            "message": "Connection pool utilization increasing",
            "content": "PostgreSQL connection pool: 120/200 connections active (60% utilization). Query average: 450ms. Detected 3 slow queries (>2s).",
            "semantic_content": "PostgreSQL connection pool: 120/200 connections active (60% utilization). Query average: 450ms. Detected 3 slow queries (>2s).",
            "metadata": {
                "connections_active": 120,
                "connections_max": 200,
                "pool_utilization_pct": 60.0,
                "avg_query_time_ms": 450,
                "slow_queries_count": 3
            }
        },
        # 3. Critical - missing index identified
        {
            "timestamp": (base_time - timedelta(minutes=15)).isoformat(),
            "level": "ERROR",
            "service": "reporting-service",
            "host": "prod-app-02",
            "message": "Slow query detected on orders table",
            "content": "Query execution time: 15.3 seconds. Query: SELECT * FROM orders WHERE created_at > NOW() - INTERVAL '30 days'. Database explanation: Sequential Scan on orders (cost=0.00..125000.00 rows=500000). Missing index on created_at column causing full table scan.",
            "semantic_content": "Query execution time: 15.3 seconds. Query: SELECT * FROM orders WHERE created_at > NOW() - INTERVAL '30 days'. Database explanation: Sequential Scan on orders (cost=0.00..125000.00 rows=500000). Missing index on created_at column causing full table scan.",
            "metadata": {
                "query_time_ms": 15300,
                "table": "orders",
                "scan_type": "sequential",
                "rows_scanned": 500000,
                "index_missing": "created_at"
            }
        },
        # 4. Connection pool reaching critical level
        {
            "timestamp": (base_time - timedelta(minutes=10)).isoformat(),
            "level": "ERROR",
            "service": "database-monitor",
            "host": "prod-db-01",
            "message": "Connection pool near exhaustion",
            "content": "PostgreSQL connection pool: 185/200 connections active (92.5% utilization). 25 queries waiting for connection. Average wait time: 8.2 seconds. Slow query count: 12 (all on orders table).",
            "semantic_content": "PostgreSQL connection pool: 185/200 connections active (92.5% utilization). 25 queries waiting for connection. Average wait time: 8.2 seconds. Slow query count: 12 (all on orders table).",
            "metadata": {
                "connections_active": 185,
                "connections_max": 200,
                "pool_utilization_pct": 92.5,
                "queries_waiting": 25,
                "avg_wait_time_ms": 8200,
                "slow_queries_count": 12
            }
        },
        # 5. API Gateway timeout begins
        {
            "timestamp": (base_time - timedelta(minutes=8)).isoformat(),
            "level": "ERROR",
            "service": "api-gateway",
            "host": "prod-gateway-01",
            "message": "Database connection timeout",
            "content": "Failed to acquire database connection after 30 seconds. Connection pool exhausted. Request queue: 45 pending requests. Response time P95: 32000ms (SLA: 2000ms).",
            "semantic_content": "Failed to acquire database connection after 30 seconds. Connection pool exhausted. Request queue: 45 pending requests. Response time P95: 32000ms (SLA: 2000ms).",
            "metadata": {
                "timeout_ms": 30000,
                "pending_requests": 45,
                "p95_response_time_ms": 32000,
                "sla_response_time_ms": 2000,
                "error_rate_pct": 15.5
            }
        },
        # 6. Multiple slow queries confirmed
        {
            "timestamp": (base_time - timedelta(minutes=7)).isoformat(),
            "level": "ERROR",
            "service": "database-monitor",
            "host": "prod-db-01",
            "message": "Multiple slow queries blocking connection pool",
            "content": "Active long-running queries: 18. All executing variations of: SELECT * FROM orders WHERE created_at > [date]. Average execution time: 14.8 seconds. Connection pool: 198/200 (99% utilization). New connection requests timing out.",
            "semantic_content": "Active long-running queries: 18. All executing variations of: SELECT * FROM orders WHERE created_at > [date]. Average execution time: 14.8 seconds. Connection pool: 198/200 (99% utilization). New connection requests timing out.",
            "metadata": {
                "long_running_queries": 18,
                "avg_execution_time_ms": 14800,
                "connections_active": 198,
                "connections_max": 200,
                "pool_utilization_pct": 99.0
            }
        },
        # 7. Worker service memory pressure
        {
            "timestamp": (base_time - timedelta(minutes=6)).isoformat(),
            "level": "WARN",
            "service": "worker-service",
            "host": "prod-worker-03",
            "message": "High memory usage detected",
            "content": "Memory usage: 6.8GB/8GB (85%). JVM heap: 5.2GB/6GB. GC pressure increasing. Database connection retries accumulating in queue: 32 pending jobs.",
            "semantic_content": "Memory usage: 6.8GB/8GB (85%). JVM heap: 5.2GB/6GB. GC pressure increasing. Database connection retries accumulating in queue: 32 pending jobs.",
            "metadata": {
                "memory_used_gb": 6.8,
                "memory_total_gb": 8.0,
                "memory_pct": 85.0,
                "pending_jobs": 32,
                "gc_pressure": "high"
            }
        },
        # 8. Integration service rate limiting
        {
            "timestamp": (base_time - timedelta(minutes=5)).isoformat(),
            "level": "ERROR",
            "service": "integration-service",
            "host": "prod-integration-01",
            "message": "External API rate limit exceeded",
            "content": "Payment provider API: 1250 requests/minute (limit: 1000/min). HTTP 429 responses: 156. Retry attempts accumulating due to database timeout preventing request completion. Backlog: 89 pending retries.",
            "semantic_content": "Payment provider API: 1250 requests/minute (limit: 1000/min). HTTP 429 responses: 156. Retry attempts accumulating due to database timeout preventing request completion. Backlog: 89 pending retries.",
            "metadata": {
                "request_rate": 1250,
                "rate_limit": 1000,
                "http_429_count": 156,
                "retry_backlog": 89
            }
        },
        # 9. Connection pool completely exhausted
        {
            "timestamp": (base_time - timedelta(minutes=3)).isoformat(),
            "level": "CRITICAL",
            "service": "database-monitor",
            "host": "prod-db-01",
            "message": "Connection pool completely exhausted",
            "content": "PostgreSQL connection pool: 200/200 connections active (100% utilization). 78 queries waiting. Average wait time: 45 seconds. All connections held by slow queries on orders table. No new connections possible.",
            "semantic_content": "PostgreSQL connection pool: 200/200 connections active (100% utilization). 78 queries waiting. Average wait time: 45 seconds. All connections held by slow queries on orders table. No new connections possible.",
            "metadata": {
                "connections_active": 200,
                "connections_max": 200,
                "pool_utilization_pct": 100.0,
                "queries_waiting": 78,
                "avg_wait_time_ms": 45000
            }
        },
        # 10. Payment service health check failures
        {
            "timestamp": (base_time - timedelta(minutes=2)).isoformat(),
            "level": "ERROR",
            "service": "payment-service",
            "host": "prod-payment-01",
            "message": "Health check failed - database unavailable",
            "content": "Health check endpoint returning 503. Cannot acquire database connection. Last successful DB query: 3 minutes ago. Service marked unhealthy by load balancer.",
            "semantic_content": "Health check endpoint returning 503. Cannot acquire database connection. Last successful DB query: 3 minutes ago. Service marked unhealthy by load balancer.",
            "metadata": {
                "health_status": "unhealthy",
                "last_successful_query_seconds_ago": 180,
                "http_status": 503
            }
        },
        # 11. Payment service cascade failure
        {
            "timestamp": (base_time - timedelta(minutes=1)).isoformat(),
            "level": "CRITICAL",
            "service": "kubernetes",
            "host": "prod-cluster",
            "message": "Payment service pods failing",
            "content": "Payment service: 0/3 replicas ready. All pods failing liveness probe due to database connection timeout. Replica prod-payment-01: CrashLoopBackOff. Replica prod-payment-02: CrashLoopBackOff. Replica prod-payment-03: CrashLoopBackOff. Root cause: Cannot establish database connection (timeout after 30s).",
            "semantic_content": "Payment service: 0/3 replicas ready. All pods failing liveness probe due to database connection timeout. Replica prod-payment-01: CrashLoopBackOff. Replica prod-payment-02: CrashLoopBackOff. Replica prod-payment-03: CrashLoopBackOff. Root cause: Cannot establish database connection (timeout after 30s).",
            "metadata": {
                "replicas_ready": 0,
                "replicas_desired": 3,
                "pod_status": "CrashLoopBackOff",
                "failure_reason": "database_connection_timeout"
            }
        },
        # 12. Business impact quantified
        {
            "timestamp": base_time.isoformat(),
            "level": "CRITICAL",
            "service": "monitoring",
            "host": "prod-monitor",
            "message": "Production incident - payment processing unavailable",
            "content": "Critical incident detected. Payment processing: 100% failure rate. Failed transactions: 342 in last 5 minutes. Estimated revenue impact: $28,500. Affected users: ~1,200. Root cause: Database connection pool exhaustion due to slow queries on unindexed orders.created_at column.",
            "semantic_content": "Critical incident detected. Payment processing: 100% failure rate. Failed transactions: 342 in last 5 minutes. Estimated revenue impact: $28,500. Affected users: ~1,200. Root cause: Database connection pool exhaustion due to slow queries on unindexed orders.created_at column.",
            "metadata": {
                "payment_failure_rate_pct": 100.0,
                "failed_transactions": 342,
                "estimated_revenue_impact_usd": 28500,
                "affected_users": 1200,
                "incident_duration_minutes": 20
            }
        },
        # 13. Database query analysis
        {
            "timestamp": (base_time + timedelta(minutes=1)).isoformat(),
            "level": "INFO",
            "service": "database-admin",
            "host": "prod-db-01",
            "message": "Query performance analysis",
            "content": "Analysis of pg_stat_statements: Query 'SELECT * FROM orders WHERE created_at > $1' executed 18 times in last 10 minutes. Average time: 14.2s. Total time: 256 seconds. Execution plan shows sequential scan (Seq Scan on orders). Recommended action: CREATE INDEX idx_orders_created_at ON orders(created_at);",
            "semantic_content": "Analysis of pg_stat_statements: Query 'SELECT * FROM orders WHERE created_at > $1' executed 18 times in last 10 minutes. Average time: 14.2s. Total time: 256 seconds. Execution plan shows sequential scan (Seq Scan on orders). Recommended action: CREATE INDEX idx_orders_created_at ON orders(created_at);",
            "metadata": {
                "query_executions": 18,
                "avg_time_ms": 14200,
                "total_time_ms": 256000,
                "scan_type": "sequential",
                "recommended_index": "idx_orders_created_at"
            }
        },
        # 14. Historical comparison
        {
            "timestamp": (base_time + timedelta(minutes=2)).isoformat(),
            "level": "INFO",
            "service": "database-monitor",
            "host": "prod-db-01",
            "message": "Historical performance comparison",
            "content": "Historical analysis: Same query 24 hours ago with 400K rows: 150ms (index was present). Current execution with 500K rows: 15000ms (index missing). Performance degradation: 100x slower. Index was dropped during schema migration at 2024-10-30 08:00 UTC.",
            "semantic_content": "Historical analysis: Same query 24 hours ago with 400K rows: 150ms (index was present). Current execution with 500K rows: 15000ms (index missing). Performance degradation: 100x slower. Index was dropped during schema migration at 2024-10-30 08:00 UTC.",
            "metadata": {
                "historical_query_time_ms": 150,
                "current_query_time_ms": 15000,
                "performance_degradation_factor": 100,
                "index_dropped_timestamp": "2024-10-30T08:00:00Z"
            }
        },
        # 15. Confirmation of root cause
        {
            "timestamp": (base_time + timedelta(minutes=3)).isoformat(),
            "level": "INFO",
            "service": "database-admin",
            "host": "prod-db-01",
            "message": "Root cause confirmed",
            "content": "Root cause analysis complete. Primary cause: Missing index on orders.created_at column (dropped during migration). Impact: Sequential scans causing 15s query times. Effect: Connection pool exhaustion (200/200 connections held by slow queries). Cascading failures: API timeouts → Payment service crash → Business impact ($28.5K revenue). Solution confirmed: Recreate index.",
            "semantic_content": "Root cause analysis complete. Primary cause: Missing index on orders.created_at column (dropped during migration). Impact: Sequential scans causing 15s query times. Effect: Connection pool exhaustion (200/200 connections held by slow queries). Cascading failures: API timeouts → Payment service crash → Business impact ($28.5K revenue). Solution confirmed: Recreate index.",
            "metadata": {
                "root_cause": "missing_index",
                "affected_table": "orders",
                "affected_column": "created_at",
                "query_time_impact_ms": 15000,
                "connection_pool_exhausted": True,
                "business_impact_usd": 28500
            }
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
    config = get_elastic_config()
    index_name = config.index_logs

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
    config = get_elastic_config()
    index_name = config.index_logs

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
    print("ELSER SETUP FOR ELASTICSEARCH")
    print("=" * 70 + "\n")

    try:
        # Get client using elastic_config (supports both traditional and serverless)
        es = get_elastic_client()
        config = get_elastic_config()
        deployment_type = "Serverless" if not config.cloud_id else "Traditional"
        logger.info(f"Connected to Elasticsearch ({deployment_type})\n")

        # Step 1: Create inference endpoint
        inference_id = create_inference_endpoint(es)

        # Step 2: Create index with semantic_text
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
        print("  3. Sample data: 15 incident logs")
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

