"""
Elastic Cloud Configuration Module
===================================

Handles connection to Elasticsearch Cloud for Multi-Agent Orchestration.

Features:
- Cloud ID authentication
- API Key security
- Connection pooling
- Health checks
- Index management
"""

import os
from typing import Optional
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, AuthenticationException
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ElasticCloudConfig:
    """
    Configuration and connection management for Elastic Cloud
    Supports both traditional deployments (Cloud ID) and Serverless (endpoint)
    """

    def __init__(
        self,
        cloud_id: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        request_timeout: int = 30,
        max_retries: int = 3,
        retry_on_timeout: bool = True
    ):
        """
        Initialize Elastic Cloud connection

        Args:
            cloud_id: Elastic Cloud deployment ID (traditional)
            api_key: Base64 encoded API key
            endpoint: Elasticsearch endpoint URL (serverless)
            request_timeout: Request timeout in seconds
            max_retries: Number of retries on failure
            retry_on_timeout: Whether to retry on timeout
        """
        self.cloud_id = cloud_id or os.getenv("ELASTIC_CLOUD_ID")
        self.api_key = api_key or os.getenv("ELASTIC_API_KEY")
        self.endpoint = endpoint or os.getenv("ELASTIC_ENDPOINT")

        # Index names from environment
        self.index_logs = os.getenv("ELASTIC_INDEX_LOGS", "incident-logs")
        self.index_memory = os.getenv("ELASTIC_INDEX_MEMORY", "agent-memory")

        # Validate credentials
        if not self.api_key:
            raise ValueError("ELASTIC_API_KEY not found in environment variables")

        if not self.cloud_id and not self.endpoint:
            raise ValueError("Either ELASTIC_CLOUD_ID or ELASTIC_ENDPOINT must be provided")

        # Initialize Elasticsearch client
        if self.endpoint:
            # Direct endpoint connection (Serverless or custom endpoint)
            logger.info(f"Connecting to Elasticsearch via endpoint: {self.endpoint}")
            self.es = Elasticsearch(
                hosts=[self.endpoint],
                api_key=self.api_key,
                request_timeout=request_timeout,
                max_retries=max_retries,
                retry_on_timeout=retry_on_timeout
            )
        else:
            # Traditional Cloud deployment (Cloud ID)
            logger.info(f"Connecting to Elasticsearch Cloud (Cloud ID)")
            self.es = Elasticsearch(
                cloud_id=self.cloud_id,
                api_key=self.api_key,
                request_timeout=request_timeout,
                max_retries=max_retries,
                retry_on_timeout=retry_on_timeout
            )

        # Verify connection
        self._verify_connection()

    def _verify_connection(self):
        """Verify connection to Elastic Cloud"""
        try:
            if not self.es.ping():
                raise ConnectionError("Failed to ping Elasticsearch")

            # Get cluster info
            info = self.es.info()
            logger.info(f"Connected to Elasticsearch Cloud")
            logger.info(f"   Cluster: {info['cluster_name']}")
            logger.info(f"   Version: {info['version']['number']}")

            # Check cluster health (not available for all deployment types)
            if self.cloud_id:  # Traditional deployments support cluster health
                try:
                    health = self.es.cluster.health()
                    logger.info(f"   Status: {health['status']}")
                except Exception:
                    logger.info(f"   Status: N/A (cluster.health not available)")
            else:
                logger.info(f"   Status: N/A (cluster.health not available for this deployment type)")

        except AuthenticationException:
            logger.error("Authentication failed - check your API key")
            raise
        except ConnectionError as e:
            logger.error(f"Connection failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    def get_client(self) -> Elasticsearch:
        """
        Get Elasticsearch client

        Returns:
            Elasticsearch client instance
        """
        return self.es

    def health_check(self) -> dict:
        """
        Perform health check (works with both traditional and serverless deployments)

        Returns:
            dict with health status
        """
        try:
            indices_health = {
                "logs": self.es.indices.exists(index=self.index_logs),
                "memory": self.es.indices.exists(index=self.index_memory)
            }

            result = {
                "indices": indices_health,
                "elser_ready": self._check_elser()
            }

            # Cluster health only for traditional deployments
            if self.cloud_id:
                try:
                    health = self.es.cluster.health()
                    result.update({
                        "cluster_status": health['status'],
                        "cluster_name": health['cluster_name'],
                        "number_of_nodes": health['number_of_nodes'],
                        "active_shards": health['active_shards']
                    })
                except Exception:
                    result.update({
                        "cluster_status": "N/A (cluster.health not available)",
                        "cluster_name": "N/A",
                        "number_of_nodes": "N/A",
                        "active_shards": "N/A"
                    })
            else:
                # Endpoint-based connection (may be serverless or custom)
                info = self.es.info()
                result.update({
                    "cluster_status": "N/A (cluster.health not available)",
                    "cluster_name": info.get('cluster_name', 'N/A'),
                    "number_of_nodes": "N/A",
                    "active_shards": "N/A"
                })

            return result

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"error": str(e)}

    def _check_elser(self) -> bool:
        """Check if ELSER inference endpoint is available"""
        try:
            # Check for inference endpoint (modern approach)
            inference_id = "elser-incident-analysis"
            try:
                endpoint = self.es.inference.get(inference_id=inference_id)
                return endpoint is not None
            except Exception:
                # Fallback: Check for traditional ML model deployment (legacy)
                try:
                    stats = self.es.ml.get_trained_models_stats(model_id=".elser_model_2")
                    if stats['trained_model_stats']:
                        deployment_stats = stats['trained_model_stats'][0].get('deployment_stats', {})
                        state = deployment_stats.get('state', 'not_deployed')
                        return state == 'started'
                except Exception:
                    pass
            return False
        except Exception:
            return False

    def create_indices_if_not_exist(self):
        """
        Create required indices if they don't exist

        IMPORTANT: The incident-logs index with ELSER semantic_text field
        should be created by running setup_elser.py first.
        This method only creates the memory indices.
        """
        logger.info("Checking indices...")

        # Determine if using serverless (no shards/replicas config)
        is_serverless = not self.cloud_id

        # Incident logs index with ELSER
        # NOTE: This index should be created by setup_elser.py
        # which properly configures semantic_text field with inference endpoint
        if not self.es.indices.exists(index=self.index_logs):
            logger.warning(f"Index '{self.index_logs}' does not exist!")
            logger.warning("Please run 'python setup_elser.py' first to create the index with ELSER configuration.")
            logger.warning("Skipping automatic creation to avoid incorrect schema.")


        # Long-term memory index with semantic_text field for semantic search
        if not self.es.indices.exists(index=self.index_memory):
            logger.info(f"Creating index: {self.index_memory}")

            # Get inference endpoint ID (same as used for incident-logs)
            inference_id = "elser-incident-analysis"

            index_body = {
                "mappings": {
                    "properties": {
                        "memory_id": {"type": "keyword"},
                        "agent_name": {"type": "keyword"},
                        "memory_type": {"type": "keyword"},
                        "content": {
                            "type": "text",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        # Semantic search field - enables concept-based retrieval
                        "semantic_content": {
                            "type": "semantic_text",
                            "inference_id": inference_id  # Links to ELSER endpoint
                        },
                        "timestamp": {"type": "date"},
                        "success": {"type": "boolean"},
                        "metadata": {"type": "object"}
                    }
                }
            }

            if not is_serverless:
                index_body["settings"] = {
                    "number_of_shards": 1,
                    "number_of_replicas": 1
                }

            self.es.indices.create(index=self.index_memory, body=index_body)

        logger.info("All indices ready")

    def close(self):
        """Close Elasticsearch connection"""
        if self.es:
            self.es.close()
            logger.info("Connection closed")


# Singleton instance
_elastic_instance: Optional[ElasticCloudConfig] = None


def get_elastic_client() -> Elasticsearch:
    """
    Get or create Elasticsearch client (singleton pattern)

    Returns:
        Elasticsearch client instance

    Example:
        >>> es = get_elastic_client()
        >>> es.search(index="incident-logs", body={"query": {"match_all": {}}})
    """
    global _elastic_instance

    if _elastic_instance is None:
        _elastic_instance = ElasticCloudConfig()
        _elastic_instance.create_indices_if_not_exist()

    return _elastic_instance.get_client()


def get_elastic_config() -> ElasticCloudConfig:
    """
    Get ElasticCloudConfig instance

    Returns:
        ElasticCloudConfig instance
    """
    global _elastic_instance

    if _elastic_instance is None:
        _elastic_instance = ElasticCloudConfig()
        _elastic_instance.create_indices_if_not_exist()

    return _elastic_instance


# Test connection when module is imported
if __name__ == "__main__":
    print("=" * 70)
    print("ELASTIC CLOUD CONNECTION TEST")
    print("=" * 70)
    print()

    try:
        # Initialize connection
        config = ElasticCloudConfig()

        # Create indices
        config.create_indices_if_not_exist()

        # Health check
        health = config.health_check()
        print("\nHealth Check:")
        print(f"   Cluster Status: {health.get('cluster_status')}")
        print(f"   Nodes: {health.get('number_of_nodes')}")
        print(f"   Active Shards: {health.get('active_shards')}")
        print(f"   ELSER Ready: {health.get('elser_ready')}")
        print()

        print("Indices Status:")
        for idx_name, exists in health.get('indices', {}).items():
            status = "OK" if exists else "MISSING"
            print(f"   [{status}] {idx_name}")

        print()
        print("=" * 70)
        print("CONNECTION TEST PASSED")
        print("=" * 70)

        # Close connection
        config.close()

    except Exception as e:
        print(f"ERROR: Connection test failed: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check your .env file has ELASTIC_CLOUD_ID and ELASTIC_API_KEY")
        print("  2. Verify API key is valid")
        print("  3. Ensure deployment is running in Elastic Cloud")
