"""
Multi-Agent Orchestration with Reflection Pattern
==================================================

POC: Incident Analysis with Self-Correction Loop

PREREQUISITES:
  1. Run setup_elser_serverless.py FIRST to create incident-logs index
  2. Ensure .env file has correct Elasticsearch credentials
  3. Verify ELSER model is deployed and running

Architecture (3 Specialized Agents):
┌─────────────────────────────────────────────────────┐
│                  LangGraph Engine                    │
│  (Orchestration - NOT "Orchestrator Pattern")       │
└─────────────────────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┬─────────────┐
        ▼                           ▼             ▼
  ┌──────────┐              ┌──────────┐   ┌──────────────┐
  │  Search  │─────────────►│ Analyser │──►│  Reflection  │
  │  Agent   │              │  Agent   │   │    Agent     │
  └──────────┘              └──────────┘   └──────────────┘
   (runs once)                    ▲                │
                                  │                │
                                  └────[decisor]───┘
                                   (quality < 0.8?)

Agent Responsibilities:
1. SearchAgent: Hybrid search (ELSER semantic + BM25 keyword)
2. AnalyserAgent: LLM-based root cause analysis
3. ReflectionAgent: Quality evaluation + feedback

Elastic Cloud Integration:
- Hybrid Search: ELSER (semantic_text field) + BM25 (keyword)
- Long-Term Memory: agent-memory index
- RAG: incident-logs index with semantic_text field

Author: Tutorial LangGraph + Elastic
"""

import os
from typing import TypedDict, Annotated, List, Literal
from datetime import datetime
import logging

# LangGraph & LangChain
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama

# Elastic
from elasticsearch import Elasticsearch
from elastic_config import get_elastic_client, get_elastic_config

# Utils
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()


# ============================================================================
# STATE DEFINITION
# ============================================================================

class IncidentState(TypedDict):
    """
    Shared state between agents in the Reflection cycle

    Attributes:
        messages: Message history (SystemMessage, HumanMessage, AIMessage)
        query: Original user query
        search_results: Search results from Elasticsearch
        analysis: Initial analysis from AnalyserAgent
        reflection: Feedback from ReflectionAgent
        final_output: Final output after approval
        quality_score: Quality score (0.0 to 1.0)
        iteration: Current iteration number
        max_iterations: Iteration limit
    """
    messages: Annotated[List, add_messages]
    query: str
    search_results: List[dict]
    analysis: str
    reflection: str
    final_output: str
    quality_score: float
    iteration: int
    max_iterations: int


# ============================================================================
# ELASTIC SEARCH FUNCTIONS
# ============================================================================

def hybrid_search_incidents(es: Elasticsearch, query: str, size: int = 5) -> List[dict]:
    """
    Hybrid Search: ELSER (semantic) + BM25 (keyword)

    Serverless approach using semantic_text field type
    """
    config = get_elastic_config()

    search_body = {
        "size": size,
        "query": {
            "bool": {
                "should": [
                    # ELSER semantic search (using semantic_text field)
                    {
                        "semantic": {
                            "field": "semantic_content",
                            "query": query
                        }
                    },
                    # BM25 keyword search
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["message^2", "content", "service"],
                            "type": "best_fields"
                        }
                    }
                ]
            }
        },
        "sort": [
            {"_score": {"order": "desc"}},
            {"timestamp": {"order": "desc"}}
        ]
    }

    try:
        result = es.search(index=config.index_logs, body=search_body)

        hits = []
        for hit in result["hits"]["hits"]:
            source = hit["_source"]
            hits.append({
                "score": hit["_score"],
                "timestamp": source.get("timestamp"),
                "level": source.get("level"),
                "message": source.get("message"),
                "content": source.get("content"),
                "service": source.get("service"),
                "host": source.get("host")
            })

        logger.info(f"Hybrid search found {len(hits)} results")
        return hits

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def save_to_long_term_memory(
    es: Elasticsearch,
    agent_name: str,
    memory_type: str,
    content: str,
    success: bool,
    metadata: dict = None
):
    """
    Saves Long-Term Memory (LTM) to Elasticsearch

    Types: "decision", "lesson", "pattern"
    Used to improve agents over time
    """
    config = get_elastic_config()

    doc = {
        "memory_id": f"{agent_name}_{datetime.now().timestamp()}",
        "agent_name": agent_name,
        "memory_type": memory_type,
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "metadata": metadata or {}
    }

    es.index(index=config.index_memory, body=doc)
    logger.info(f"LTM saved: {agent_name} - {memory_type}")


# ============================================================================
# AGENT NODES
# ============================================================================

def search_agent(state: IncidentState) -> IncidentState:
    """
    SearchAgent: Performs hybrid search on Elasticsearch

    Specialization: Retrieval from external data sources

    Steps:
    1. Extract query from state
    2. Execute hybrid search (ELSER semantic + BM25 keyword)
    3. Store results in state for AnalyserAgent

    This agent runs ONCE at the start of the workflow.
    Results are reused across all reflection iterations.
    """
    logger.info("SearchAgent: Performing hybrid search")

    # Get Elasticsearch client
    es = get_elastic_client()

    # Search incidents
    query = state["query"]
    search_results = hybrid_search_incidents(es, query, size=5)

    # Update state
    state["search_results"] = search_results
    state["messages"].append(
        AIMessage(content=f"[SearchAgent] Found {len(search_results)} relevant logs")
    )

    logger.info(f"Search completed: {len(search_results)} logs found")

    return state


def analyser_agent(state: IncidentState) -> IncidentState:
    """
    AnalyserAgent: Analyzes logs found by SearchAgent

    Specialization: LLM-based reasoning and root cause analysis

    Steps:
    1. Read search_results from state (populated by SearchAgent)
    2. Format logs for LLM context
    3. Incorporate previous reflection feedback (if exists)
    4. Generate analysis: root cause + impact + actions
    5. Update state with analysis
    """
    current_iter = state.get('iteration', 1)
    logger.info(f"AnalyserAgent (iteration {current_iter})")

    # Read search results from state (already populated by SearchAgent)
    search_results = state.get("search_results", [])
    query = state["query"]

    if not search_results:
        logger.warning("No search results found in state")
        state["analysis"] = "No logs found to analyze."
        return state

    # Format results for LLM
    context = "\n\n".join([
        f"[{r['level']}] {r['timestamp']}\n"
        f"Service: {r['service']} | Host: {r['host']}\n"
        f"{r['message']}\n"
        f"{r['content']}"
        for r in search_results
    ])

    # LLM analysis
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        temperature=0.3
    )

    # Consider previous reflection if exists
    previous_feedback = state.get("reflection", "")
    feedback_context = f"\n\n**Previous feedback:**\n{previous_feedback}" if previous_feedback else ""

    prompt = f"""You are an IT incident analysis expert.

**User query:** {query}

**Logs found:**
{context}
{feedback_context}

**Task:**
Analyze the logs and provide:
1. Root cause
2. Impact
3. Recommended actions

Be specific and base your analysis only on the provided logs."""

    messages = [
        SystemMessage(content="You are an incident analysis expert."),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)
    analysis = response.content

    # Update state (search_results already set by SearchAgent)
    state["analysis"] = analysis
    state["messages"].append(AIMessage(content=f"[AnalyserAgent] {analysis}"))

    logger.info(f"Analysis completed ({len(search_results)} logs analyzed)")

    return state


def reflection_agent(state: IncidentState) -> IncidentState:
    """
    ReflectionAgent: Evaluates analysis quality

    Quality Criteria:
    - Completeness (all aspects covered?)
    - Evidence (based on data?)
    - Actionability (clear actions?)
    - Precision (logical conclusions?)

    Score: 0.0 to 1.0
    """
    logger.info(f"ReflectionAgent (iteration {state['iteration']})")

    analysis = state["analysis"]

    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        temperature=0.2
    )

    prompt = f"""You are a technical analysis critic.

**Analysis to evaluate:**
{analysis}

**Task:**
Evaluate the analysis quality using these criteria:
1. **Completeness**: Does it cover root cause, impact, actions? (0-25 points)
2. **Evidence**: Is it based on concrete data from logs? (0-25 points)
3. **Actionability**: Are recommended actions clear and specific? (0-25 points)
4. **Precision**: Are conclusions logical and well-founded? (0-25 points)

**Response format:**
SCORE: [0-100]
FEEDBACK: [your detailed critique and improvement suggestions]

Be critical but constructive."""

    messages = [
        SystemMessage(content="You are a rigorous technical critic."),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)
    reflection_text = response.content

    # Parse score
    try:
        score_line = [line for line in reflection_text.split('\n') if 'SCORE:' in line][0]
        score = int(score_line.split(':')[1].strip()) / 100.0
    except:
        score = 0.5  # Default if parsing fails

    # Update state
    state["reflection"] = reflection_text
    state["quality_score"] = score
    state["messages"].append(AIMessage(content=f"[ReflectionAgent] Score: {score:.2f}\n{reflection_text}"))

    logger.info(f"Reflection completed (score: {score:.2f})")

    return state


def increment_iteration(state: IncidentState) -> IncidentState:
    """
    Increments iteration counter
    """
    state["iteration"] += 1
    logger.info(f"Incrementing iteration to {state['iteration']}")
    return state


def finalize_output(state: IncidentState) -> IncidentState:
    """
    Finalizes output and saves to Long-Term Memory
    """
    logger.info("Finalizing output")

    # Save successful analysis to LTM
    es = get_elastic_client()

    save_to_long_term_memory(
        es=es,
        agent_name="AnalyserAgent",
        memory_type="decision",
        content=state["analysis"],
        success=True,
        metadata={
            "query": state["query"],
            "quality_score": state["quality_score"],
            "iterations": state["iteration"]
        }
    )

    state["final_output"] = state["analysis"]
    state["messages"].append(AIMessage(content=f"[FINAL] Analysis approved after {state['iteration']} iteration(s)"))

    logger.info(f"Output finalized (iterations: {state['iteration']})")

    return state


# ============================================================================
# CONDITIONAL ROUTING (decisor_router)
# ============================================================================

def decisor_router(state: IncidentState) -> Literal["increment", "finalize"]:
    """
    Decisor Router: Defines next step in Reflection cycle

    Logic:
    - quality_score >= 0.8 -> finalize (success)
    - iteration >= max_iterations -> finalize (limit reached)
    - otherwise -> increment -> analyser (new iteration)
    """
    quality = state["quality_score"]
    iteration = state["iteration"]
    max_iter = state["max_iterations"]

    logger.info(f"Router: quality={quality:.2f}, iteration={iteration}/{max_iter}")

    # Check quality threshold
    if quality >= float(os.getenv("QUALITY_THRESHOLD", "0.8")):
        logger.info("Quality threshold met -> finalize")
        return "finalize"

    # Check max iterations
    if iteration >= max_iter:
        logger.info("Max iterations reached -> finalize")
        return "finalize"

    # Continue reflection loop
    logger.info("Quality below threshold -> retry")
    return "increment"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_reflection_graph() -> StateGraph:
    """
    Creates StateGraph with Reflection Pattern

    Flow:
    START → search → analyser → reflection → decisor_router
                        ↑                         │
                        │                    [increment]
                        └───── increment ─────────┘
                                  │
                             [finalize]
                                  ↓
                              finalize → END

    Note: SearchAgent runs ONCE at the start.
          Reflection loop iterates only on analyser → reflection.
    """
    # Initialize graph
    workflow = StateGraph(IncidentState)

    # Add nodes (3 specialized agents + utility nodes)
    workflow.add_node("search", search_agent)        # NEW: Hybrid search
    workflow.add_node("analyser", analyser_agent)    # LLM analysis
    workflow.add_node("reflection", reflection_agent) # Quality evaluation
    workflow.add_node("increment", increment_iteration)
    workflow.add_node("finalize", finalize_output)

    # Add edges
    workflow.set_entry_point("search")              # CHANGED: Start with search
    workflow.add_edge("search", "analyser")         # NEW: search → analyser
    workflow.add_edge("analyser", "reflection")
    workflow.add_edge("increment", "analyser")      # Loop back to analyser (NOT search)

    # Conditional edge (reflection → increment OR finalize)
    workflow.add_conditional_edges(
        "reflection",
        decisor_router,
        {
            "increment": "increment",  # Retry (increment first)
            "finalize": "finalize"     # Approved
        }
    )

    workflow.add_edge("finalize", END)

    return workflow


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_reflection_analysis(query: str, thread_id: str = "default") -> dict:
    """
    Runs incident analysis with Reflection Pattern

    Args:
        query: User query (e.g.: "database timeout errors")
        thread_id: Thread ID for checkpointing

    Returns:
        dict with final_output, quality_score, iterations
    """
    logger.info("=" * 70)
    logger.info("MULTI-AGENT ORCHESTRATION - REFLECTION PATTERN")
    logger.info("=" * 70)

    # Build graph (without checkpointing for simplicity)
    workflow = create_reflection_graph()
    app = workflow.compile()

    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "search_results": [],
        "analysis": "",
        "reflection": "",
        "final_output": "",
        "quality_score": 0.0,
        "iteration": 1,
        "max_iterations": int(os.getenv("MAX_REFLECTION_ITERATIONS", "3"))
    }

    logger.info(f"Starting analysis: '{query}'")
    logger.info("")

    final_state = app.invoke(initial_state)

    logger.info("")
    logger.info("=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Quality Score: {final_state['quality_score']:.2f}")
    logger.info(f"Iterations: {final_state['iteration']}")
    logger.info("")

    return {
        "final_output": final_state["final_output"],
        "quality_score": final_state["quality_score"],
        "iterations": final_state["iteration"],
        "search_results_count": len(final_state["search_results"])
    }


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys

    print("\n" + "=" * 70)
    print("MULTI-AGENT INCIDENT ANALYSIS")
    print("Reflection Pattern + Elastic Cloud")
    print("=" * 70 + "\n")

    # Check if query provided as argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # Interactive mode
        print("Examples:")
        print("  - database connection timeout")
        print("  - high memory usage")
        print("  - API rate limit exceeded")
        print()
        query = input("Enter your query: ").strip()

    if not query:
        print("ERROR: Query cannot be empty")
        sys.exit(1)

    try:
        # Run analysis
        result = run_reflection_analysis(query)

        # Print results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\n{result['final_output']}\n")
        print("=" * 70)
        print(f"Quality: {result['quality_score']:.0%} | "
              f"Iterations: {result['iterations']} | "
              f"Logs analyzed: {result['search_results_count']}")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\nERROR: Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
