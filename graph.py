from langgraph.graph import StateGraph, START, END

from state import AgentState
from parser import parse_resumes
from embedder import chunk_and_embed
from matcher import match_resumes


def build_graph():
    """Build and compile the LangGraph resume ranking pipeline."""
    graph = StateGraph(AgentState)

    graph.add_node("parse_resumes", parse_resumes)
    graph.add_node("chunk_and_embed", chunk_and_embed)
    graph.add_node("match_resumes", match_resumes)

    graph.add_edge(START, "parse_resumes")
    graph.add_edge("parse_resumes", "chunk_and_embed")
    graph.add_edge("chunk_and_embed", "match_resumes")
    graph.add_edge("match_resumes", END)

    return graph.compile()
