# v1.py

from langchain_neo4j import Neo4jGraph
from src.QA_integration import *
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def create_graph_database_connection(uri, userName, password, database):
    graph = Neo4jGraph(url=uri, database=database, username=userName, password=password, refresh_schema=False, sanitize=True)    
    return graph

def main():
    # Load Neo4j credentials from environment variables
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

    graph = create_graph_database_connection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)

    result = QA_RAG(
        graph=graph,
        model="gpt-4o",
        question="What genetic mutations are associated with SCLC?",
        document_names=None,
        session_id="test_session",
        mode="graph_vector_fulltext",
        write_access=True
        )

    return result

def test_iterative_mode():
    # Load Neo4j credentials from environment variables
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

    graph = create_graph_database_connection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)

    # Test with a complex question
    result = QA_RAG(
        graph=graph,
        model="gpt-4o",
        question="You are assisting a research scientist. Focus on molecular mechanisms, research data, and scientific literature. Question: Describe the molecular pathways in lung cancer",
        document_names=None,
        session_id="test_iterative",
        mode="iterative_multihop",  # Use the new mode
        write_access=True
    )
    
    # Print results
    print("Answer:", result["message"])
        
    return result

if __name__ == "__main__":
    result = test_iterative_mode()
    print(result.get("message", "No message returned"))