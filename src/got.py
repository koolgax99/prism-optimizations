import logging
import time
from typing import Dict, Any, List

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.QA_integration import (
    get_llm, get_neo4j_retriever, create_document_retriever_chain,
    retrieve_documents, format_documents, get_rag_chain
)

# Prompts for Graph of Thoughts
THOUGHT_GENERATION_PROMPT = """
You are an expert at reasoning and breaking down complex questions.
Given the original question and the current state of gathered evidence, generate a set of diverse and insightful next-step questions or hypotheses to explore.
These "thoughts" should aim to uncover new information, connect existing evidence, or validate hypotheses.

Original Question: {question}
Current Evidence:
{evidence}

Generate 2-4 distinct thoughts. Each thought should be a concise question or a statement to investigate.
Return your response as a JSON object with the following structure:
{{
    "thoughts": [
        "thought 1",
        "thought 2",
        ...
    ]
}}
"""

THOUGHT_EVALUATION_PROMPT = """
You are a critical thinker and evaluator.
Given the original question and a set of generated "thoughts", evaluate the potential of each thought to contribute to a comprehensive answer.
Assign a score from 0.0 to 1.0 to each thought, where 1.0 is the most promising.

Original Question: {question}
Generated Thoughts:
{thoughts}

Return your response as a JSON object with the following structure:
{{
    "scores": [
        {{"thought": "thought 1", "score": 0.9}},
        {{"thought": "thought 2", "score": 0.7}},
        ...
    ]
}}
"""

SYNTHESIS_PROMPT = """
You are a master synthesizer of information.
Given the original question and a collection of evidence gathered from a graph of thoughts, synthesize a comprehensive and coherent final answer.
Structure your answer logically, addressing all aspects of the original question.

Original Question: {original_question}

Evidence from Thought Exploration:
{evidence_summary}

Synthesized Answer:
"""


class GraphOfThoughtsRetriever:
    def __init__(self, llm, graph, document_names, chat_mode_settings):
        self.llm = llm
        self.graph = graph
        self.document_names = document_names
        self.chat_mode_settings = chat_mode_settings
        self.retriever = get_neo4j_retriever(
            graph=graph,
            chat_mode_settings=chat_mode_settings,
            document_names=document_names
        )
        self.doc_retriever = create_document_retriever_chain(llm, self.retriever)

    def generate_thoughts(self, question: str, evidence: str) -> List[str]:
        prompt = ChatPromptTemplate.from_template(THOUGHT_GENERATION_PROMPT)
        chain = prompt | self.llm | JsonOutputParser()
        try:
            result = chain.invoke({"question": question, "evidence": evidence})
            return result.get("thoughts", [])
        except Exception as e:
            logging.error(f"Error generating thoughts: {e}")
            return [question]

    def evaluate_thoughts(self, question: str, thoughts: List[str]) -> List[Dict[str, Any]]:
        prompt = ChatPromptTemplate.from_template(THOUGHT_EVALUATION_PROMPT)
        chain = prompt | self.llm | JsonOutputParser()
        try:
            result = chain.invoke({"question": question, "thoughts": thoughts})
            return sorted(result.get("scores", []), key=lambda x: x["score"], reverse=True)
        except Exception as e:
            logging.error(f"Error evaluating thoughts: {e}")
            return [{"thought": thought, "score": 0.5} for thought in thoughts]

    def retrieve_for_thought(self, thought: str) -> Dict[str, Any]:
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content=thought)]
        docs, _ = retrieve_documents(self.doc_retriever, messages)

        if docs:
            formatted_docs, sources, entities, communities = format_documents(
                docs, self.llm.model_name, self.chat_mode_settings
            )
            rag_chain = get_rag_chain(self.llm)
            ai_response = rag_chain.invoke({
                "messages": [],
                "context": formatted_docs,
                "input": thought
            })
            return {
                "thought": thought,
                "answer": ai_response.content,
                "sources": list(sources),
                "entities": entities,
                "communities": communities,
            }
        else:
            return {
                "thought": thought,
                "answer": "No relevant information found for this thought.",
                "sources": [],
                "entities": {},
                "communities": [],
            }

    def synthesize_answer(self, question: str, evidence_summary: str) -> str:
        prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
        chain = prompt | self.llm
        response = chain.invoke({
            "original_question": question,
            "evidence_summary": evidence_summary
        })
        return response.content

    def conduct_graph_of_thought(self, question: str, max_iterations: int = 3, thoughts_per_iteration: int = 2):
        evidence_graph = {"nodes": [], "edges": []}
        all_retrieved_info = []

        initial_thoughts = self.generate_thoughts(question, "Initial question.")
        evaluated_thoughts = self.evaluate_thoughts(question, initial_thoughts)

        queue = [thought['thought'] for thought in evaluated_thoughts[:thoughts_per_iteration]]
        processed_thoughts = set(queue)

        for i in range(max_iterations):
            if not queue:
                break

            current_thought = queue.pop(0)
            logging.info(f"Iteration {i + 1}: Processing thought: {current_thought}")

            retrieved_info = self.retrieve_for_thought(current_thought)
            all_retrieved_info.append(retrieved_info)

            current_evidence = "\n\n".join([f"Thought: {info['thought']}\nAnswer: {info['answer']}" for info in all_retrieved_info])
            new_thoughts = self.generate_thoughts(question, current_evidence)
            evaluated_thoughts = self.evaluate_thoughts(question, new_thoughts)

            for thought_info in evaluated_thoughts:
                thought = thought_info["thought"]
                if thought not in processed_thoughts:
                    queue.append(thought)
                    processed_thoughts.add(thought)

        evidence_summary = "\n---\n".join(
            [f"Thought: {info['thought']}\nAnswer: {info['answer']}\nSources: {', '.join(info['sources'])}"
             for info in all_retrieved_info]
        )

        final_answer = self.synthesize_answer(question, evidence_summary)
        return final_answer, all_retrieved_info


def graph_of_thought_qa(graph, model, question, document_names, chat_mode_settings):
    start_time = time.time()

    try:
        llm, model_name = get_llm(model=model)
        retriever = GraphOfThoughtsRetriever(
            llm=llm,
            graph=graph,
            document_names=document_names,
            chat_mode_settings=chat_mode_settings
        )

        final_answer, all_retrieved_info = retriever.conduct_graph_of_thought(question)

        all_sources = set()
        all_entities = {"entityids": set(), "relationshipids": set()}

        for info in all_retrieved_info:
            all_sources.update(info["sources"])
            if "entityids" in info.get("entities", {}):
                all_entities["entityids"].update(info["entities"]["entityids"])
            if "relationshipids" in info.get("entities", {}):
                all_entities["relationshipids"].update(info["entities"]["relationshipids"])

        total_time = time.time() - start_time

        return {
            "message": final_answer,
            "info": {
                "sources": list(all_sources),
                "model": model_name,
                "mode": "graph_of_thought",
                "response_time": total_time,
                "thought_process": all_retrieved_info,
                "entities": {
                    "entityids": list(all_entities["entityids"]),
                    "relationshipids": list(all_entities["relationshipids"])
                },
            },
            "user": "chatbot"
        }

    except Exception as e:
        logging.exception(f"Error in Graph of Thought QA: {str(e)}")
        return {
            "message": "An error occurred during the Graph of Thought retrieval process.",
            "info": {
                "error": str(e),
                "mode": "graph_of_thought"
            },
            "user": "chatbot"
        }