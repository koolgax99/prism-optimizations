import logging
import json
from typing import List, Dict, Any, Tuple
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import time

# Import your existing modules
from src.constants import *
from src.QA_integration import (
    get_llm, get_neo4j_retriever, create_document_retriever_chain,
    retrieve_documents, format_documents, get_rag_chain
)

# Prompts for iterative retrieval
QUERY_DECOMPOSITION_PROMPT = """
You are an expert at analyzing complex questions and breaking them down into simpler sub-questions.

Given the following question, decompose it into 2-5 sub-questions that, when answered together, would provide a comprehensive answer to the original question. Each sub-question should focus on a specific aspect or entity.

Original Question: {question}

Return your response as a JSON object with the following structure:
{{
    "sub_questions": [
        {{
            "id": 1,
            "question": "sub-question text",
            "focus": "main entity or concept this sub-question is about",
            "dependency": null or id of previous sub-question this depends on
        }}
    ],
    "reasoning": "brief explanation of your decomposition strategy"
}}

Example:
Question: "What genetic mutations are associated with SCLC and how do they affect treatment options?"

Response:
{{
    "sub_questions": [
        {{
            "id": 1,
            "question": "What genetic mutations are commonly found in SCLC?",
            "focus": "SCLC genetic mutations",
            "dependency": null
        }},
        {{
            "id": 2,
            "question": "How do these mutations affect SCLC treatment options?",
            "focus": "treatment implications",
            "dependency": 1
        }}
    ],
    "reasoning": "First identify the mutations, then explore their therapeutic implications"
}}
"""

EVIDENCE_SYNTHESIS_PROMPT = """
You are tasked with synthesizing evidence from multiple retrieval steps to create a comprehensive, well-structured answer.

Original Question: {original_question}

Sub-questions and Retrieved Evidence:
{evidence_summary}

Instructions for synthesis:

1. **Create a unified narrative**: DO NOT structure your answer by sub-questions. Instead, synthesize all evidence into a cohesive response that flows naturally from one topic to another.

2. **Avoid redundant headings**: Do not repeat similar section headings. Each section should have a unique, specific title that advances the narrative.

3. **NO source citations**: Do not include source references, citations, or a sources section anywhere in your answer. Focus purely on the synthesized content

4. **Structure guidelines**:
   - Start with a brief overview paragraph
   - Use sections with clear, descriptive headings
   - Within each section, use sub-points or paragraphs to organize information
   - Ensure logical flow between sections
   - End with key takeaways or implications (not a traditional conclusion)

5. **Content requirements**:
   - Integrate evidence across all sub-questions seamlessly
   - Highlight connections and relationships between different aspects
   - Be specific with examples and mechanisms
   - Prioritize the most relevant and impactful information
   - Avoid generic statements; be precise and actionable

6. **Writing style**:
   - Use active voice and clear, concise language
   - Avoid repetition across sections
   - Each paragraph should serve a distinct purpose
   - Technical terms should be explained when first introduced

IMPORTANT: STRICTLY NO OUT OF CONTEXT INFORMATION. Your answer must be grounded solely in the provided evidence. Do not make assumptions or include any information not directly supported by the retrieved documents.

Synthesized Answer:
"""

VERIFICATION_PROMPT = """
You are verifying the answer to ensure it's grounded in the retrieved evidence.

Question: {question}
Answer: {answer}
Evidence: {evidence}

Verify that:
1. All claims in the answer are supported by the evidence
2. No information is hallucinated or assumed
3. The answer fully addresses the question

Return a JSON with:
{{
    "is_verified": true/false,
    "unsupported_claims": ["list of any claims not supported by evidence"],
    "missing_aspects": ["aspects of the question not addressed"],
    "confidence": 0.0-1.0
}}
"""

class IterativeMultiHopRetriever:
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
        
    def decompose_query(self, question: str) -> Dict[str, Any]:
        """Decompose complex query into sub-questions"""
        prompt = ChatPromptTemplate.from_template(QUERY_DECOMPOSITION_PROMPT)
        
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            result = chain.invoke({"question": question})
            logging.info(f"Query decomposed into {len(result['sub_questions'])} sub-questions")
            return result
        except Exception as e:
            logging.error(f"Error decomposing query: {e}")
            # Fallback to single question
            return {
                "sub_questions": [
                    {
                        "id": 1,
                        "question": question,
                        "focus": "main query",
                        "dependency": None
                    }
                ],
                "reasoning": "Using original question as single retrieval"
            }
    
    def retrieve_for_subquestion(self, sub_question: Dict[str, Any], 
                                previous_context: str = "") -> Dict[str, Any]:
        """Retrieve evidence for a specific sub-question"""
        # Enhance sub-question with previous context if there's a dependency
        enhanced_question = sub_question["question"]
        if sub_question.get("dependency") and previous_context:
            enhanced_question = f"Given this context: {previous_context[:500]}...\n\n{enhanced_question}"
        
        # Create messages for retrieval
        messages = [HumanMessage(content=enhanced_question)]
        
        # Retrieve documents
        docs, transformed_question = retrieve_documents(self.doc_retriever, messages)
        
        if docs:
            # Format documents and extract information
            formatted_docs, sources, entities, communities = format_documents(
                docs, self.llm.model_name, self.chat_mode_settings
            )
            
            # Get answer for this sub-question
            rag_chain = get_rag_chain(self.llm)
            ai_response = rag_chain.invoke({
                "messages": [],
                "context": formatted_docs,
                "input": sub_question["question"]
            })
            
            return {
                "sub_question": sub_question,
                "answer": ai_response.content,
                "formatted_docs": formatted_docs,
                "sources": list(sources),
                "entities": entities,
                "communities": communities,
                "docs": docs
            }
        else:
            return {
                "sub_question": sub_question,
                "answer": "No relevant evidence found for this sub-question.",
                "formatted_docs": "",
                "sources": [],
                "entities": {},
                "communities": [],
                "docs": []
            }
    
    def iterative_retrieval(self, question: str) -> List[Dict[str, Any]]:
        """Perform iterative retrieval for all sub-questions"""
        # Decompose the query
        decomposition = self.decompose_query(question)
        sub_questions = decomposition["sub_questions"]
        
        # Sort by dependency order
        sorted_questions = sorted(sub_questions, 
                                key=lambda x: (x.get("dependency") or 0, x["id"]))
        
        retrieval_results = []
        context_map = {}
        
        for sub_q in sorted_questions:
            # Get previous context if there's a dependency
            prev_context = ""
            if sub_q.get("dependency"):
                prev_result = context_map.get(sub_q["dependency"])
                if prev_result:
                    prev_context = prev_result["answer"]
            
            # Retrieve for this sub-question
            result = self.retrieve_for_subquestion(sub_q, prev_context)
            retrieval_results.append(result)
            context_map[sub_q["id"]] = result
            
            logging.info(f"Retrieved evidence for sub-question {sub_q['id']}: {sub_q['question'][:50]}...")
        
        return retrieval_results, decomposition["reasoning"]
    
    def synthesize_answer(self, question: str, retrieval_results: List[Dict[str, Any]]) -> str:
        """Synthesize final answer from all retrieved evidence"""
        # Build evidence summary
        evidence_parts = []
        for i, result in enumerate(retrieval_results, 1):
            sub_q = result["sub_question"]
            evidence_parts.append(
                f"Sub-question {i}: {sub_q['question']}\n"
                f"Focus: {sub_q['focus']}\n"
                f"Answer: {result['answer']}\n"
                f"Sources: {', '.join(result['sources'])}\n"
            )
        
        evidence_summary = "\n---\n".join(evidence_parts)
        
        # Synthesize final answer
        prompt = ChatPromptTemplate.from_template(EVIDENCE_SYNTHESIS_PROMPT)
        chain = prompt | self.llm
        
        response = chain.invoke({
            "original_question": question,
            "evidence_summary": evidence_summary
        })
        
        return response.content
    
    def verify_answer(self, question: str, answer: str, 
                     retrieval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify that the answer is grounded in evidence"""
        # Combine all evidence
        all_evidence = "\n---\n".join([
            f"Sub-question: {r['sub_question']['question']}\n"
            f"Evidence: {r['formatted_docs'][:1000]}..."
            for r in retrieval_results
        ])
        
        prompt = ChatPromptTemplate.from_template(VERIFICATION_PROMPT)
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            verification = chain.invoke({
                "question": question,
                "answer": answer,
                "evidence": all_evidence
            })
            return verification
        except Exception as e:
            logging.error(f"Error verifying answer: {e}")
            return {
                "is_verified": True,
                "unsupported_claims": [],
                "missing_aspects": [],
                "confidence": 0.8
            }
    
    def get_explainable_path(self, retrieval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create an explainable retrieval path"""
        path = {
            "steps": [],
            "entities_traversed": set(),
            "relationships_explored": set(),
            "sources_used": set()
        }
        
        for result in retrieval_results:
            step = {
                "question": result["sub_question"]["question"],
                "focus": result["sub_question"]["focus"],
                "entities": result.get("entities", {}),
                "sources": result["sources"],
                "answer_summary": result["answer"][:200] + "..."
            }
            path["steps"].append(step)
            
            # Aggregate entities and sources
            if "entityids" in result.get("entities", {}):
                path["entities_traversed"].update(result["entities"]["entityids"])
            if "relationshipids" in result.get("entities", {}):
                path["relationships_explored"].update(result["entities"]["relationshipids"])
            path["sources_used"].update(result["sources"])
        
        # Convert sets to lists for JSON serialization
        path["entities_traversed"] = list(path["entities_traversed"])
        path["relationships_explored"] = list(path["relationships_explored"])
        path["sources_used"] = list(path["sources_used"])
        
        return path


def iterative_multihop_qa(graph, model, question, document_names, chat_mode_settings):
    """Main function for iterative multi-hop QA"""
    start_time = time.time()
    
    try:
        # Initialize LLM
        llm, model_name = get_llm(model=model)
        
        # Create iterative retriever
        retriever = IterativeMultiHopRetriever(
            llm=llm,
            graph=graph,
            document_names=document_names,
            chat_mode_settings=chat_mode_settings
        )
        
        # Perform iterative retrieval
        retrieval_results, reasoning = retriever.iterative_retrieval(question)
        
        # Synthesize answer
        final_answer = retriever.synthesize_answer(question, retrieval_results)

        # Aggregate all sources and entities
        all_sources = set()
        all_entities = {"entityids": set(), "relationshipids": set()}
        all_chunks = []
        
        for result in retrieval_results:
            all_sources.update(result["sources"])
            if "entityids" in result.get("entities", {}):
                all_entities["entityids"].update(result["entities"]["entityids"])
            if "relationshipids" in result.get("entities", {}):
                all_entities["relationshipids"].update(result["entities"]["relationshipids"])
            
            # Extract chunk details from docs
            for doc in result.get("docs", []):
                if "chunkdetails" in doc.metadata:
                    all_chunks.extend(doc.metadata["chunkdetails"])
        
        total_time = time.time() - start_time
        
        return {
            "message": final_answer,
            "info": {
                "sources": list(all_sources),
                "model": model_name,
                "mode": "iterative_multihop",
                "response_time": total_time,
                "multi_hop_details": {
                    "decomposition_reasoning": reasoning,
                    "sub_questions": [r["sub_question"] for r in retrieval_results],
                },
                "entities": {
                    "entityids": list(all_entities["entityids"]),
                    "relationshipids": list(all_entities["relationshipids"])
                },
                "nodedetails": {
                    "chunkdetails": all_chunks
                },
            },
            "user": "chatbot"
        }
        
    except Exception as e:
        logging.exception(f"Error in iterative multi-hop QA: {str(e)}")
        return {
            "message": "An error occurred during iterative retrieval",
            "info": {
                "error": str(e),
                "mode": "iterative_multihop"
            },
            "user": "chatbot"
        }