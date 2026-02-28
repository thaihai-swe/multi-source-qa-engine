"""Agentic RAG system with ReAct (Reasoning + Acting) pattern"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI

from src.config import get_config
from src.models import RetrievedDocument, RAGResponse
from src.utils import get_logger

logger = get_logger()


class AgentAction(Enum):
    """Available actions for the RAG agent"""
    SIMPLE_RETRIEVAL = "simple_retrieval"
    MULTI_HOP_REASONING = "multi_hop_reasoning"
    QUERY_DECOMPOSITION = "query_decomposition"
    QUERY_EXPANSION = "query_expansion"
    SEARCH_WIKIPEDIA = "search_wikipedia"
    SEARCH_WEB = "search_web"
    SEARCH_LOCAL_DOCS = "search_local_docs"
    SELF_CRITIQUE = "self_critique"
    RE_RETRIEVE = "re_retrieve"
    FINALIZE_ANSWER = "finalize_answer"


@dataclass
class AgentStep:
    """Single step in agent reasoning"""
    step_number: int
    thought: str
    action: AgentAction
    action_input: Dict[str, Any]
    observation: str
    confidence: float = 0.0


@dataclass
class AgentResponse:
    """Response from agentic RAG"""
    final_answer: str
    reasoning_steps: List[AgentStep]
    total_steps: int
    sources: List[RetrievedDocument]
    confidence: float
    strategy_used: str


class AgenticRAG:
    """
    Autonomous RAG agent that decides its own retrieval and reasoning strategy.

    Uses ReAct (Reasoning + Acting) pattern:
    1. THOUGHT: Analyze the query and plan action
    2. ACTION: Execute the chosen action
    3. OBSERVATION: Observe the result
    4. Repeat until confident answer is found
    """

    def __init__(self, rag_system, max_steps: int = 5):
        """
        Args:
            rag_system: The RAGSystem instance
            max_steps: Maximum reasoning steps before returning
        """
        self.rag = rag_system
        self.max_steps = max_steps

        config = get_config()
        self.client = OpenAI(
            base_url=config.llm.api_base_url,
            api_key=config.llm.api_key
        )
        self.model = config.llm.model_name

        logger.info("✅ Agentic RAG initialized")

    def process_query(self, query: str) -> AgentResponse:
        """
        Process query autonomously, choosing optimal strategy

        Args:
            query: User's query

        Returns:
            AgentResponse with reasoning trace
        """
        logger.info(f"🤖 Agent processing query: {query}")

        steps = []
        all_retrieved_docs = []
        current_query = query
        answer = None

        for step_num in range(1, self.max_steps + 1):
            # THINK: Analyze situation and decide next action
            thought, action, action_input = self._think(
                query=query,
                current_context=steps,
                retrieved_docs=all_retrieved_docs
            )

            logger.info(f"💭 Step {step_num} - Thought: {thought[:100]}...")
            logger.info(f"🎬 Action: {action.value}")

            # ACT: Execute the chosen action
            observation, docs, confidence = self._act(action, action_input, current_query)

            if docs:
                all_retrieved_docs.extend(docs)

            # Record step
            step = AgentStep(
                step_number=step_num,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                confidence=confidence
            )
            steps.append(step)

            logger.info(f"👁️  Observation: {observation[:100]}...")

            # Check if action was to finalize
            if action == AgentAction.FINALIZE_ANSWER:
                answer = observation
                break

            # Check if confidence is high enough
            if confidence > 0.85 and all_retrieved_docs:
                # Generate final answer
                answer = self.rag.generator.generate(query, all_retrieved_docs[:10])
                break

        # If no answer yet, generate one from collected documents
        if not answer and all_retrieved_docs:
            answer = self.rag.generator.generate(query, all_retrieved_docs[:10])
        elif not answer:
            answer = "I apologize, but I couldn't find sufficient information to answer your question."

        # Determine strategy used
        actions_used = [s.action for s in steps]
        if AgentAction.MULTI_HOP_REASONING in actions_used:
            strategy = "multi_hop_reasoning"
        elif AgentAction.QUERY_DECOMPOSITION in actions_used:
            strategy = "query_decomposition"
        elif AgentAction.QUERY_EXPANSION in actions_used:
            strategy = "query_expansion"
        else:
            strategy = "simple_retrieval"

        # Calculate overall confidence
        avg_confidence = sum(s.confidence for s in steps) / len(steps) if steps else 0.5

        logger.info(f"✅ Agent completed in {len(steps)} steps with {len(all_retrieved_docs)} docs")

        return AgentResponse(
            final_answer=answer,
            reasoning_steps=steps,
            total_steps=len(steps),
            sources=all_retrieved_docs[:10],  # Top 10 sources
            confidence=avg_confidence,
            strategy_used=strategy
        )

    def _think(
        self,
        query: str,
        current_context: List[AgentStep],
        retrieved_docs: List[RetrievedDocument]
    ) -> tuple[str, AgentAction, Dict]:
        """
        Agent's thinking step - decide what to do next

        Returns:
            (thought, action, action_input)
        """
        # Build context from previous steps
        context_str = ""
        if current_context:
            context_str = "\n".join([
                f"Step {s.step_number}: {s.action.value} -> {s.observation[:100]}..."
                for s in current_context
            ])

        # Prompt for reasoning
        reasoning_prompt = f"""You are an intelligent RAG agent. Analyze the query and decide the best action.

Query: {query}

Previous steps:
{context_str if context_str else "None (first step)"}

Documents retrieved so far: {len(retrieved_docs)}

Available actions:
1. simple_retrieval - Standard retrieval from knowledge base
2. multi_hop_reasoning - Break into sub-questions for complex queries
3. query_decomposition - Split multi-aspect questions
4. query_expansion - Generate query variations for better coverage
5. search_wikipedia - Search Wikipedia specifically
6. search_local_docs - Search local documents only
7. self_critique - Evaluate if current answer is sufficient
8. re_retrieve - Try retrieval with modified query
9. finalize_answer - Generate final answer from collected information

Analyze the query and decide:
1. What action should I take next?
2. Why is this the best choice?
3. What information do I need?

Respond in this format:
THOUGHT: [your reasoning]
ACTION: [action name]
INPUT: [query or parameters for the action]
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strategic reasoning agent for a RAG system."},
                    {"role": "user", "content": reasoning_prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )

            result = response.choices[0].message.content

            # Parse response
            lines = result.strip().split('\n')
            thought = ""
            action_str = ""
            input_str = ""

            for line in lines:
                if line.startswith("THOUGHT:"):
                    thought = line.replace("THOUGHT:", "").strip()
                elif line.startswith("ACTION:"):
                    action_str = line.replace("ACTION:", "").strip().lower().replace(" ", "_")
                elif line.startswith("INPUT:"):
                    input_str = line.replace("INPUT:", "").strip()

            # Map to action enum
            try:
                action = AgentAction(action_str)
            except ValueError:
                # Default to simple retrieval if parsing fails
                logger.warning(f"Unknown action '{action_str}', defaulting to simple_retrieval")
                action = AgentAction.SIMPLE_RETRIEVAL

            action_input = {"query": input_str or query}

            return thought, action, action_input

        except Exception as e:
            logger.error(f"Error in thinking step: {e}")
            # Default behavior
            if not current_context:
                return "Starting with simple retrieval", AgentAction.SIMPLE_RETRIEVAL, {"query": query}
            else:
                return "Finalizing answer", AgentAction.FINALIZE_ANSWER, {"query": query}

    def _act(
        self,
        action: AgentAction,
        action_input: Dict,
        original_query: str
    ) -> tuple[str, List[RetrievedDocument], float]:
        """
        Execute an action

        Returns:
            (observation, retrieved_docs, confidence)
        """
        query = action_input.get("query", original_query)

        try:
            if action == AgentAction.SIMPLE_RETRIEVAL:
                docs = self.rag._retrieve_documents(query)
                if docs:
                    return f"Retrieved {len(docs)} documents", docs, 0.7
                return "No documents found", [], 0.2

            elif action == AgentAction.MULTI_HOP_REASONING:
                # Use multi-hop reasoner
                steps = self.rag.multi_hop_reasoner.decompose(query, max_steps=3)
                all_docs = []
                for step in steps:
                    # step is a string, not an object
                    docs = self.rag._retrieve_documents(step)
                    if docs:
                        all_docs.extend(docs)
                return f"Multi-hop reasoning with {len(steps)} steps, retrieved {len(all_docs)} docs", all_docs, 0.8

            elif action == AgentAction.QUERY_DECOMPOSITION:
                decomposition = self.rag.self_query_decomposer.decompose(query)
                all_docs = []
                if decomposition and decomposition.is_complex:
                    for sub_q in decomposition.sub_queries:
                        docs = self.rag._retrieve_documents(sub_q.query)
                        if docs:
                            all_docs.extend(docs)
                return f"Decomposed into sub-queries, retrieved {len(all_docs)} docs", all_docs, 0.75

            elif action == AgentAction.QUERY_EXPANSION:
                variations = self.rag.query_expander.expand(query, num_variations=3)
                all_docs = []
                for var in variations:
                    docs = self.rag._retrieve_documents(var)
                    if docs:
                        all_docs.extend(docs[:3])  # Top 3 from each variation
                return f"Expanded query into {len(variations)} variations, retrieved {len(all_docs)} docs", all_docs, 0.7

            elif action == AgentAction.SEARCH_WIKIPEDIA:
                # Search Wikipedia specifically
                docs = self.rag._retrieve_documents(query)
                wiki_docs = [d for d in docs if d.source_type == 'wikipedia']
                return f"Found {len(wiki_docs)} Wikipedia articles", wiki_docs, 0.65

            elif action == AgentAction.SEARCH_LOCAL_DOCS:
                docs = self.rag._retrieve_documents(query)
                local_docs = [d for d in docs if d.source_type in ['file', 'pdf']]
                return f"Found {len(local_docs)} local documents", local_docs, 0.65

            elif action == AgentAction.SELF_CRITIQUE:
                # Evaluate if we have enough information
                if len(action_input.get("docs", [])) >= 3:
                    return "Sufficient information gathered", [], 0.85
                return "Need more information", [], 0.4

            elif action == AgentAction.RE_RETRIEVE:
                # Try retrieval with modified query
                docs = self.rag._retrieve_documents(query)
                return f"Re-retrieved {len(docs)} documents", docs, 0.6

            elif action == AgentAction.FINALIZE_ANSWER:
                # Generate final answer from all context
                docs_to_use = action_input.get("docs", [])
                if docs_to_use:
                    answer = self.rag.generator.generate(original_query, docs_to_use[:10])
                    return answer, docs_to_use, 0.9
                return "No documents available for answer generation", [], 0.3

            else:
                return f"Unknown action: {action}", [], 0.3

        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            return f"Error: {str(e)}", [], 0.2


__all__ = ["AgenticRAG", "AgentAction", "AgentStep", "AgentResponse"]
