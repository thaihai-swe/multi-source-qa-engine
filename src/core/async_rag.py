"""Async RAG pipeline for improved performance"""
import asyncio
from typing import List, Optional
from datetime import datetime

from src.models import RetrievedDocument, RAGResponse
from src.utils import get_logger

logger = get_logger()


class AsyncRAGPipeline:
    """
    Asynchronous RAG pipeline that parallelizes retrieval, reranking, and evaluation.

    Benefits:
    - Parallel retrieval from multiple sources
    - Non-blocking I/O operations
    - Concurrent LLM calls for evaluation metrics
    - Faster overall query processing
    """

    def __init__(self, rag_system):
        """
        Args:
            rag_system: Synchronous RAGSystem instance
        """
        self.rag = rag_system
        logger.info("✅ Async RAG Pipeline initialized")

    async def process_query_async(
        self,
        query: str,
        use_expansion: bool = False
    ) -> RAGResponse:
        """
        Process query asynchronously with parallel operations

        Args:
            query: User's query
            use_expansion: Whether to expand query

        Returns:
            RAGResponse
        """
        logger.info(f"⚡ Async processing query: {query}")
        start_time = datetime.now()

        try:
            # Phase 1: Parallel domain check and query expansion
            domain_check_task = asyncio.create_task(
                self._async_domain_check(query)
            ) if self.rag.config.search.enable_domain_guard else None

            expansion_task = asyncio.create_task(
                self._async_query_expansion(query, use_expansion)
            ) if use_expansion else None

            # Wait for pre-retrieval tasks
            domain_check_result = await domain_check_task if domain_check_task else None
            queries = await expansion_task if expansion_task else [query]

            # Phase 2: Parallel retrieval from multiple sources
            retrieval_tasks = [
                self._async_retrieve(q) for q in queries
            ]
            retrieved_results = await asyncio.gather(*retrieval_tasks)

            # Combine and deduplicate documents
            all_docs = []
            seen_content = set()
            for docs in retrieved_results:
                for doc in docs:
                    if doc.content not in seen_content:
                        all_docs.append(doc)
                        seen_content.add(doc.content)

            if not all_docs:
                return RAGResponse(
                    answer="No relevant documents found.",
                    sources=[],
                    confidence_score=0.0,
                    source_types=[],
                    conversation_context=None,
                    execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    domain_check=domain_check_result
                )

            # Limit to max results
            all_docs = all_docs[:self.rag.config.search.max_results]

            # Phase 3: Parallel reranking and answer generation
            rerank_task = asyncio.create_task(
                self._async_rerank(query, all_docs)
            ) if self.rag.config.search.enable_reranking else None

            generation_task = asyncio.create_task(
                self._async_generate(query, all_docs)
            )

            # Wait for generation and reranking
            answer = await generation_task
            reranked_docs = await rerank_task if rerank_task else all_docs

            # Phase 4: Parallel evaluation, fact-checking, hallucination detection
            eval_tasks = []

            if self.rag.config.evaluation.compute_ragas:
                eval_tasks.append(
                    self._async_evaluate(query, reranked_docs, answer)
                )

            if self.rag.config.evaluation.check_facts:
                eval_tasks.append(
                    self._async_fact_check(answer, reranked_docs)
                )

            if self.rag.config.evaluation.enable_hallucination_detection:
                eval_tasks.append(
                    self._async_hallucination_check(query, answer, reranked_docs)
                )

            # Wait for all evaluation tasks
            eval_results = await asyncio.gather(*eval_tasks) if eval_tasks else []

            # Extract results
            confidence_score = 0.5
            fact_results = None
            hallucination_report = None

            for result in eval_results:
                if isinstance(result, tuple) and result[0] == "ragas":
                    confidence_score = result[1].rag_score
                elif isinstance(result, tuple) and result[0] == "facts":
                    fact_results = result[1]
                elif isinstance(result, tuple) and result[0] == "hallucination":
                    hallucination_report = result[1]

            # Store in history
            self.rag.conversation_history.append({
                "query": query,
                "answer": answer,
                "sources": [doc.source for doc in reranked_docs],
                "timestamp": datetime.now().isoformat()
            })

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            response = RAGResponse(
                answer=answer,
                sources=reranked_docs,
                confidence_score=confidence_score,
                source_types=list(set(doc.source_type for doc in reranked_docs)),
                conversation_context=None,
                execution_time_ms=execution_time,
                fact_check_results=fact_results,
                domain_check=domain_check_result,
                hallucination_report=hallucination_report,
                reranker_applied=self.rag.config.search.enable_reranking
            )

            logger.info(f"⚡ Async query processed in {execution_time:.1f}ms")
            return response

        except Exception as e:
            logger.error(f"❌ Async processing error: {e}")
            return RAGResponse(
                answer=f"Error: {str(e)}",
                sources=[],
                confidence_score=0.0,
                source_types=[],
                conversation_context=None,
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    async def _async_domain_check(self, query: str):
        """Async domain relevance check"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.rag.domain_guard.check_domain_relevance,
            query
        )

    async def _async_query_expansion(self, query: str, should_expand: bool) -> List[str]:
        """Async query expansion"""
        if not should_expand:
            return [query]

        loop = asyncio.get_event_loop()
        variations = await loop.run_in_executor(
            None,
            self.rag.query_expander.expand,
            query,
            4
        )
        return variations

    async def _async_retrieve(self, query: str) -> List[RetrievedDocument]:
        """Async document retrieval"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.rag._retrieve_documents,
            query
        )

    async def _async_rerank(
        self,
        query: str,
        docs: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """Async document reranking"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.rag.reranker.rerank,
            query,
            docs
        )

    async def _async_generate(
        self,
        query: str,
        docs: List[RetrievedDocument]
    ) -> str:
        """Async answer generation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.rag.generator.generate,
            query,
            docs
        )

    async def _async_evaluate(
        self,
        query: str,
        docs: List[RetrievedDocument],
        answer: str
    ):
        """Async RAGAS evaluation"""
        loop = asyncio.get_event_loop()
        context = "\n".join([doc.content for doc in docs])
        metrics = await loop.run_in_executor(
            None,
            self.rag.evaluator.evaluate,
            query,
            context,
            answer
        )
        return ("ragas", metrics)

    async def _async_fact_check(
        self,
        answer: str,
        docs: List[RetrievedDocument]
    ):
        """Async fact-checking"""
        loop = asyncio.get_event_loop()
        context = "\n".join([doc.content for doc in docs])
        results = await loop.run_in_executor(
            None,
            self.rag.fact_checker.check_answer,
            answer,
            context
        )
        return ("facts", results)

    async def _async_hallucination_check(
        self,
        query: str,
        answer: str,
        docs: List[RetrievedDocument]
    ):
        """Async hallucination detection"""
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(
            None,
            self.rag.hallucination_detector.analyze,
            query,
            answer,
            docs,
            self.rag.config.evaluation.auto_mitigate_hallucinations
        )
        return ("hallucination", report)

    async def parallel_batch_queries(
        self,
        queries: List[str],
        max_concurrent: int = 5
    ) -> List[RAGResponse]:
        """
        Process multiple queries in parallel with concurrency limit

        Args:
            queries: List of queries to process
            max_concurrent: Maximum concurrent queries

        Returns:
            List of RAGResponse objects
        """
        logger.info(f"⚡ Processing {len(queries)} queries with max {max_concurrent} concurrent")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(query: str):
            async with semaphore:
                return await self.process_query_async(query)

        tasks = [process_with_semaphore(q) for q in queries]
        results = await asyncio.gather(*tasks)

        logger.info(f"✅ Completed {len(results)} queries")
        return results


def run_async_query(rag_system, query: str, **kwargs) -> RAGResponse:
    """
    Convenience function to run async query from synchronous code

    Args:
        rag_system: RAGSystem instance
        query: Query string
        **kwargs: Additional arguments for process_query_async

    Returns:
        RAGResponse
    """
    pipeline = AsyncRAGPipeline(rag_system)
    return asyncio.run(pipeline.process_query_async(query, **kwargs))


async def batch_queries_async(
    rag_system,
    queries: List[str],
    max_concurrent: int = 5
) -> List[RAGResponse]:
    """
    Process multiple queries asynchronously

    Args:
        rag_system: RAGSystem instance
        queries: List of queries
        max_concurrent: Max concurrent queries

    Returns:
        List of responses
    """
    pipeline = AsyncRAGPipeline(rag_system)
    return await pipeline.parallel_batch_queries(queries, max_concurrent)


__all__ = ["AsyncRAGPipeline", "run_async_query", "batch_queries_async"]
