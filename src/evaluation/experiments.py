"""Experimentation framework for RAG optimization"""
import json
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from src.utils import get_logger
from src.evaluation import RAGASEvaluator
from src.models import RAGASMetrics

logger = get_logger()


@dataclass
class ExperimentResult:
    """Result from a single experiment"""
    experiment_id: str
    experiment_type: str  # "chunk_size", "top_k", "reranking", etc.
    timestamp: str
    config: Dict[str, Any]

    # Performance metrics
    avg_retrieval_time_ms: float
    avg_generation_time_ms: float
    avg_total_time_ms: float

    # Quality metrics
    avg_rag_score: float
    avg_context_relevance: float
    avg_answer_relevance: float
    avg_faithfulness: float

    # Retrieval metrics
    avg_docs_retrieved: float
    avg_precision_at_k: Optional[float] = None

    # Test queries
    num_test_queries: int = 0
    test_queries: List[str] = None

    def to_dict(self):
        return asdict(self)


class ChunkSizeExperiment:
    """Experiment with different document chunk sizes"""

    def __init__(self, rag_system):
        self.rag = rag_system
        self.evaluator = RAGASEvaluator()

    def run_experiment(
        self,
        test_queries: List[str],
        chunk_sizes: List[int] = [128, 256, 512, 1024, 2048],
        overlap_ratio: float = 0.1
    ) -> List[ExperimentResult]:
        """
        Test different chunk sizes and measure impact

        Args:
            test_queries: List of test queries
            chunk_sizes: Chunk sizes to test
            overlap_ratio: Overlap between chunks (as ratio of chunk size)

        Returns:
            List of experiment results
        """
        logger.info(f"🧪 Starting chunk size experiment with {len(test_queries)} queries")
        results = []

        for chunk_size in chunk_sizes:
            logger.info(f"Testing chunk size: {chunk_size}")

            # Update chunk size config
            original_chunk_size = self.rag.chunker.max_chunk_size
            original_overlap = self.rag.chunker.overlap

            self.rag.chunker.max_chunk_size = chunk_size
            self.rag.chunker.overlap = int(chunk_size * overlap_ratio)

            # Run queries and collect metrics
            metrics_list = []
            retrieval_times = []
            generation_times = []
            total_times = []
            docs_retrieved = []

            for query in test_queries:
                start_time = time.time()

                # Retrieve documents
                retrieval_start = time.time()
                docs = self.rag._retrieve_documents(query)
                retrieval_time = (time.time() - retrieval_start) * 1000

                if not docs:
                    continue

                # Generate answer
                generation_start = time.time()
                answer = self.rag.generator.generate(query, docs)
                generation_time = (time.time() - generation_start) * 1000

                total_time = (time.time() - start_time) * 1000

                # Evaluate
                context = "\n".join([doc.content for doc in docs])
                metrics = self.evaluator.evaluate(query, context, answer)

                metrics_list.append(metrics)
                retrieval_times.append(retrieval_time)
                generation_times.append(generation_time)
                total_times.append(total_time)
                docs_retrieved.append(len(docs))

            # Aggregate metrics
            if metrics_list:
                result = ExperimentResult(
                    experiment_id=f"chunk_size_{chunk_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    experiment_type="chunk_size",
                    timestamp=datetime.now().isoformat(),
                    config={
                        "chunk_size": chunk_size,
                        "overlap": int(chunk_size * overlap_ratio),
                        "overlap_ratio": overlap_ratio
                    },
                    avg_retrieval_time_ms=sum(retrieval_times) / len(retrieval_times),
                    avg_generation_time_ms=sum(generation_times) / len(generation_times),
                    avg_total_time_ms=sum(total_times) / len(total_times),
                    avg_rag_score=sum(m.rag_score for m in metrics_list) / len(metrics_list),
                    avg_context_relevance=sum(m.context_relevance for m in metrics_list) / len(metrics_list),
                    avg_answer_relevance=sum(m.answer_relevance for m in metrics_list) / len(metrics_list),
                    avg_faithfulness=sum(m.faithfulness for m in metrics_list) / len(metrics_list),
                    avg_docs_retrieved=sum(docs_retrieved) / len(docs_retrieved),
                    num_test_queries=len(test_queries),
                    test_queries=test_queries
                )
                results.append(result)
                logger.info(f"✅ Chunk size {chunk_size}: RAG Score = {result.avg_rag_score:.2f}")

            # Restore original config
            self.rag.chunker.max_chunk_size = original_chunk_size
            self.rag.chunker.overlap = original_overlap

        return results

    def find_optimal_chunk_size(
        self,
        test_queries: List[str],
        chunk_sizes: List[int] = [128, 256, 512, 1024, 2048]
    ) -> Tuple[int, ExperimentResult]:
        """
        Find the optimal chunk size based on RAG score

        Returns:
            (optimal_chunk_size, best_result)
        """
        results = self.run_experiment(test_queries, chunk_sizes)

        if not results:
            return None, None

        # Find best by RAG score
        best_result = max(results, key=lambda r: r.avg_rag_score)
        optimal_chunk_size = best_result.config["chunk_size"]

        logger.info(f"🎯 Optimal chunk size: {optimal_chunk_size} (RAG Score: {best_result.avg_rag_score:.2f})")
        return optimal_chunk_size, best_result


class TopKExperiment:
    """Experiment with different retrieval top-k values"""

    def __init__(self, rag_system):
        self.rag = rag_system
        self.evaluator = RAGASEvaluator()

    def run_experiment(
        self,
        test_queries: List[str],
        k_values: List[int] = [1, 3, 5, 10, 15, 20]
    ) -> List[ExperimentResult]:
        """
        Test different top-k values for retrieval

        Args:
            test_queries: List of test queries
            k_values: Different k values to test

        Returns:
            List of experiment results
        """
        logger.info(f"🧪 Starting top-k experiment with {len(test_queries)} queries")
        results = []

        original_max_results = self.rag.config.search.max_results

        for k in k_values:
            logger.info(f"Testing top-k: {k}")

            # Update k config
            self.rag.config.search.max_results = k

            # Run queries and collect metrics
            metrics_list = []
            retrieval_times = []
            generation_times = []
            total_times = []
            docs_retrieved = []

            for query in test_queries:
                start_time = time.time()

                # Retrieve documents
                retrieval_start = time.time()
                docs = self.rag._retrieve_documents(query)
                retrieval_time = (time.time() - retrieval_start) * 1000

                if not docs:
                    continue

                # Generate answer
                generation_start = time.time()
                answer = self.rag.generator.generate(query, docs)
                generation_time = (time.time() - generation_start) * 1000

                total_time = (time.time() - start_time) * 1000

                # Evaluate
                context = "\n".join([doc.content for doc in docs])
                metrics = self.evaluator.evaluate(query, context, answer)

                metrics_list.append(metrics)
                retrieval_times.append(retrieval_time)
                generation_times.append(generation_time)
                total_times.append(total_time)
                docs_retrieved.append(len(docs))

            # Aggregate metrics
            if metrics_list:
                result = ExperimentResult(
                    experiment_id=f"top_k_{k}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    experiment_type="top_k",
                    timestamp=datetime.now().isoformat(),
                    config={"top_k": k},
                    avg_retrieval_time_ms=sum(retrieval_times) / len(retrieval_times),
                    avg_generation_time_ms=sum(generation_times) / len(generation_times),
                    avg_total_time_ms=sum(total_times) / len(total_times),
                    avg_rag_score=sum(m.rag_score for m in metrics_list) / len(metrics_list),
                    avg_context_relevance=sum(m.context_relevance for m in metrics_list) / len(metrics_list),
                    avg_answer_relevance=sum(m.answer_relevance for m in metrics_list) / len(metrics_list),
                    avg_faithfulness=sum(m.faithfulness for m in metrics_list) / len(metrics_list),
                    avg_docs_retrieved=sum(docs_retrieved) / len(docs_retrieved),
                    num_test_queries=len(test_queries),
                    test_queries=test_queries
                )
                results.append(result)
                logger.info(f"✅ Top-k {k}: RAG Score = {result.avg_rag_score:.2f}, Avg Time = {result.avg_total_time_ms:.0f}ms")

            # Restore original config
            self.rag.config.search.max_results = original_max_results

        return results

    def find_optimal_k(
        self,
        test_queries: List[str],
        k_values: List[int] = [1, 3, 5, 10, 15, 20],
        quality_weight: float = 0.7,
        speed_weight: float = 0.3
    ) -> Tuple[int, ExperimentResult]:
        """
        Find optimal k balancing quality and speed

        Args:
            test_queries: Test queries
            k_values: K values to test
            quality_weight: Weight for quality score (0-1)
            speed_weight: Weight for speed score (0-1)

        Returns:
            (optimal_k, best_result)
        """
        results = self.run_experiment(test_queries, k_values)

        if not results:
            return None, None

        # Normalize metrics and compute weighted score
        max_time = max(r.avg_total_time_ms for r in results)
        min_time = min(r.avg_total_time_ms for r in results)

        scored_results = []
        for result in results:
            # Normalize RAG score (already 0-1)
            quality_score = result.avg_rag_score

            # Normalize time (inverted - lower is better)
            if max_time > min_time:
                speed_score = 1 - (result.avg_total_time_ms - min_time) / (max_time - min_time)
            else:
                speed_score = 1.0

            # Weighted combination
            combined_score = quality_weight * quality_score + speed_weight * speed_score
            scored_results.append((combined_score, result))

        best_score, best_result = max(scored_results, key=lambda x: x[0])
        optimal_k = best_result.config["top_k"]

        logger.info(f"🎯 Optimal top-k: {optimal_k} (Score: {best_score:.2f})")
        return optimal_k, best_result


class ExperimentRunner:
    """Central experiment runner and results manager"""

    def __init__(self, rag_system, output_dir: str = "./experiments"):
        self.rag = rag_system
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.chunk_size_exp = ChunkSizeExperiment(rag_system)
        self.top_k_exp = TopKExperiment(rag_system)

        self.results = []

    def save_results(self, results: List[ExperimentResult], filename: str):
        """Save experiment results to JSON"""
        filepath = self.output_dir / f"{filename}.json"

        with open(filepath, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

        logger.info(f"💾 Results saved to {filepath}")

    def generate_report(self, results: List[ExperimentResult], filename: str = "experiment_report.html"):
        """Generate HTML report from experiment results"""
        filepath = self.output_dir / filename

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RAG Optimization Experiments</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1, h2 {{ color: #333; }}
        .experiment {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .config {{ background: #e3f2fd; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
        .metric-box {{ background: #f5f5f5; padding: 15px; border-radius: 4px; text-align: center; }}
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #1976d2; margin: 5px 0; }}
        .best {{ border: 3px solid #4caf50; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #1976d2; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <h1>🧪 RAG Optimization Experiments</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Total Experiments: {len(results)}</p>

    <h2>Summary Table</h2>
    <table>
        <tr>
            <th>Experiment Type</th>
            <th>Configuration</th>
            <th>RAG Score</th>
            <th>Avg Time (ms)</th>
            <th>Faithfulness</th>
            <th>Context Rel.</th>
        </tr>
        {''.join(f'''
        <tr>
            <td>{r.experiment_type}</td>
            <td>{json.dumps(r.config)}</td>
            <td>{r.avg_rag_score:.3f}</td>
            <td>{r.avg_total_time_ms:.0f}</td>
            <td>{r.avg_faithfulness:.3f}</td>
            <td>{r.avg_context_relevance:.3f}</td>
        </tr>
        ''' for r in results)}
    </table>

    <h2>Detailed Results</h2>
"""

        # Group by experiment type
        by_type = defaultdict(list)
        for result in results:
            by_type[result.experiment_type].append(result)

        for exp_type, exp_results in by_type.items():
            best_result = max(exp_results, key=lambda r: r.avg_rag_score)

            html += f"<h3>{exp_type.replace('_', ' ').title()}</h3>\n"

            for result in exp_results:
                is_best = result == best_result
                html += f'''
    <div class="experiment {'best' if is_best else ''}">
        <h4>{result.experiment_id} {'🏆 BEST' if is_best else ''}</h4>
        <div class="config">
            <strong>Configuration:</strong> {json.dumps(result.config, indent=2)}
        </div>

        <div class="metrics">
            <div class="metric-box">
                <div class="metric-label">RAG Score</div>
                <div class="metric-value">{result.avg_rag_score:.3f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total Time</div>
                <div class="metric-value">{result.avg_total_time_ms:.0f}ms</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Faithfulness</div>
                <div class="metric-value">{result.avg_faithfulness:.3f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Context Relevance</div>
                <div class="metric-value">{result.avg_context_relevance:.3f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Answer Relevance</div>
                <div class="metric-value">{result.avg_answer_relevance:.3f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Docs Retrieved</div>
                <div class="metric-value">{result.avg_docs_retrieved:.1f}</div>
            </div>
        </div>

        <p><strong>Test Queries:</strong> {result.num_test_queries}</p>
    </div>
'''

        html += "</body></html>"

        with open(filepath, 'w') as f:
            f.write(html)

        logger.info(f"📊 Report generated: {filepath}")
        return filepath


__all__ = [
    "ExperimentResult",
    "ChunkSizeExperiment",
    "TopKExperiment",
    "ExperimentRunner"
]
