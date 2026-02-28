"""Observability and performance monitoring for RAG system"""
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from pathlib import Path
from src.utils import get_logger

logger = get_logger()


@dataclass
class QueryMetrics:
    """Metrics for a single query"""
    query_id: str
    query: str
    timestamp: str

    # Timing metrics (milliseconds)
    total_time_ms: float
    retrieval_time_ms: float
    reranking_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    evaluation_time_ms: float = 0.0

    # Retrieval metrics
    num_docs_retrieved: int = 0
    num_docs_reranked: int = 0
    retrieval_method: str = "hybrid"

    # Quality metrics
    confidence_score: float = 0.0
    rag_score: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    context_relevance: Optional[float] = None

    # Safety metrics
    input_risk_level: str = "LOW"
    output_risk_level: str = "LOW"
    guardrails_passed: bool = True

    # Source tracking
    sources: List[str] = field(default_factory=list)
    source_types: List[str] = field(default_factory=list)

    # Answer metadata
    answer_length: int = 0
    num_tokens_generated: Optional[int] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class SystemMetrics:
    """Aggregate system performance metrics"""
    period_start: str
    period_end: str
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0

    avg_total_time_ms: float = 0.0
    avg_retrieval_time_ms: float = 0.0
    avg_generation_time_ms: float = 0.0

    avg_confidence_score: float = 0.0
    avg_rag_score: float = 0.0

    total_docs_retrieved: int = 0
    avg_docs_per_query: float = 0.0

    guardrail_violations: int = 0

    def to_dict(self):
        return asdict(self)


class PerformanceTracker:
    """Track performance of individual operations"""

    def __init__(self):
        self.timers = {}

    def start(self, operation: str):
        """Start timing an operation"""
        self.timers[operation] = time.time()

    def end(self, operation: str) -> float:
        """End timing and return duration in milliseconds"""
        if operation not in self.timers:
            logger.warning(f"Timer for '{operation}' was not started")
            return 0.0

        duration_ms = (time.time() - self.timers[operation]) * 1000
        del self.timers[operation]
        return duration_ms

    def clear(self):
        """Clear all timers"""
        self.timers.clear()


class QueryLogger:
    """Log all queries, retrieved chunks, and generated answers"""

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.queries = []

    def log_query(
        self,
        query: str,
        retrieved_docs: List[Dict],
        answer: str,
        metrics: QueryMetrics
    ):
        """Log a complete query execution"""
        log_entry = {
            "query_id": metrics.query_id,
            "timestamp": metrics.timestamp,
            "query": query,
            "answer": answer,
            "retrieved_documents": [
                {
                    "content": doc.get("content", "")[:200],  # First 200 chars
                    "source": doc.get("source", ""),
                    "relevance_score": doc.get("relevance_score", 0.0)
                }
                for doc in retrieved_docs
            ],
            "metrics": metrics.to_dict()
        }

        self.queries.append(log_entry)

        # Write to daily log file
        log_file = self.log_dir / f"queries_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        logger.debug(f"Query logged: {metrics.query_id}")

    def get_recent_queries(self, n: int = 10) -> List[Dict]:
        """Get the n most recent queries"""
        return self.queries[-n:]

    def get_queries_by_date(self, date: str) -> List[Dict]:
        """Get all queries from a specific date (YYYYMMDD)"""
        log_file = self.log_dir / f"queries_{date}.jsonl"
        if not log_file.exists():
            return []

        queries = []
        with open(log_file, 'r') as f:
            for line in f:
                queries.append(json.loads(line))
        return queries


class MetricsAggregator:
    """Aggregate metrics for analysis"""

    def __init__(self):
        self.query_metrics: List[QueryMetrics] = []

    def add_query_metrics(self, metrics: QueryMetrics):
        """Add metrics from a query"""
        self.query_metrics.append(metrics)

    def compute_system_metrics(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> SystemMetrics:
        """Compute aggregate system metrics"""
        # Filter by time period if specified
        metrics_in_period = self.query_metrics
        if period_start or period_end:
            metrics_in_period = [
                m for m in self.query_metrics
                if (not period_start or datetime.fromisoformat(m.timestamp) >= period_start)
                and (not period_end or datetime.fromisoformat(m.timestamp) <= period_end)
            ]

        if not metrics_in_period:
            return SystemMetrics(
                period_start=period_start.isoformat() if period_start else "",
                period_end=period_end.isoformat() if period_end else ""
            )

        total = len(metrics_in_period)
        successful = sum(1 for m in metrics_in_period if m.guardrails_passed)

        return SystemMetrics(
            period_start=period_start.isoformat() if period_start else metrics_in_period[0].timestamp,
            period_end=period_end.isoformat() if period_end else metrics_in_period[-1].timestamp,
            total_queries=total,
            successful_queries=successful,
            failed_queries=total - successful,
            avg_total_time_ms=sum(m.total_time_ms for m in metrics_in_period) / total,
            avg_retrieval_time_ms=sum(m.retrieval_time_ms for m in metrics_in_period) / total,
            avg_generation_time_ms=sum(m.generation_time_ms for m in metrics_in_period) / total,
            avg_confidence_score=sum(m.confidence_score for m in metrics_in_period) / total,
            avg_rag_score=sum(m.rag_score for m in metrics_in_period if m.rag_score) / max(sum(1 for m in metrics_in_period if m.rag_score), 1),
            total_docs_retrieved=sum(m.num_docs_retrieved for m in metrics_in_period),
            avg_docs_per_query=sum(m.num_docs_retrieved for m in metrics_in_period) / total,
            guardrail_violations=sum(1 for m in metrics_in_period if not m.guardrails_passed)
        )

    def get_slowest_queries(self, n: int = 10) -> List[QueryMetrics]:
        """Get the n slowest queries"""
        return sorted(self.query_metrics, key=lambda m: m.total_time_ms, reverse=True)[:n]

    def get_low_confidence_queries(self, threshold: float = 0.5, n: int = 10) -> List[QueryMetrics]:
        """Get queries with confidence below threshold"""
        low_conf = [m for m in self.query_metrics if m.confidence_score < threshold]
        return sorted(low_conf, key=lambda m: m.confidence_score)[:n]


class ABComparisonTracker:
    """Track A/B comparisons between different configurations"""

    def __init__(self):
        self.comparisons = []

    def add_comparison(
        self,
        query: str,
        config_a: Dict,
        result_a: Dict,
        metrics_a: QueryMetrics,
        config_b: Dict,
        result_b: Dict,
        metrics_b: QueryMetrics
    ):
        """Log an A/B comparison"""
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "variant_a": {
                "config": config_a,
                "answer": result_a.get("answer", ""),
                "metrics": metrics_a.to_dict()
            },
            "variant_b": {
                "config": config_b,
                "answer": result_b.get("answer", ""),
                "metrics": metrics_b.to_dict()
            },
            "winner": self._determine_winner(metrics_a, metrics_b)
        }

        self.comparisons.append(comparison)

    def _determine_winner(self, metrics_a: QueryMetrics, metrics_b: QueryMetrics) -> str:
        """Determine which variant performed better"""
        # Simple scoring: higher confidence + faster
        score_a = metrics_a.confidence_score - (metrics_a.total_time_ms / 10000)
        score_b = metrics_b.confidence_score - (metrics_b.total_time_ms / 10000)

        if abs(score_a - score_b) < 0.05:
            return "tie"
        return "A" if score_a > score_b else "B"

    def get_summary(self) -> Dict:
        """Get summary of A/B tests"""
        if not self.comparisons:
            return {}

        winners = defaultdict(int)
        for comp in self.comparisons:
            winners[comp["winner"]] += 1

        return {
            "total_comparisons": len(self.comparisons),
            "variant_a_wins": winners.get("A", 0),
            "variant_b_wins": winners.get("B", 0),
            "ties": winners.get("tie", 0)
        }


class ObservabilityDashboard:
    """Central observability system"""

    def __init__(self, log_dir: str = "./logs"):
        self.tracker = PerformanceTracker()
        self.logger = QueryLogger(log_dir)
        self.aggregator = MetricsAggregator()
        self.ab_tracker = ABComparisonTracker()
        logger.info("✅ Observability dashboard initialized")

    def start_operation(self, operation: str):
        """Start tracking an operation"""
        self.tracker.start(operation)

    def end_operation(self, operation: str) -> float:
        """End tracking and return duration"""
        return self.tracker.end(operation)

    def log_query_execution(
        self,
        query: str,
        retrieved_docs: List,
        answer: str,
        metrics: QueryMetrics
    ):
        """Log complete query execution"""
        self.logger.log_query(query, retrieved_docs, answer, metrics)
        self.aggregator.add_query_metrics(metrics)

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        return self.aggregator.compute_system_metrics()

    def export_report(self, filepath: str = "./logs/report.html"):
        """Export metrics as HTML report"""
        system_metrics = self.get_system_metrics()
        slowest = self.aggregator.get_slowest_queries(5)
        low_conf = self.aggregator.get_low_confidence_queries(0.5, 5)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RAG System Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .metric {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>RAG System Performance Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <h2>System Overview</h2>
    <div class="metric">
        <div>Total Queries: <span class="metric-value">{system_metrics.total_queries}</span></div>
        <div>Success Rate: <span class="metric-value">{system_metrics.successful_queries / max(system_metrics.total_queries, 1) * 100:.1f}%</span></div>
        <div>Avg Response Time: <span class="metric-value">{system_metrics.avg_total_time_ms:.0f}ms</span></div>
        <div>Avg Confidence: <span class="metric-value">{system_metrics.avg_confidence_score:.2f}</span></div>
    </div>

    <h2>Performance Breakdown</h2>
    <div class="metric">
        <div>Retrieval: {system_metrics.avg_retrieval_time_ms:.0f}ms</div>
        <div>Generation: {system_metrics.avg_generation_time_ms:.0f}ms</div>
        <div>Docs per Query: {system_metrics.avg_docs_per_query:.1f}</div>
    </div>

    <h2>Slowest Queries</h2>
    <table>
        <tr>
            <th>Query</th>
            <th>Time (ms)</th>
            <th>Confidence</th>
        </tr>
        {''.join(f"<tr><td>{q.query[:60]}...</td><td>{q.total_time_ms:.0f}</td><td>{q.confidence_score:.2f}</td></tr>" for q in slowest)}
    </table>

    <h2>Low Confidence Queries</h2>
    <table>
        <tr>
            <th>Query</th>
            <th>Confidence</th>
            <th>Time (ms)</th>
        </tr>
        {''.join(f"<tr><td>{q.query[:60]}...</td><td>{q.confidence_score:.2f}</td><td>{q.total_time_ms:.0f}</td></tr>" for q in low_conf)}
    </table>
</body>
</html>
        """

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(html)

        logger.info(f"📊 Report exported to {filepath}")
        return filepath


__all__ = [
    "QueryMetrics",
    "SystemMetrics",
    "PerformanceTracker",
    "QueryLogger",
    "MetricsAggregator",
    "ABComparisonTracker",
    "ObservabilityDashboard"
]
