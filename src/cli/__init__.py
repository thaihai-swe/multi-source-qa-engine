"""Command-line interface for RAG system"""
from typing import Optional, List
from src.core import RAGSystem
from src.config import RAGConfig, get_config
from src.utils import get_logger
from datetime import datetime
from tabulate import tabulate

logger = get_logger()


class InteractiveRAG:
    """Interactive CLI interface for RAG system"""

    def __init__(self, rag: Optional[RAGSystem] = None):
        self.rag = rag or RAGSystem()
        self.running = True
        self.enable_streaming = False
        self.enable_fact_checking = False
        self.evaluation_results = []

    def run(self) -> None:
        """Start interactive loop"""
        print("\n" + "="*80)
        print("🚀 Advanced RAG System - Interactive Mode")
        print("="*80)


        print("\n📋 CORE COMMANDS:")
        print("  load <source> [collection]     - Load Wikipedia, URL, or file")
        print("  query <question>               - Standard RAG query")
        print("  sources                        - Show loaded sources")
        print("  history                        - Show conversation history")
        print("  metrics                        - Show RAGAS metrics")

        print("\n🚀 ADVANCED COMMANDS:")
        print("  expand <query>                 - Query expansion (4 variations)")
        print("  multihop <query>               - Multi-hop reasoning")
        print("  agent <query>                  - Use agentic RAG (autonomous strategy)")
        print("  async <query1> | <query2> ...  - Batch async queries")

        print("\n⚡ SETTINGS & TOOLS:")
        print("  streaming                      - Toggle streaming responses")
        print("  fact-check                     - Toggle fact-checking")
        print("  self-query                     - Toggle self-query decomposition")
        print("  domain                         - Toggle domain guard (out-of-domain detection)")
        print("  hallucination                  - Toggle hallucination detection & mitigation")
        print("  rerank                         - Toggle cross-encoder reranking & MMR diversity")
        print("  highlight                      - Toggle passage highlighting")
        print("  guardrail                      - Toggle input/output guardrails & safety")
        print("  cache                          - Show cache statistics")
        print("  facts                          - Show fact-check results")
        print("  passages                       - Show highlighted passages from last query")
        print("  hallucination-report           - Show last hallucination report")
        print("  observability                  - Show observability metrics & export report")
        print("  experiments                    - Run optimization experiments")
        print("  domain-stats                   - Show domain guard profile")

        print("\n📚 OTHER:")
        print("  save [filename]                - Save conversation")
        print("  clear                          - Clear history")
        print("  help                           - Show this help")
        print("  quit/exit                      - Exit")
        print("="*80 + "\n")

        while self.running:
            try:
                user_input = input("❓ > ").strip()

                if not user_input:
                    continue

                if self._handle_command(user_input):
                    continue

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                self.running = False
            except Exception as e:
                logger.error(f"Error: {e}")

    def _handle_command(self, user_input: str) -> bool:
        """Handle user commands. Returns True if command handled."""
        cmd = user_input.lower().split()[0] if user_input else ""

        # Exit commands
        if cmd in ('quit', 'exit'):
            print("\n👋 Goodbye!")
            self.running = False
            return True

        # Help
        if cmd == 'help':
            self._show_help()
            return True

        # Core commands
        if cmd == 'cache':
            self._show_cache_stats()
            return True

        if cmd == 'history':
            self._show_history()
            return True

        if cmd == 'sources':
            self._show_sources()
            return True

        if cmd == 'clear':
            self.rag.conversation_history = []
            print("✅ Conversation history cleared")
            return True

        if cmd == 'load':
            self._handle_load(user_input)
            return True

        if cmd == 'save':
            self._handle_save(user_input)
            return True

        if cmd == 'metrics':
            self._show_metrics()
            return True

        # Advanced commands
        if cmd == 'expand':
            self._handle_expand(user_input)
            return True

        if cmd == 'multihop':
            self._handle_multihop(user_input)
            return True

        if cmd == 'agent':
            self._handle_agent(user_input)
            return True

        if cmd == 'async':
            self._handle_async(user_input)
            return True

        if cmd == 'observability':
            self._handle_observability()
            return True

        if cmd == 'experiments':
            self._handle_experiments()
            return True

        # Settings
        if cmd == 'streaming':
            self.enable_streaming = not self.enable_streaming
            status = "✅ ENABLED" if self.enable_streaming else "❌ DISABLED"
            print(f"💬 Streaming: {status}")
            return True

        if cmd == 'fact-check':
            self.enable_fact_checking = not self.enable_fact_checking
            status = "✅ ENABLED" if self.enable_fact_checking else "❌ DISABLED"
            print(f"🔍 Fact-checking: {status}")
            return True

        if cmd == 'facts':
            self._show_fact_results()
            return True

        # --- New feature toggles ---
        if cmd == 'self-query':
            self.rag.config.reasoning.enable_self_query = not self.rag.config.reasoning.enable_self_query
            status = "✅ ENABLED" if self.rag.config.reasoning.enable_self_query else "❌ DISABLED"
            print(f"🧩 Self-query decomposition: {status}")
            return True

        if cmd == 'domain':
            self.rag.config.search.enable_domain_guard = not self.rag.config.search.enable_domain_guard
            status = "✅ ENABLED" if self.rag.config.search.enable_domain_guard else "❌ DISABLED"
            print(f"🛡️  Domain guard: {status}")
            return True

        if cmd == 'hallucination':
            self.rag.config.evaluation.enable_hallucination_detection = (
                not self.rag.config.evaluation.enable_hallucination_detection
            )
            status = "✅ ENABLED" if self.rag.config.evaluation.enable_hallucination_detection else "❌ DISABLED"
            print(f"🧪 Hallucination detection: {status}")
            return True

        if cmd == 'rerank':
            self.rag.config.search.enable_reranking = not self.rag.config.search.enable_reranking
            status = "✅ ENABLED" if self.rag.config.search.enable_reranking else "❌ DISABLED"
            print(f"🔄 Reranking (cross-encoder + MMR): {status}")
            if self.rag.config.search.enable_reranking:
                print(f"   Cross-encoder: {self.rag.config.search.use_cross_encoder}")
                print(f"   MMR diversity: {self.rag.config.search.use_mmr_diversity} (λ={self.rag.config.search.mmr_lambda})")
            return True

        if cmd == 'highlight':
            self.rag.config.search.enable_passage_highlighting = (
                not self.rag.config.search.enable_passage_highlighting
            )
            status = "✅ ENABLED" if self.rag.config.search.enable_passage_highlighting else "❌ DISABLED"
            print(f"📍 Passage highlighting: {status}")
            return True

        if cmd == 'guardrail':
            self.rag.config.evaluation.enable_guardrails = (
                not self.rag.config.evaluation.enable_guardrails
            )
            status = "✅ ENABLED" if self.rag.config.evaluation.enable_guardrails else "❌ DISABLED"
            print(f"🛡️  Guardrails & Safety: {status}")
            if self.rag.config.evaluation.enable_guardrails:
                print("   - Prompt injection detection")
                print("   - PII detection & redaction")
                print("   - Toxicity filtering")
                print("   - Rate limiting")
            return True

        if cmd == 'passages':
            self._show_highlighted_passages()
            return True

        if cmd == 'hallucination-report':
            self._show_hallucination_report()
            return True

        if cmd == 'domain-stats':
            self._show_domain_stats()
            return True

        if cmd == 'query':
            self._handle_query(user_input)
            return True

        # Default: treat as query
        self._handle_query(f"query {user_input}")
        return True

    def _handle_load(self, command: str) -> None:
        """Handle load command"""
        import shlex

        # Parse command with proper quote handling
        try:
            parts = shlex.split(command)
        except ValueError:
            # Fallback to simple split if shlex fails
            parts = command.split(maxsplit=2)

        if len(parts) < 2:
            print("❌ Usage: load <source> [collection_name]")
            print("   Examples:")
            print("     load wikipedia \"Machine Learning\"")
            print("     load https://example.com")
            print("     load myfile.txt")
            print("     load document.pdf")
            return

        # Handle 'load wikipedia "Topic"' format
        if parts[1].lower() == 'wikipedia' and len(parts) > 2:
            source = f"wikipedia {parts[2]}"
            collection_name = parts[3] if len(parts) > 3 else None
        else:
            source = parts[1]
            collection_name = parts[2] if len(parts) > 2 else None

        try:
            self.rag.load_data(source, collection_name or self.rag._get_collection_name(source))
        except Exception as e:
            print(f"❌ Error loading: {e}")
            logger.error(f"Load error: {e}")

    def _handle_query(self, command: str) -> None:
        """Handle query command"""
        query_text = command.replace("query ", "", 1).strip()
        if not query_text:
            print("❌ Please provide a query")
            return

        try:
            if self.enable_streaming:
                # Streaming mode: retrieve documents first, then stream answer
                docs = self.rag._retrieve_documents(query_text)
                if not docs:
                    print("\n❌ No relevant documents found.\n")
                    return

                print("\n💬 Answer (streaming):")
                print("-" * 80)
                answer_chunks = []
                for chunk in self.rag.generator.generate_streaming(query_text, docs):
                    print(chunk, end="", flush=True)
                    answer_chunks.append(chunk)
                print("\n" + "-" * 80)

                # Create a simplified response for display
                from src.models import RAGResponse
                response = RAGResponse(
                    answer="".join(answer_chunks),
                    sources=docs,
                    confidence_score=0.5,
                    source_types=list(set(doc.source_type for doc in docs)),
                    conversation_context=None,
                )
                self._display_response(response)
            else:
                # Buffered mode: normal processing
                response = self.rag.process_query(query_text)
                self._display_response(response)

            # Store evaluation result
            if response.confidence_score:
                self.evaluation_results.append({
                    "query": query_text,
                    "confidence": response.confidence_score,
                    "sources": len(response.sources),
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Query error: {e}")

    def _handle_expand(self, command: str) -> None:
        """Handle query expansion"""
        query_text = command.replace("expand ", "", 1).strip()
        if not query_text:
            print("❌ Usage: expand <query>")
            return

        try:
            print("\n🔄 Generating query variations...")
            variations = self.rag.query_expander.expand(query_text, num_variations=4)

            print(f"\n📝 Original: {query_text}")
            print(f"\n🔄 Variations ({len(variations)}):")
            for i, var in enumerate(variations, 1):
                print(f"   {i}. {var}")

            # Auto-query with expansion enabled
            print("\n💡 Querying with expanded variations...")
            response = self.rag.process_query(query_text, use_expansion=True)
            self._display_response(response)

        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Expansion error: {e}")

    def _handle_multihop(self, command: str) -> None:
        """Handle multi-hop reasoning"""
        query_text = command.replace("multihop ", "", 1).strip()
        if not query_text:
            print("❌ Usage: multihop <query>")
            return

        try:
            print("\n🎯 Decomposing query into steps...")
            steps = self.rag.multi_hop_reasoner.decompose(query_text, max_steps=3)

            print(f"\n📝 Original Query: {query_text}")
            print(f"\n🎯 Decomposed into {len(steps)} steps:")
            for i, step in enumerate(steps, 1):
                print(f"   {i}. {step}")

            # Retrieve for each step
            print("\n📚 Retrieving context for each step...")
            step_results = []
            for i, step in enumerate(steps, 1):
                docs = self.rag._retrieve_relevant_chunks(step, n_results=2)
                context = "\n".join([d.content[:100] for d in docs])
                step_results.append({
                    "subquery": step,
                    "answer": context[:200]
                })
                print(f"   ✅ Step {i}: Retrieved {len(docs)} docs")

            # Synthesize
            print("\n🔗 Synthesizing final answer...")
            final_answer = self.rag.multi_hop_reasoner.synthesize(query_text, step_results)

            print(f"\n💡 SYNTHESIZED ANSWER:")
            print("="*80)
            print(final_answer)
            print("="*80)

        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Multi-hop error: {e}")

    def _handle_agent(self, command: str) -> None:
        """Handle agentic RAG query"""
        query_text = command.replace("agent ", "", 1).strip()
        if not query_text:
            print("❌ Usage: agent <query>")
            print("   Example: agent Compare machine learning and deep learning")
            return

        try:
            from src.reasoning import AgenticRAG

            print("\n🤖 Initializing agentic RAG...")
            agent = AgenticRAG(self.rag, max_steps=5)

            print(f"💭 Processing: {query_text}")
            response = agent.process_query(query_text)

            print(f"\n🎯 Strategy Used: {response.strategy_used}")
            print(f"🔢 Steps Taken: {response.total_steps}")
            print(f"🎲 Confidence: {response.confidence:.2f}")

            print(f"\n💡 REASONING TRACE:")
            print("="*80)
            for step in response.reasoning_steps:
                print(f"\nStep {step.step_number}:")
                print(f"  💭 Thought: {step.thought}")
                print(f"  🎬 Action: {step.action.value}")
                print(f"  📝 Input: {step.action_input}")
                print(f"  👁️  Observation: {step.observation[:100]}...")

            print(f"\n💡 FINAL ANSWER:")
            print("="*80)
            print(response.final_answer)
            print("="*80)
            print(f"📖 Sources: {len(response.sources)} documents used\n")

        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Agent error: {e}")

    def _handle_async(self, command: str) -> None:
        """Handle async batch queries"""
        queries_text = command.replace("async ", "", 1).strip()
        if not queries_text:
            print("❌ Usage: async <query1> | <query2> | <query3>")
            print("   Example: async What is AI? | What is ML? | What is DL?")
            return

        try:
            import asyncio
            from src.core.async_rag import batch_queries_async

            # Split by pipe
            queries = [q.strip() for q in queries_text.split('|') if q.strip()]

            if len(queries) < 2:
                print("⚠️  Async is best for multiple queries. Use '|' to separate them.")
                print("   Processing single query normally...")
                self._handle_query(f"query {queries[0]}")
                return

            print(f"\n⚡ Processing {len(queries)} queries in parallel...")
            print("-"*80)

            # Run async queries
            responses = asyncio.run(batch_queries_async(self.rag, queries, max_concurrent=3))

            print(f"\n✅ Completed {len(responses)} queries!")
            print("="*80)

            # Display results
            for i, (query, response) in enumerate(zip(queries, responses), 1):
                print(f"\n[{i}] {query}")
                print(f"    Answer: {response.answer[:100]}...")
                print(f"    Sources: {len(response.sources)} | Time: {response.execution_time_ms:.0f}ms")

            print("="*80 + "\n")

        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Async error: {e}")

    def _handle_observability(self) -> None:
        """Handle observability dashboard"""
        try:
            from src.evaluation import ObservabilityDashboard

            print("\n📊 Observability Dashboard")
            print("="*80)

            # Check if we have a dashboard instance
            if not hasattr(self.rag, 'observability_dashboard'):
                print("⚠️  Observability not yet initialized.")
                print("💡 Processing some queries first will populate metrics.")
                return

            dashboard = self.rag.observability_dashboard
            metrics = dashboard.get_system_metrics()

            if metrics.total_queries == 0:
                print("⚠️  No queries processed yet.")
                return

            print(f"\n📈 System Metrics:")
            print(f"   Total Queries: {metrics.total_queries}")
            print(f"   Avg Response Time: {metrics.avg_total_time_ms:.0f}ms")
            print(f"   Avg Retrieval Time: {metrics.avg_retrieval_time_ms:.0f}ms")
            print(f"   Avg Generation Time: {metrics.avg_generation_time_ms:.0f}ms")
            print(f"   Avg Confidence: {metrics.avg_confidence_score:.2%}")
            print(f"   Avg Docs per Query: {metrics.avg_docs_per_query:.1f}")

            # Export report
            print(f"\n📄 Exporting HTML report...")
            report_path = dashboard.export_report("./logs/dashboard_report.html")
            print(f"✅ Report saved: {report_path}")
            print(f"💡 Open in browser: file://{report_path}")
            print("="*80 + "\n")

        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Observability error: {e}")

    def _handle_experiments(self) -> None:
        """Handle optimization experiments"""
        try:
            from src.evaluation import ExperimentRunner

            print("\n🧪 Optimization Experiments")
            print("="*80)
            print("This will test different configurations to find optimal settings.")
            print("⚠️  Warning: This may take several minutes!\n")

            confirm = input("Continue? (yes/no): ").strip().lower()
            if confirm not in ['yes', 'y']:
                print("Cancelled.")
                return

            # Get test queries
            print("\nEnter test queries (one per line, empty line to finish):")
            test_queries = []
            while True:
                query = input(f"  Query {len(test_queries)+1}: ").strip()
                if not query:
                    break
                test_queries.append(query)

            if len(test_queries) < 3:
                print("⚠️  Need at least 3 queries for meaningful results.")
                return

            print(f"\n⚡ Running experiments with {len(test_queries)} test queries...")
            runner = ExperimentRunner(self.rag, output_dir="./experiments")

            # Run chunk size experiment
            print("\n1️⃣  Testing chunk sizes [256, 512, 1024]...")
            chunk_results = runner.chunk_size_exp.run_experiment(
                test_queries=test_queries,
                chunk_sizes=[256, 512, 1024],
                overlap_ratio=0.1
            )

            print("\n   Results:")
            for result in chunk_results:
                print(f"     Chunk {result.config['chunk_size']}: "
                      f"RAG Score {result.avg_rag_score:.3f}, "
                      f"Time {result.avg_total_time_ms:.0f}ms")

            # Run top-k experiment
            print("\n2️⃣  Testing top-k values [3, 5, 10]...")
            topk_results = runner.top_k_exp.run_experiment(
                test_queries=test_queries,
                k_values=[3, 5, 10]
            )

            print("\n   Results:")
            for result in topk_results:
                print(f"     Top-K {result.config['top_k']}: "
                      f"RAG Score {result.avg_rag_score:.3f}, "
                      f"Time {result.avg_total_time_ms:.0f}ms")

            # Generate report
            print("\n📄 Generating HTML report...")
            report_path = runner.generate_report(
                chunk_results + topk_results,
                "optimization_experiments.html"
            )
            print(f"✅ Report saved: {report_path}")
            print(f"💡 Open in browser: file://{report_path}")
            print("="*80 + "\n")

        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Experiments error: {e}")

    def _handle_save(self, command: str) -> None:
        """Handle save command"""
        parts = command.split(maxsplit=1)
        filename = parts[1] if len(parts) > 1 else "conversation"

        try:
            self.rag.save_conversation(filename)
            print(f"✅ Saved to {filename}.json")
        except Exception as e:
            print(f"❌ Error saving: {e}")

    def _show_help(self) -> None:
        """Show help information"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║           ADVANCED RAG SYSTEM - COMMAND REFERENCE             ║
╚══════════════════════════════════════════════════════════════╝

📋 CORE COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  load <source> [collection]        Load data (Wikipedia/URL/File/PDF)
  query <question>                  Standard RAG query
  sources                           Show all loaded sources
  history                           Show conversation history
  metrics                           Show RAGAS evaluation metrics
  clear                             Clear conversation history
  save [filename]                   Save conversation to JSON

🚀 ADVANCED FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  expand <query>                    Query expansion (4 variations)
  multihop <query>                  Multi-hop reasoning (3 steps)
  agent <query>                     Agentic RAG with autonomous strategy
  async <q1> | <q2> | <q3>          Batch queries in parallel

⚡ SETTINGS & TOGGLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  streaming                         Toggle streaming responses
  fact-check                        Toggle fact-checking
  guardrail                         Toggle guardrails (input/output safety)
  self-query                        Toggle self-query decomposition
  domain                            Toggle domain guard
  hallucination                     Toggle hallucination detection & mitigation
  cache                             Show cache statistics
  facts                             Show fact-check results
  hallucination-report              Show last hallucination analysis report
  domain-stats                      Show domain guard profile & topics
  observability                     Show performance metrics & export report
  experiments                       Run optimization experiments

📚 GENERAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  help                              Show this help
  quit / exit                       Exit the program

💡 EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  > load https://en.wikipedia.org/wiki/Machine_learning
  > guardrail                               # enable safety checks
  > query "What is machine learning?"
  > agent "Compare supervised and unsupervised learning"
  > async What is AI? | What is ML? | What is DL?
  > observability                           # view performance metrics
  > experiments                             # optimize settings
        """)

    def _show_cache_stats(self) -> None:
        """Show cache statistics"""
        try:
            stats = self.rag.embedding_cache.get_stats()
            print(f"\n💾 EMBEDDING CACHE STATISTICS")
            print("="*80)
            print(f"  Cache Size:        {stats['size']}/{stats['max_size']} embeddings")
            print(f"  Total Lookups:     {stats['total_lookups']}")
            print(f"  Cache Hits:        {stats['hits']}")
            print(f"  Cache Misses:      {stats['misses']}")
            print(f"  Hit Rate:          {stats['hit_rate']}")
            print("="*80 + "\n")
        except Exception as e:
            print(f"❌ Error: {e}")

    def _show_history(self) -> None:
        """Show conversation history"""
        if not self.rag.conversation_history:
            print("\n📝 No conversation history yet.\n")
            return

        print(f"\n📜 CONVERSATION HISTORY ({len(self.rag.conversation_history)} messages)")
        print("="*80)

        for i, msg in enumerate(self.rag.conversation_history, 1):
            role = "👤 USER" if msg.get("role") == "user" else "🤖 ASSISTANT"
            content = msg.get("content", msg.get("answer", ""))[:100]
            print(f"\n[{i}] {role}")
            print(f"    {content}...")
            if msg.get("sources"):
                print(f"    Sources: {len(msg['sources'])} docs")

        print("\n" + "="*80 + "\n")

    def _show_sources(self) -> None:
        """Show loaded sources"""
        if not self.rag.loaded_sources:
            print("\n❌ No sources loaded yet.\n")
            return

        print(f"\n📂 LOADED SOURCES ({len(self.rag.loaded_sources)})")
        print("="*80)

        for source, source_type in self.rag.loaded_sources.items():
            emoji = "🌐" if source_type == "url" else "📚" if source_type == "wikipedia" else "📄"
            print(f"{emoji} [{source_type.upper()}] {source}")

        print("="*80 + "\n")

    def _show_metrics(self) -> None:
        """Show RAGAS evaluation metrics"""
        if not self.evaluation_results:
            print("\n📊 No evaluations available yet.\n")
            return

        print(f"\n📊 RAGAS EVALUATION METRICS ({len(self.evaluation_results)} evaluations)")
        print("="*80)

        avg_confidence = sum(r["confidence"] for r in self.evaluation_results) / len(self.evaluation_results)
        total_docs = sum(r["sources"] for r in self.evaluation_results)

        print(f"\n📈 Summary:")
        print(f"  Total Queries: {len(self.evaluation_results)}")
        print(f"  Avg Confidence: {avg_confidence:.1%}")
        print(f"  Total Docs Retrieved: {total_docs}")

        print(f"\n📋 Recent Results:")
        table_data = []
        for result in self.evaluation_results[-5:]:
            table_data.append([
                result["query"][:30] + "..." if len(result["query"]) > 30 else result["query"],
                f"{result['confidence']:.1%}",
                result["sources"]
            ])

        print(tabulate(table_data, headers=["Query", "Confidence", "Docs"], tablefmt="grid"))
        print("="*80 + "\n")

    def _show_fact_results(self) -> None:
        """Show fact-checking results"""
        if not hasattr(self.rag, 'last_fact_check_results') or not self.rag.last_fact_check_results:
            print("\n🔍 No fact-check results available. Run a query first.\n")
            return

        print(f"\n🔍 FACT-CHECK RESULTS")
        print("="*80)

        results = self.rag.last_fact_check_results
        supported = sum(1 for r in results if r.get("is_supported"))

        print(f"\n📊 Summary:")
        print(f"  Total Facts: {len(results)}")
        print(f"  Supported: {supported}/{len(results)} ({supported/len(results)*100:.0f}%)")

        print(f"\n📋 Results:")
        table_data = []
        for result in results:
            status = "✅" if result.get("is_supported") else "⚠️"
            fact = result.get("fact", "")[:40]
            conf = f"{result.get('confidence', 0)*100:.0f}%"
            table_data.append([status, fact, conf])

        print(tabulate(table_data, headers=["Status", "Fact", "Confidence"], tablefmt="grid"))
        print("="*80 + "\n")

    def _show_hallucination_report(self) -> None:
        """Show the last hallucination detection report."""
        report = getattr(self.rag, 'last_hallucination_report', None)
        if not report:
            print("\n🧪 No hallucination report available. Enable hallucination detection and run a query first.\n")
            return

        risk_icons = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
        icon = risk_icons.get(report.overall_risk_level, "⚪")

        print(f"\n🧪 HALLUCINATION DETECTION REPORT")
        print("="*80)
        print(f"  Risk Level:       {icon} {report.overall_risk_level}")
        print(f"  Grounding Score:  {report.grounding_score:.1%} (higher = better grounded)")
        print(f"  Hallucination:    {report.hallucination_score:.1%}")
        print(f"  Total Claims:     {len(report.supported_claims) + len(report.unsupported_claims)}")
        print(f"  Supported:        {len(report.supported_claims)}")
        print(f"  Unsupported:      {len(report.unsupported_claims)}")
        print(f"  Mitigated:        {'✅ Yes' if report.mitigation_applied else '❌ No'}")

        if report.unsupported_claims:
            print(f"\n⚠️  Unsupported Claims:")
            for i, claim in enumerate(report.unsupported_claims, 1):
                print(f"   {i}. {claim[:100]}{'...' if len(claim) > 100 else ''}")

        if report.mitigation_applied and report.mitigation_explanation:
            print(f"\n🔧 Mitigation Applied: {report.mitigation_explanation}")

        if report.supported_claims:
            print(f"\n✅ Well-Grounded Claims (top 3):")
            for cg in report.supported_claims[:3]:
                print(f"   • [{cg.grounding_score:.0%}] {cg.claim[:80]}{'...' if len(cg.claim) > 80 else ''}")
                if cg.best_source:
                    print(f"          Source: {cg.best_source[:60]}")
        print("="*80 + "\n")

    def _show_domain_stats(self) -> None:
        """Show domain guard profile statistics."""
        stats = self.rag.domain_guard.get_domain_stats()
        enabled = self.rag.config.search.enable_domain_guard
        status = "✅ ENABLED" if enabled else "❌ DISABLED"

        print(f"\n🛡️  DOMAIN GUARD STATISTICS")
        print("="*80)
        print(f"  Status:           {status}")
        print(f"  Profile Built:    {'✅ Yes' if stats['has_profile'] else '❌ No (load data first)'}")
        print(f"  Sample Chunks:    {stats['sample_chunks_count']}")
        print(f"  Similarity Threshold: {stats['similarity_threshold']:.0%}")
        if stats['domain_topics']:
            print(f"  Domain Topics:    {', '.join(stats['domain_topics'])}")
        else:
            print("  Domain Topics:    (none — load data to build profile)")
        print("="*80 + "\n")

    def _show_highlighted_passages(self) -> None:
        """Show highlighted passages from the last query."""
        passages = getattr(self.rag, 'last_highlighted_passages', None)
        if not passages:
            print("\n📍 No highlighted passages available.")
            print("   Enable passage highlighting with 'highlight' and run a query first.\n")
            return

        from src.retrieval.passage_highlighter import PassageHighlighter
        highlighter = PassageHighlighter()
        print(highlighter.format_highlighted_passages(passages, max_display=5))
        print()

    def _display_response(self, response) -> None:
        """Display RAG response with formatting"""
        # Guardrails warning (if any)
        if hasattr(self.rag, 'last_guardrail_results') and self.rag.last_guardrail_results:
            high_risk = [r for r in self.rag.last_guardrail_results if r.risk_level == "HIGH"]
            medium_risk = [r for r in self.rag.last_guardrail_results if r.risk_level == "MEDIUM"]
            if high_risk or medium_risk:
                print(f"\n🛡️  GUARDRAIL WARNING")
                print("-"*80)
                for result in high_risk + medium_risk:
                    print(f"  [{result.risk_level}] {result.message}")
                print("-"*80)

        # Domain warning — shown first, before the answer
        if response.domain_check and not response.domain_check.is_in_domain:
            print(f"\n⚠️  DOMAIN WARNING")
            print("-"*80)
            print(f"  {response.domain_check.warning_message}")
            if response.domain_check.recommendation:
                print(f"  💡 {response.domain_check.recommendation}")
            print(f"  Relevance score: {response.domain_check.similarity_score:.0%}")
            print("-"*80)

        # Self-query decomposition notice
        if response.self_query_decomposition and response.self_query_decomposition.is_complex:
            sq = response.self_query_decomposition
            print(f"\n🧩 Self-Query Decomposition: split into {len(sq.sub_queries)} sub-queries")
            for i, sub in enumerate(sq.sub_queries, 1):
                print(f"   {i}. [{sub.aspect}] {sub.query[:70]}{'...' if len(sub.query) > 70 else ''}")

        print(f"\n💡 ANSWER:")
        print("="*80)
        print(response.answer)
        print("="*80)
        print(f"📊 Confidence: {response.confidence_score:.1%}")
        if response.source_types:
            print(f"📚 Sources: {', '.join(response.source_types)}")
        if response.sources:
            print(f"📖 Retrieved {len(response.sources)} documents:")
            for i, doc in enumerate(response.sources[:3], 1):
                print(f"   {i}. [{doc.source_type}] {doc.source[:40]}...")

        # Hallucination summary inline
        if response.hallucination_report:
            hr = response.hallucination_report
            risk_icons = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
            icon = risk_icons.get(hr.overall_risk_level, "⚪")
            mit = " → mitigated ✅" if hr.mitigation_applied else ""
            print(f"🧪 Hallucination Risk: {icon} {hr.overall_risk_level} "
                  f"(grounding {hr.grounding_score:.0%}){mit}")
            print("   Run 'hallucination-report' for full details.")

        # Reranking indicator
        if hasattr(response, 'reranker_applied') and response.reranker_applied:
            print(f"🔄 Reranking: Applied (cross-encoder + MMR diversity)")

        # Highlighted passages inline preview
        if response.highlighted_passages and len(response.highlighted_passages) > 0:
            print(f"📍 Highlighted Passages: {len(response.highlighted_passages)} relevant excerpts found")
            print("   Run 'passages' to see detailed highlights.")

        if hasattr(response, 'execution_time_ms'):
            print(f"⏱️  Time: {response.execution_time_ms:.1f}ms")
        print()


def main() -> None:
    """Main entry point"""
    rag = InteractiveRAG()
    rag.run()


if __name__ == "__main__":
    main()
