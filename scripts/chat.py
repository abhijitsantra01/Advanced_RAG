"""Interactive Multi-Mode Terminal REPL for RAG and Agentic RAG.

Usage:
    python scripts/chat.py
    python scripts/chat.py --mode agent --use-case technical
"""
import argparse
import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct CLI execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

from rag.agent import AgenticRAG
from rag.config import RAGSettings
from rag.observability import setup_observability
from rag.pipeline import RAGPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive terminal chat for Enterprise RAG / Agentic RAG."
    )
    parser.add_argument(
        "--mode",
        choices=["rag", "agent"],
        default="rag",
        help="Execution mode: 'rag' (linear retrieval) or 'agent' (autonomous multi-step).",
    )
    parser.add_argument(
        "--use-case",
        type=str,
        default="general",
        help="Initial use-case domain namespace (default: general).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.40,
        help="Cosine similarity score threshold for RAG retrieval (default: 0.40).",
    )
    return parser.parse_args()


def print_banner(mode: str, use_case: str) -> None:
    print("=" * 70)
    print(" Enterprise Agentic RAG — Interactive Shell")
    print(f" Mode: [{mode.upper()}]  |  Use-Case: [{use_case}]")
    print(" Commands: /mode <rag|agent> | /usecase <name> | /exit")
    print("=" * 70 + "\n")


def main() -> None:
    args = parse_args()
    load_dotenv()
    settings = RAGSettings()
    setup_observability(settings)

    pipeline = RAGPipeline(settings=settings)
    agent = AgenticRAG(settings=settings)

    current_mode = args.mode.lower()
    current_use_case = args.use_case

    print_banner(current_mode, current_use_case)

    while True:
        try:
            prompt_str = f"[{current_mode.upper()}:{current_use_case}] > "
            user_input = input(prompt_str).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat session.")
            break

        if not user_input:
            continue

        # Command handling
        if user_input.lower() in {"/exit", "exit", "quit"}:
            print("Session ended.")
            break

        if user_input.startswith("/mode"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].lower() in {"rag", "agent"}:
                current_mode = parts[1].lower()
                print(f"Switched mode to [{current_mode.upper()}].\n")
            else:
                print("Usage: /mode <rag|agent>\n")
            continue

        if user_input.startswith("/usecase"):
            parts = user_input.split()
            if len(parts) == 2:
                current_use_case = parts[1].strip()
                print(f"Switched domain namespace to [{current_use_case}].\n")
            else:
                print("Usage: /usecase <name>\n")
            continue

        # Execution
        print("\nProcessing query...")
        if current_mode == "rag":
            resp = pipeline.query(
                question=user_input,
                use_case=current_use_case,
                score_threshold=args.score_threshold,
            )
            print("-" * 50)
            print(resp.answer)
            print("-" * 50)
            print(
                f"Model: {resp.generation_model_used} "
                f"(Fallback: {resp.fallback_triggered}) | "
                f"Embedding: {resp.embedding_model_used} | "
                f"Retrieved: {len(resp.retrieved_chunks)} chunks"
            )
            if resp.retrieved_chunks:
                citations = [
                    f"[{c.chunk.doc_id}:{c.chunk.chunk_index}] (score: {c.score:.3f})"
                    for c in resp.retrieved_chunks
                ]
                print(f"Context: {', '.join(citations)}\n")
            else:
                print("Context: No matching chunks met the score threshold.\n")

        elif current_mode == "agent":
            agent_res = agent.run(prompt=user_input)
            print("-" * 50)
            print(agent_res.answer)
            print("-" * 50)
            print(
                f"Model: {agent_res.model_used} "
                f"(Fallback: {agent_res.fallback_triggered}) | "
                f"Steps: {agent_res.steps_taken} | "
                f"Tools called: {agent_res.tools_called or 'None'}\n"
            )


if __name__ == "__main__":
    main()
