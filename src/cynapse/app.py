"""Terminal entry point for seeding the database and chatting with the query agent.

Examples:
    cynapse seed
    cynapse chat --ai
    cynapse chat --nonai
    cynapse seed --db sample.db
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from cynapse.helpers.database.database_manager import SG, seed
from cynapse.orchestration import build_graph
from cynapse.helpers.openai_client import configured_extractor


def main():
    """Read terminal arguments and run the selected command.

    Args:
        None directly. Reads these arguments from the command line:
            command: "seed" creates/updates simulated records; "chat" starts
                an interactive conversation after refreshing existing data.
            --db: SQLite file path; defaults to frames.db.
            --nonai: Use the nonai version without an API key.
            --ai: Use OpenAI with CYNAPSE_MODEL from the environment.
                Chat requires exactly one of --ai or --nonai.

    Returns:
        None. Prints seed confirmation or JSON responses to the terminal.
        Seeding writes to the database; chat also refreshes it at startup.

    Chat behavior:
        Requires an existing database; run "cynapse seed" first.
        Keeps one conversation under thread_id "cli" for this process.
        Passes each request and the current Singapore time to the graph.
        Supports next/previous pages and ends on "exit", EOF, or Ctrl+C
        while reading input. Closes the model client when the chat loop ends.

    Raises:
        SystemExit: argparse exits for --help, invalid arguments, a missing
            chat database, or invalid model configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["seed", "chat"])
    parser.add_argument("--db", type=Path, default=Path("frames.db"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--nonai", action="store_true", help="Use the nonai version; no API key needed")
    mode.add_argument("--ai", action="store_true", help="Use OpenAI configured by CYNAPSE_MODEL")
    args = parser.parse_args()
    if args.command == "seed":
        if args.ai or args.nonai:
            parser.error("--ai and --nonai are only valid for chat")
        seed(args.db, datetime.now(SG))
        print(f"Seeded {args.db}")
        return
    if not (args.ai or args.nonai):
        parser.error("Choose cynapse chat --ai or cynapse chat --nonai")
    if not args.db.exists():
        parser.error("Run cynapse seed first.")
    print("Updating simulated frame records through now...", flush=True)
    seed(args.db, datetime.now(SG))
    extractor = None
    try:
        extractor = None if args.nonai else configured_extractor()
        graph = build_graph(args.db, extractor=extractor)
    except ValueError as exc:
        parser.error(str(exc))
    config = {"configurable": {"thread_id": "cli"}}
    print("Try: Show me frames from CTE today. Use next/previous for pages; exit to quit.")
    try:
        while True:
            try:
                request = input("> ")
            except (EOFError, KeyboardInterrupt):
                break
            if request.lower().strip() == "exit":
                break
            result = graph.invoke({"request": request, "now": datetime.now(SG)}, config)
            print(json.dumps(result["response"], indent=2))
    finally:
        if extractor is not None:
            extractor.close()


if __name__ == "__main__":
    main()
