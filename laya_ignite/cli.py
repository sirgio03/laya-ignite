"""
Command-line interface for Laya-Ignite.
Usage:
    laya-ignite ignite --questions task.json --out ./adapted_checkpoint
    laya-ignite ui
"""

import argparse
import json
import sys
from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(description="🔥 Laya-Ignite: Zero-Shot to System-1 Bootstrap Engine")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Command: ignite
    ignite_parser = subparsers.add_parser("ignite", help="Synthesize and adapt Laya on a question bundle")
    ignite_parser.add_argument("--questions", "-q", required=True, help="Path to JSON file containing questions definition")
    ignite_parser.add_argument("--generator", "-g", default="mock", help="Generator backend (mock, ollama, openai, huggingface)")
    ignite_parser.add_argument("--samples", "-s", type=int, default=20, help="Samples per class to synthesize")
    ignite_parser.add_argument("--out", "-o", default="./laya_ignited", help="Output path for adapted checkpoint")

    # Command: ui
    ui_parser = subparsers.add_parser("ui", help="Launch interactive Gradio Web UI")
    ui_parser.add_argument("--port", "-p", type=int, default=7860, help="Port to serve Gradio on")

    args = parser.parse_args()

    if args.command == "ignite":
        from . import bootstrap
        try:
            with open(args.questions, "r", encoding="utf-8") as f:
                questions_data = json.load(f)
        except Exception as e:
            console.print(f"[bold red]Error loading questions JSON:[/bold red] {e}")
            sys.exit(1)

        console.print(f"[bold yellow]Igniting Laya on {args.questions}...[/bold yellow]")
        agent = bootstrap(questions=questions_data, samples_per_class=args.samples, generator=args.generator)
        console.print(f"[bold green]🔥 Ready to serve decisions at sub-35ms![/bold green]")

    elif args.command == "ui":
        from .app import launch_ui
        launch_ui(port=args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
