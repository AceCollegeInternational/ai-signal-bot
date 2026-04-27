import pandas as pd
import json
import os
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table

console = Console()


def load_dataset(path: str = "logs/ml_dataset.jsonl") -> pd.DataFrame:
    """Load the JSONL dataset into a flat DataFrame."""
    if not os.path.exists(path):
        return pd.DataFrame()

    data = []
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)
            # Flatten features
            row = {
                "timestamp": entry["timestamp"],
                "symbol": entry["symbol"],
                "direction": entry["direction"],
                "pnl_pct": entry["pnl_pct"],
                "win": entry["win"],
            }
            row.update(entry["features"])
            data.append(row)

    return pd.DataFrame(data)


def show_stats():
    df = load_dataset()
    if df.empty:
        console.print(
            "[yellow]No ML data collected yet. Start trading to build your dataset![/yellow]"
        )
        return

    table = Table(title="ML Data Collection Stats")
    table.add_column("Total Trades", justify="right")
    table.add_column("Wins", justify="right", style="green")
    table.add_column("Losses", justify="right", style="red")
    table.add_column("Win Rate", justify="right", style="cyan")
    table.add_column("Features Count", justify="right")

    wins = len(df[df["win"] == 1])
    losses = len(df[df["win"] == 0])
    win_rate = (wins / len(df)) * 100
    features = len(df.columns) - 5  # Subtract metadata cols

    table.add_row(
        str(len(df)), str(wins), str(losses), f"{win_rate:.1f}%", str(features)
    )

    console.print(table)
    console.print(f"\n[bold blue]Dataset Location:[/bold blue] logs/ml_dataset.jsonl")
    console.print(f"[dim]Columns: {', '.join(df.columns[:10])}...[/dim]")


if __name__ == "__main__":
    show_stats()
