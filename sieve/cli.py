import os
import time
from datetime import datetime

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from sieve.batchtracker import BatchTracker
from sieve.integration import IntegrationManager
from sieve.openai import OpenAIManager
from sieve.rayyan import RayyanManager
from sieve.utils import create_stats_table, setup_logger

app = typer.Typer()


@app.command()
def screenft(
    dotenv_path: str = typer.Option(None, help="Path to .env file with credentials"),
    openai_api_key: str = typer.Option(None, help="OpenAI API key"),
    openai_model: str = typer.Option("gpt-5.2", help="OpenAI model to use"),
    rayyan_creds_path: str = typer.Option(
        None, help="Path to Rayyan credentials JSON file"
    ),
    max_articles: int = typer.Option(
        None, help="Maximum number of articles to process"
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug logging to console"
    ),
):
    setup_logger()

    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        load_dotenv(os.path.join(BASE_DIR, ".env"))

    console = Console()

    openai = OpenAIManager(openai_api_key, openai_model)
    rayyan = RayyanManager(rayyan_creds_path)
    integration = IntegrationManager(
        openai_manager=openai,
        rayyan_manager=rayyan,
        console=console,
        debug=debug,
    )

    assert integration.rayyan

    with console.status("Getting unscreened fulltexts..."):
        articles = integration.rayyan.get_unscreened_fulltexts(
            max_articles=max_articles
        )
        console.log(f"Found {len(articles)} unscreened fulltexts.")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),  #
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Screening fulltexts...", total=len(articles))
        for article in articles:
            integration.screen_fulltext(article)
            progress.advance(task, advance=1)
    console.log("Screening complete.")


@app.command()
def screenabstract(
    dotenv_path: str = typer.Option(None, help="Path to .env file with credentials"),
    openai_api_key: str = typer.Option(None, help="OpenAI API key"),
    openai_model: str = typer.Option("gpt-5.2", help="OpenAI model to use"),
    rayyan_creds_path: str = typer.Option(
        None, help="Path to Rayyan credentials JSON file"
    ),
    max_articles: int = typer.Option(
        None, help="Maximum number of articles to process"
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug logging to console"
    ),
):
    setup_logger()

    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        load_dotenv(os.path.join(BASE_DIR, ".env"))

    console = Console()

    openai = OpenAIManager(openai_api_key, openai_model)
    rayyan = RayyanManager(rayyan_creds_path)
    integration = IntegrationManager(
        openai_manager=openai,
        rayyan_manager=rayyan,
        console=console,
        debug=debug,
    )

    assert integration.rayyan

    with console.status("Getting unscreened abstracts..."):
        articles = integration.rayyan.get_unscreened_abstracts(
            max_articles=max_articles
        )

        console.log(f"Found {len(articles)} unscreened abstracts.")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),  #
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Screening abstracts...", total=len(articles))
        for article in articles:
            integration.screen_abstract(article)
            progress.advance(task, advance=1)
    console.log("Screening complete.")


@app.command()
def monitor(
    dotenv_path: str = typer.Option(None, help="Path to .env file with credentials"),
    openai_api_key: str = typer.Option(None, help="OpenAI API key"),
    openai_model: str = typer.Option("gpt-5.2", help="OpenAI model to use"),
    rayyan_creds_path: str = typer.Option(
        None, help="Path to Rayyan credentials JSON file"
    ),
    interval: int = typer.Option(
        60, help="Interval in seconds between checks for changes"
    ),
    max_errors: int = typer.Option(
        5, help="Maximum number of consecutive errors before stopping"
    ),
    full_frequency: int = typer.Option(5, help="Frequency of full syncs (in cycles)"),
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug logging to console"
    ),
):
    setup_logger()

    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        load_dotenv(os.path.join(BASE_DIR, ".env"))

    console = Console()

    integration = IntegrationManager(
        openai_manager=OpenAIManager(openai_api_key, openai_model),
        rayyan_manager=RayyanManager(rayyan_creds_path),
        batch_tracker=BatchTracker(),
        console=console,
        debug=debug,
    )

    assert integration.rayyan and integration.openai and integration.tracker

    stats = {
        "status": "[green]Running[/green]",
        "last_check": {"rayyan": "Never", "openai": "Never"},
        "last_sync": {"rayyan": "Never", "openai": "Never"},
        "total_syncs": {"rayyan": 0, "openai": 0},
        "total_polls": {"rayyan": 0, "openai": 0},
        "pending_batches": {"abstract_screen": 0, "fulltext_screen": 0},
        "start_time": datetime.now(),
        "consecutive_errors": {"rayyan": 0, "openai": 0},
    }

    try:
        with Live(
            create_stats_table(stats), refresh_per_second=1, console=console
        ) as live:
            cycle_count = 0
            while True:
                pending = integration.tracker.get_pending_batches()
                stats = integration.update_stats_pending_batches(live, stats, pending)

                if cycle_count % full_frequency == 0:
                    (
                        unscreened_abstracts,
                        unscreened_fulltexts,
                        stats,
                    ) = integration.monitor_rayyan(live, stats)

                    stats = integration.create_batches(
                        live,
                        stats,
                        unscreened_abstracts,
                        unscreened_fulltexts,
                    )
                    pending = integration.tracker.get_pending_batches()
                    stats = integration.update_stats_pending_batches(
                        live, stats, pending
                    )

                    stats = integration.process_pending_batches_cli(
                        live, stats, pending
                    )

                if (
                    stats["consecutive_errors"]["rayyan"] >= max_errors
                    or stats["consecutive_errors"]["openai"] >= max_errors
                ):
                    stats["status"] = "[bold red]Stopped (too many errors)[/bold red]"
                    live.update(create_stats_table(stats))
                    break

                for t in range(interval):
                    time_to_sync = interval - t
                    stats["status"] = (
                        f"[green]Idle (syncing in {time_to_sync}s)[/green]"
                    )
                    live.update(create_stats_table(stats))
                    time.sleep(1)

                cycle_count += 1

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped by user[/yellow]")


click_app = typer.main.get_command(app)

if __name__ == "__main__":
    app()
