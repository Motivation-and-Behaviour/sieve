import logging
import unicodedata
from datetime import datetime
from logging.handlers import RotatingFileHandler

from rich.table import Table


def setup_logger(name: str = "sieve", log_file: str = "sieve.log") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def sanitize_text(text: str | None) -> str:
    if not text:
        return ""
    replacements = {
        "\u2018": "'",  # Left single quote
        "\u2019": "'",  # Right single quote
        "\u201c": '"',  # Left double quote
        "\u201d": '"',  # Right double quote
        "\u2013": "-",  # En-dash
        "\u2014": "-",  # Em-dash
        "…": "...",  # Ellipsis
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("\n", " ").replace("\r", "")

    return " ".join(text.split())


def create_stats_table(stats: dict) -> Table:
    def make_subtable(subgroups: dict, substats: dict) -> Table:
        platform_table = Table(show_header=False, show_edge=False)
        for key, value in subgroups.items():
            platform_table.add_row(value, str(substats[key]))
        return platform_table

    # Main table
    table = Table(
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
        title="Bigger Picker Status",
    )
    table.add_column("Metric", style="cyan", vertical="middle")
    table.add_column("Value", style="green", vertical="middle", justify="center")
    uptime = str(datetime.now() - stats["start_time"]).split(".")[0]
    table.add_row("Status", stats["status"])
    table.add_row("Uptime", uptime)

    # Subtables
    platforms_dict = {"rayyan": "Rayyan", "openai": "OpenAI"}
    last_check_table = make_subtable(platforms_dict, stats["last_check"])
    table.add_row("Last Check", last_check_table)
    last_sync_table = make_subtable(platforms_dict, stats["last_sync"])
    table.add_row("Last Sync", last_sync_table)
    total_syncs_table = make_subtable(platforms_dict, stats["total_syncs"])
    table.add_row("Total Syncs", total_syncs_table)
    total_polls_table = make_subtable(platforms_dict, stats["total_polls"])
    table.add_row("Total Polls", total_polls_table)
    pending_batches_table = make_subtable(
        {
            "abstract_screen": "Abstracts",
            "fulltext_screen": "Fulltexts",
            "extraction": "Extractions",
        },
        stats["pending_batches"],
    )
    table.add_row("Pending Batches", pending_batches_table)

    return table
