import json
import logging
import time
from datetime import datetime
from functools import wraps
from itertools import batched
from pathlib import Path

from rich.console import Console
from rich.live import Live

import sieve.config as config
import sieve.utils as utils
from sieve.batchtracker import BatchTracker
from sieve.openai import OpenAIManager
from sieve.rayyan import RayyanManager


def requires_services(*required_services):
    """
    Ensures all listed services are not None before running the method.
    Usage: @requires_services("asana", "airtable")
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            missing = [s for s in required_services if getattr(self, s, None) is None]

            if missing:
                raise RuntimeError(f"Missing required services: {', '.join(missing)}")

            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class IntegrationManager:
    def __init__(
        self,
        rayyan_manager: RayyanManager | None = None,
        openai_manager: OpenAIManager | None = None,
        batch_tracker: BatchTracker | None = None,
        console: Console | None = None,
        debug: bool = False,
    ):
        self.rayyan = rayyan_manager
        self.openai = openai_manager
        self.tracker = batch_tracker
        self.console = console or Console()
        self.debug = debug
        self.logger = logging.getLogger("sieve")

    @requires_services("openai", "rayyan")
    def screen_abstract(self, article: dict):
        assert self.openai and self.rayyan

        abstracts = article.get("abstracts", [])

        if not abstracts:
            # No abstract to screen
            return

        abstract_text = abstracts[0].get("content", "")
        if not abstract_text:
            return

        decision = self.openai.screen_record_abstract(abstract_text)
        if decision is None:
            # Something with the LLM failed
            return

        decision_dict = decision.model_dump()

        self._action_screening_decision(decision_dict, article["id"], is_abstract=True)

    @requires_services("openai", "rayyan")
    def screen_fulltext(self, article: dict):
        assert self.openai and self.rayyan

        pdf_path = self.rayyan.download_pdf(article)
        if pdf_path is None:
            # This shouldn't happen, but you never know
            return
        decision = self.openai.screen_record_fulltext(pdf_path)
        if decision is None:
            # Something with the LLM failed
            return

        decision_dict = decision.model_dump()

        self._action_screening_decision(decision_dict, article["id"], is_abstract=False)

    @requires_services("openai", "rayyan", "tracker")
    def create_abstract_screening_batch(self, articles: list[dict]):
        assert self.openai and self.rayyan and self.tracker

        self._log(f"Preparing abstract screening batch for {len(articles)} articles...")
        requests = []

        for article in articles:
            abstracts = article.get("abstracts", [])
            if not abstracts:
                plan = {config.RAYYAN_LABELS["abstract_missing"]: 1}
                self._log(f"No abstract found for {article['id']}, updating labels")
                self.rayyan.update_article_labels(article["id"], plan)
                continue

            abstract_text = abstracts[0].get("content", "")
            if not abstract_text:
                plan = {config.RAYYAN_LABELS["abstract_missing"]: 1}
                self._log(f"No abstract text for {article['id']}, updating labels")
                self.rayyan.update_article_labels(article["id"], plan)
                continue

            custom_id = f"abstract-{article['id']}"
            body = self.openai.prepare_abstract_body(abstract_text)
            request = self.openai.create_batch_row(custom_id, body)
            requests.append(request)
            plan = {config.RAYYAN_LABELS["batch_pending"]: 1}
            self.rayyan.update_article_labels(article["id"], plan)

        if requests:
            self._submit_batch(requests, "abstract_screen")
        else:
            self._log("No requests to submit for abstract screening batch.")

    @requires_services("openai", "rayyan", "tracker")
    def create_fulltext_screening_batch(self, articles: list[dict]):
        assert self.openai and self.rayyan and self.tracker

        self._log(f"Preparing fulltext screening batch for {len(articles)} articles...")
        requests = []

        for article in articles:
            pdf_path = self.rayyan.download_pdf(article)
            if not pdf_path:
                self._log(f"No PDF found for {article['id']}, skipping.")
                continue

            try:
                file = self.openai.upload_file(pdf_path)
            except Exception as e:
                self._log(f"Failed to upload PDF for {article['id']}: {e}")
                continue

            # 3. Prepare Request
            custom_id = f"fulltext-{article['id']}"
            body = self.openai.prepare_fulltext_body(file.id)
            request = self.openai.create_batch_row(custom_id, body)
            requests.append(request)
            plan = {config.RAYYAN_LABELS["batch_pending"]: 1}
            self.rayyan.update_article_labels(article["id"], plan)

        if requests:
            self._submit_batch(requests, "fulltext_screen")

    @requires_services("openai")
    def process_pending_batches(self, pending: dict):
        assert self.openai

        for batch_id, info in pending.items():
            self._log(f"Checking status for batch {batch_id} ({info['type']})...")
            try:
                batch = self.openai.retrieve_batch(batch_id)
            except Exception as e:
                self._log(f"Could not retrieve batch {batch_id}: {e}")
                continue

            if batch.status == "completed":
                self._log("Batch completed! Downloading results...")
                if batch.output_file_id:
                    self._handle_completed_batch(
                        batch.output_file_id, info["type"], batch_id
                    )
                else:
                    self._log("Batch completed but has no output file ID.")
                    if batch.error_file_id:
                        self._log(f"Batch has errors. File ID: {batch.error_file_id}")
                        # TODO: handle errors

            elif batch.status in ["failed", "expired", "cancelled"]:
                self._log(f"Batch {batch_id} ended with status: {batch.status}")
            else:
                self._log(f"Batch {batch_id} is {batch.status}")

    @requires_services("rayyan")
    def monitor_rayyan(self, live: Live, stats: dict):
        assert self.rayyan
        try:
            stats["status"] = "[cyan]Checking Rayyan for unscreened abstracts...[/cyan]"
            live.update(utils.create_stats_table(stats))
            self._log("Checking Rayyan for unscreened abstracts...")
            unscreened_abstracts = self.rayyan.get_unscreened_abstracts()

            stats["status"] = "[cyan]Checking Rayyan for unscreened fulltexts...[/cyan]"
            live.update(utils.create_stats_table(stats))
            self._log("Checking Rayyan for unscreened fulltexts...")
            unscreened_fulltexts = self.rayyan.get_unscreened_fulltexts()

            stats["last_check"]["rayyan"] = datetime.now().strftime("%H:%M:%S")
            stats["last_sync"]["rayyan"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            stats["consecutive_errors"]["rayyan"] = 0
            stats["total_polls"]["rayyan"] += 1
            stats["total_syncs"]["rayyan"] += 1

            live.update(utils.create_stats_table(stats))
            self._log("All Rayyan checks complete.")
            return (
                unscreened_abstracts,
                unscreened_fulltexts,
                stats,
            )

        except Exception as e:
            stats["consecutive_errors"]["rayyan"] += 1
            stats["status"] = f"[red]Rayyan Error: {e}[/red]"
            self._log(f"Rayyan monitoring error: {e}", "error")
            live.update(utils.create_stats_table(stats))
            return None, None, None, stats

    @requires_services("openai", "tracker")
    def create_batches(
        self,
        live: Live,
        stats: dict,
        unscreened_abstracts: list | None,
        unscreened_fulltexts: list | None,
        max_batch_size_abs: int = 1000,
        max_batch_size_ft: int = 100,
        max_num_batches_per_type: int = 3,
    ):
        try:
            if unscreened_abstracts:
                stats["status"] = (
                    "[yellow]Creating abstract screening batches...[/yellow]"
                )
                live.update(utils.create_stats_table(stats))
                self._log(
                    f"Creating batches for {len(unscreened_abstracts)} abstracts..."
                )
                batch_count = 0
                for batch in batched(
                    unscreened_abstracts, max_batch_size_abs, strict=False
                ):
                    if batch_count >= max_num_batches_per_type:
                        self._log("Reached max number of batches for this cycle.")
                        break
                    self.create_abstract_screening_batch(list(batch))
                    stats["pending_batches"]["abstract_screen"] += 1
                    live.update(utils.create_stats_table(stats))
                    batch_count += 1
            if unscreened_fulltexts:
                stats["status"] = (
                    "[yellow]Creating fulltext screening batches...[/yellow]"
                )
                live.update(utils.create_stats_table(stats))
                self._log(
                    f"Creating batches for {len(unscreened_fulltexts)} fulltexts..."
                )
                batch_count = 0
                for batch in batched(
                    unscreened_fulltexts, max_batch_size_ft, strict=False
                ):
                    if batch_count >= max_num_batches_per_type:
                        self._log("Reached max number of batches for this cycle.")
                        break
                    self.create_fulltext_screening_batch(list(batch))
                    stats["pending_batches"]["fulltext_screen"] += 1
                    live.update(utils.create_stats_table(stats))
                    batch_count += 1
            stats["consecutive_errors"]["openai"] = 0

        except Exception as e:
            stats["consecutive_errors"]["openai"] += 1
            stats["status"] = f"[red]Batch Creation Error: {e}[/red]"
            self._log(f"Batch creation error: {e}", "error")
            live.update(utils.create_stats_table(stats))

        return stats

    @requires_services("openai")
    def process_pending_batches_cli(
        self, live: Live, stats: dict, pending: dict, max_batches: int = 5
    ) -> dict:
        assert self.openai

        batch_count = 0
        for batch_id, info in pending.items():
            if batch_count > max_batches:
                self._log("Reached max number of batches for this cycle.")
                break
            stats["status"] = "[cyan]Checking batch status...[/cyan]"
            stats["last_check"]["openai"] = datetime.now().strftime("%H:%M:%S")
            stats["total_polls"]["openai"] += 1
            live.update(utils.create_stats_table(stats))
            self._log(f"Checking status for batch {batch_id} ({info['type']})...")
            try:
                batch = self.openai.retrieve_batch(batch_id)
            except Exception as e:
                self._log(f"Could not retrieve batch {batch_id}: {e}")
                continue

            if batch.status == "completed":
                self._log("Batch completed! Downloading results...")
                if batch.output_file_id:
                    stats["status"] = (
                        f"[yellow]Processing completed {info['type']} batch [/yellow]"
                    )
                    stats["total_syncs"]["openai"] += 1
                    stats["last_sync"]["openai"] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    live.update(utils.create_stats_table(stats))
                    self._handle_completed_batch(
                        batch.output_file_id, info["type"], batch_id
                    )
                    stats["pending_batches"][info["type"]] -= 1
                    stats["last_sync"]["openai"] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    live.update(utils.create_stats_table(stats))
                    batch_count += 1
                else:
                    self._log("Batch completed but has no output file ID.")
                    if batch.error_file_id:
                        self._log(f"Batch has errors. File ID: {batch.error_file_id}")
                        # TODO: handle errors

            elif batch.status in ["failed", "expired", "cancelled"]:
                self._log(f"Batch {batch_id} ended with status: {batch.status}")
            else:
                self._log(f"Batch {batch_id} is {batch.status}")

        return stats

    def update_stats_pending_batches(
        self, live: Live, stats: dict, pending: dict
    ) -> dict:
        stats["pending_batches"] = {
            "abstract_screen": sum(
                1 for b in pending.values() if b["type"] == "abstract_screen"
            ),
            "fulltext_screen": sum(
                1 for b in pending.values() if b["type"] == "fulltext_screen"
            ),
        }
        live.update(utils.create_stats_table(stats))
        return stats

    @requires_services("openai", "tracker")
    def _handle_completed_batch(
        self, output_file_id: str, batch_type: str, batch_id: str
    ):
        assert self.openai and self.tracker
        file_response = self.openai.client.files.content(output_file_id)
        content = file_response.text

        results = []
        for line in content.split("\n"):
            if line.strip():
                results.append(json.loads(line))

        self._log(f"Processing {len(results)} results for {batch_type}...")

        if batch_type == "abstract_screen":
            self._process_abstract_results(results)
        elif batch_type == "fulltext_screen":
            self._process_fulltext_results(results)

        self.tracker.mark_completed(batch_id)

    @requires_services("rayyan")
    def _action_screening_decision(
        self, decision: dict, article_id: int, is_abstract: bool, is_batch: bool = False
    ):
        assert self.rayyan

        if decision["vote"] not in ["include", "exclude"]:
            # Invalid vote, skip
            return

        if decision["vote"] == "exclude":
            # TODO: These should actually do a vote on rayyan
            if is_abstract:
                plan = {config.RAYYAN_LABELS["abstract_excluded"]: 1}
            else:
                plan = {config.RAYYAN_LABELS["excluded"]: 1}
                if decision.get("triggered_exclusion"):
                    excl_reason_idx = decision["triggered_exclusion"][0]
                    excl_label = config.RAYYAN_EXCLUSION_LABELS[excl_reason_idx - 1]
                    plan[excl_label] = 1
                elif decision.get("failed_inclusion"):
                    excl_reason_idx = decision["failed_inclusion"][0]
                    excl_label = config.RAYYAN_EXCLUSION_LABELS[excl_reason_idx - 1]
                    plan[excl_label] = 1
            rationale = decision["rationale"]
            rationale = utils.sanitize_text(rationale)
            if len(rationale) > 1000:
                rationale = rationale[:981] + "..."
            try:
                self.rayyan.create_article_note(
                    article_id, f"LLM Reasoning: {rationale}"
                )
            except Exception as e:
                self._log(f"Failed to create note for article {article_id}: {e}")
        else:
            if is_abstract:
                plan = {config.RAYYAN_LABELS["abstract_included"]: 1}
            else:
                plan = {
                    config.RAYYAN_LABELS["included"]: 1,
                }

        if is_batch:
            plan[config.RAYYAN_LABELS["batch_pending"]] = -1

        self.rayyan.update_article_labels(article_id, plan)

    @requires_services("rayyan", "openai")
    def _process_abstract_results(self, results: list):
        assert self.rayyan and self.openai
        for item in results:
            try:
                article_id = int(item["custom_id"].split("-")[-1])
                response_body = item["response"]["body"]

                if item["response"]["status_code"] != 200:
                    self._log(
                        f"Error in batch result for {article_id}: {response_body}"
                    )
                    plan = {config.RAYYAN_LABELS["batch_pending"]: -1}
                    self.rayyan.update_article_labels(article_id, plan)
                    continue

                content_str = response_body["output"][0]["content"][0]["text"]
                decision = self.openai.parse_screening_decision(content_str)
                decision_dict = decision.model_dump()

                self._action_screening_decision(
                    decision_dict, article_id, is_abstract=True, is_batch=True
                )

            except Exception as e:
                self._log(f"Failed to process abstract result: {e}")

    @requires_services("rayyan", "openai")
    def _process_fulltext_results(self, results: list):
        assert self.rayyan and self.openai
        for item in results:
            try:
                article_id = int(item["custom_id"].split("-")[-1])
                response_body = item["response"]["body"]

                if item["response"]["status_code"] != 200:
                    self._log(
                        f"Error in batch result for {article_id}: {response_body}"
                    )
                    plan = {config.RAYYAN_LABELS["batch_pending"]: -1}
                    self.rayyan.update_article_labels(article_id, plan)
                    continue

                content_str = response_body["output"][0]["content"][0]["text"]
                decision = self.openai.parse_screening_decision(content_str)
                decision_dict = decision.model_dump()

                self._action_screening_decision(
                    decision_dict, article_id, is_abstract=False, is_batch=True
                )

            except Exception as e:
                self._log(f"Failed to process fulltext result: {e}")

    @requires_services("openai", "tracker")
    def _submit_batch(self, requests: list, batch_type: str):
        """Internal helper to write JSONL, upload, and create batch."""
        assert self.openai and self.tracker

        timestamp = int(time.time())
        filename = f"batch_input_{batch_type}_{timestamp}.jsonl"

        self._log(f"Writing {len(requests)} requests to {filename}...")
        with open(filename, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")

        try:
            self._log("Creating batch job...")
            batch_job = self.openai.create_batch(filename, batch_type)
            self.tracker.add_batch(batch_job.id, batch_type)
            self._log(f"Batch {batch_job.id} submitted successfully.")

        except Exception as e:
            self._log(f"Error submitting batch: {e}")
        finally:
            if Path(filename).exists():
                Path(filename).unlink()

    def _log(self, message: str | object, level: str = "info", **kwargs):
        msg_str = str(message)
        if level.lower() == "error":
            self.logger.error(msg_str)
        elif level.lower() == "warning":
            self.logger.warning(msg_str)
        else:
            self.logger.info(msg_str)

        if self.debug:
            self.console.log(message, **kwargs)
