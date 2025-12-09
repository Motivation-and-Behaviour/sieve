import json
import logging
import time
from datetime import datetime
from functools import wraps
from itertools import batched
from pathlib import Path

from pyairtable.api.types import RecordDict
from rich.console import Console
from rich.live import Live

import bigger_picker.config as config
import bigger_picker.utils as utils
from bigger_picker.airtable import AirtableManager
from bigger_picker.asana import AsanaManager
from bigger_picker.batchtracker import BatchTracker
from bigger_picker.datamodels import Article, ArticleLLMExtract
from bigger_picker.openai import OpenAIManager
from bigger_picker.rayyan import RayyanManager


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
        asana_manager: AsanaManager | None = None,
        rayyan_manager: RayyanManager | None = None,
        airtable_manager: AirtableManager | None = None,
        openai_manager: OpenAIManager | None = None,
        batch_tracker: BatchTracker | None = None,
        console: Console | None = None,
        debug: bool = False,
    ):
        self.asana = asana_manager
        self.rayyan = rayyan_manager
        self.airtable = airtable_manager
        self.openai = openai_manager
        self.tracker = batch_tracker
        self.console = console or Console()
        self.debug = debug
        self.logger = logging.getLogger("bigger_picker")

    @requires_services("asana", "airtable")
    def sync_airtable_and_asana(
        self,
    ) -> None:
        assert self.asana and self.airtable
        self._log("Getting Asana tasks")
        self.asana.get_tasks(refresh=True)

        tasks = {}

        self._log("Getting Asana task IDs")
        for task in self.asana.tasks:
            tasks[
                self.asana.get_custom_field_value(
                    task, config.ASANA_CUSTOM_FIELD_IDS["BPIPD"]
                )
            ] = task

        self._log("Getting Airtable records")
        datasets = self.airtable.tables["Datasets"].all()

        for dataset in datasets:
            dataset_bpipd = dataset["fields"].get("Dataset ID", None)

            if dataset_bpipd in tasks:
                # If the dataset has a matching task, update it
                self.update_task_from_dataset(
                    tasks[dataset_bpipd],
                    dataset,
                )
            else:
                # If the dataset does not have a matching task, create one
                self._log(f"Creating task for {dataset_bpipd}")
                updated_task = self.create_task_from_dataset(dataset)

                # Update the Airtable records with the Asana task IDs
                task_bpipd = self.asana.get_custom_field_value(
                    updated_task, config.ASANA_CUSTOM_FIELD_IDS["BPIPD"]
                )
                payload = {"Dataset ID": task_bpipd}
                self._log(f"Updating Airtable record for {task_bpipd}")
                self.airtable.update_record("Datasets", dataset["id"], payload)
        self._log("Starting status sync")
        self.update_airtable_statuses()
        self._log("Status sync complete")

    @requires_services("asana", "airtable")
    def update_task_from_dataset(self, task: dict, dataset: RecordDict) -> dict:
        assert self.asana and self.airtable

        dataset_vals = {
            "name": dataset["fields"].get("Dataset Name", None),
            "value": round(dataset["fields"]["Dataset Value"], 3)
            if dataset["fields"].get("Dataset Value", None) is not None
            else None,
            "url": self.airtable.make_url(dataset["id"]),
        }
        task_vals = {
            "name": task.get("name", None),
            "value": self.asana.get_custom_field_value(
                task, config.ASANA_CUSTOM_FIELD_IDS["Dataset Value"]
            ),
            "url": self.asana.get_custom_field_value(
                task, config.ASANA_CUSTOM_FIELD_IDS["Airtable Data"]
            ),
        }

        if dataset_vals == task_vals:
            return task

        searches = set()
        for search_str in dataset["fields"].get("Searches", []):
            if not search_str:
                continue
            labels = search_str.split(",")
            for label in labels:
                label = label.strip()
                searches.add(label)

        dataset_searches = [
            config.ASANA_SEARCHES_ENUM_VALUES.get(search) for search in searches
        ]

        update_payload = {
            "data": {
                "name": dataset_vals["name"],
                "custom_fields": {
                    config.ASANA_CUSTOM_FIELD_IDS["Dataset Value"]: dataset_vals[
                        "value"
                    ],
                    config.ASANA_CUSTOM_FIELD_IDS["Airtable Data"]: dataset_vals["url"],
                    config.ASANA_CUSTOM_FIELD_IDS["Searches"]: dataset_searches,
                },
            }
        }

        task_gid = task.get("gid", None)
        if task_gid is None:
            raise ValueError("Task GID is missing in the provided task dictionary.")

        self._log(
            f"Updating task {task_gid} from dataset "
            + f"{dataset['fields'].get('Dataset ID', 'Unknown')}"
        )
        return self.asana.update_task(update_payload, task_gid)

    @requires_services("asana", "airtable")
    def create_task_from_dataset(self, dataset: RecordDict) -> dict:
        assert self.asana and self.airtable

        dataset_status = dataset["fields"].get("Status", None)
        if dataset_status is None:
            dataset_status = "Awaiting Triage"

        dataset_value = dataset["fields"].get("Dataset Value", None)
        airtable_url = self.airtable.make_url(dataset["id"])
        dataset_status_id = config.ASANA_STATUS_ENUM_VALUES.get(dataset_status, None)

        searches = set()
        for search_str in dataset["fields"].get("Searches", []):
            if not search_str:
                continue
            labels = search_str.split(",")
            for label in labels:
                label = label.strip()
                searches.add(label)

        dataset_searches = [
            config.ASANA_SEARCHES_ENUM_VALUES.get(search) for search in searches
        ]

        task_payload = {
            "data": {
                "name": dataset["fields"]["Dataset Name"],
                "projects": self.asana.project_id,
                "custom_fields": {
                    config.ASANA_CUSTOM_FIELD_IDS["Dataset Value"]: dataset_value,
                    config.ASANA_CUSTOM_FIELD_IDS["Airtable Data"]: airtable_url,
                    config.ASANA_CUSTOM_FIELD_IDS["Status"]: dataset_status_id,
                    config.ASANA_CUSTOM_FIELD_IDS["Searches"]: dataset_searches,
                },
            }
        }

        created_task = self.asana.create_task(task_payload)

        # Set the BPIPD custom field to the Dataset ID
        updated_created_task = self.asana.fetch_task_with_custom_field(
            created_task["gid"], config.ASANA_CUSTOM_FIELD_IDS["BPIPD"]
        )

        if updated_created_task is None:
            raise ValueError(
                f"Failed to fetch the created task with ID {created_task['gid']}."
            )

        created_task_bpipd = self.asana.get_custom_field_value(
            updated_created_task, config.ASANA_CUSTOM_FIELD_IDS["BPIPD"]
        )

        payload = {"Dataset ID": created_task_bpipd}
        self.airtable.update_record("Datasets", dataset["id"], payload)

        return updated_created_task

    @requires_services("asana", "airtable")
    def update_airtable_statuses(self) -> None:
        assert self.asana and self.airtable

        self._log("Getting Asana tasks")
        # Always force refresh since we need up-to-date statuses from Asana
        tasks = self.asana.get_tasks(refresh=True)
        status_map = {}
        for task in tasks:
            status_dict = self.asana.get_custom_field_value(
                task, config.ASANA_CUSTOM_FIELD_IDS["Status"]
            )
            if status_dict is None:
                # For some reason, this task does not have a status set.
                continue

            if isinstance(status_dict, dict):
                status_name = status_dict.get("name")
            else:
                status_name = None

            status_map[
                self.asana.get_custom_field_value(
                    task, config.ASANA_CUSTOM_FIELD_IDS["BPIPD"]
                )
            ] = status_name

        self._log("Getting Airtable records")
        datasets = self.airtable.tables["Datasets"].all()
        records_updated = 0
        records_not_updated = 0

        for dataset in datasets:
            dataset_bpipd = dataset["fields"].get("Dataset ID", None)

            if dataset_bpipd is None:
                # We are out of sync, so we skip this dataset.
                continue

            if dataset_bpipd in status_map:
                # If the dataset has a matching task, update it
                task_status = status_map[dataset_bpipd]
                dataset_status = dataset["fields"].get("Status", None)

                if task_status != dataset_status:
                    payload = {"Status": task_status}
                    records_updated += 1
                    self._log(f"Updating Airtable record {dataset['id']}")
                    self.airtable.update_record("Datasets", dataset["id"], payload)

                else:
                    records_not_updated += 1

        self._log(
            f"Updated {records_updated} records; {records_not_updated} unchanged."
        )

    @requires_services("airtable")
    def upload_extraction_to_airtable(
        self,
        llm_extraction: ArticleLLMExtract,
        article_metadata: dict,
        pdf_path: str | None = None,
    ) -> RecordDict:
        assert self.airtable

        def _convert_to_title_case(value):
            if isinstance(value, str):
                return value.title()
            elif isinstance(value, list):
                return [
                    item.title() if isinstance(item, str) else item for item in value
                ]
            return value

        extraction_dict = ArticleLLMExtract.model_dump(
            llm_extraction, by_alias=True, exclude_unset=True
        )

        article = Article(
            **article_metadata,
            **extraction_dict,
        )

        article_dict = article.model_dump(by_alias=True, exclude_unset=True)

        populations = article_dict.pop("populations", [])
        screen_time_measures = article_dict.pop("screen_time_measures", [])
        outcomes = article_dict.pop("outcomes", [])
        dataset_name = article_dict.pop("Dataset Name", None)

        article_record = self.airtable.create_record("Articles", article_dict)
        article_record_id = article_record["id"]

        if pdf_path is not None:
            self.airtable.upload_attachment(
                "Articles", article_record_id, "Fulltext", pdf_path
            )

        for population in populations:
            population["Rayyan ID"] = [article_record_id]
            self.airtable.create_record("Populations", population)

        for screen_time_measure in screen_time_measures:
            for key in screen_time_measure.keys():
                screen_time_measure[key] = _convert_to_title_case(
                    screen_time_measure[key]
                )
            screen_time_measure["Rayyan ID"] = [article_record_id]
            self.airtable.create_record("Screen Time Measures", screen_time_measure)

        for outcome in outcomes:
            for key in outcome.keys():
                outcome[key] = _convert_to_title_case(outcome[key])
            outcome["Rayyan ID"] = [article_record_id]
            self.airtable.create_record("Outcomes", outcome)

        # Create the dataset and sync to Airtable
        if dataset_name is None:
            author_last = article_metadata["Authors"].split(",")[0]
            year = article_metadata["Year"]
            dataset_name = f"{author_last}, {year}"

        dataset_payload = {
            "Dataset Name": dataset_name,
            "Status": "Awaiting Triage",
            "Total Sample Size": article_dict.get("Total Sample Size", None),
            "Dataset Contact Name": article_dict.get("Corresponding Author", None),
            "Dataset Contact Email": article_dict.get(
                "Corresponding Author Email", None
            ),
            "Countries of Data": article_dict.get("Countries of Data", []),
            "Articles: IDs": [article_record_id],
        }

        return self.airtable.create_record("Datasets", dataset_payload)

    @requires_services("airtable", "asana", "rayyan", "openai")
    def process_article(self, article: dict):
        assert self.airtable and self.asana and self.rayyan and self.openai

        pdf_path = self.rayyan.download_pdf(article)
        if pdf_path is None:
            # This shouldn't happen, but just in case, we skip this article
            return
        article_metadata = self.rayyan.extract_article_metadata(article)
        llm_extraction = self.openai.extract_article_info(pdf_path)
        if llm_extraction is None:
            # If the LLM extraction failed, we skip this article
            return
        dataset = self.upload_extraction_to_airtable(
            llm_extraction, article_metadata, pdf_path
        )
        self.create_task_from_dataset(dataset)
        plan = {
            self.rayyan.unextracted_label: -1,
            self.rayyan.extracted_label: 1,
        }
        self.rayyan.update_article_labels(article["id"], plan)

    @requires_services("airtable")
    def updated_datasets_scores(self) -> bool:
        assert self.airtable

        self._log("Scoring datasets...")

        datasets_included_statuses = ["Included", "Agreed & Awaiting Data"]
        datasets_potential_statuses = [
            "Validated",
            "Mail Merge",
            "Contacting Authors",
            "Awaiting Triage",
        ]
        self._log("Fetching datasets from Airtable")
        datasets = self.airtable.tables["Datasets"].all()

        datasets_included = []
        datasets_potential = []
        for dataset in datasets:
            if dataset["fields"].get("Status") in datasets_included_statuses:
                datasets_included.append(utils.fix_dataset(dataset))

            elif dataset["fields"].get("Status") in datasets_potential_statuses:
                datasets_potential.append(utils.fix_dataset(dataset))

        updated_any_datasets = False

        # Precompute data needed for scores
        year_min, year_max = utils.compute_year_range(
            datasets_included, datasets_potential
        )
        age_cache = utils.compute_age_cache(datasets_included)

        for dataset in datasets_potential:
            dataset_value = round(
                utils.compute_dataset_value(dataset, year_min, year_max, age_cache),
                3,
            )
            payload = {"Dataset Value": dataset_value}
            if dataset["fields"].get("Dataset Value") != dataset_value:
                updated_any_datasets = True
                self.airtable.update_record("Datasets", dataset["id"], payload)

        return updated_any_datasets

    @requires_services("airtable")
    def mark_duplicates(self, threshold=0.51):
        assert self.airtable

        self._log("Marking duplicates...")
        datasets = self.airtable.tables["Datasets"].all()
        duplicates = utils.identify_duplicate_datasets(datasets, threshold=threshold)

        for dataset in datasets:
            dataset_id = dataset["id"]
            if dataset_id in duplicates:
                dataset_duplicates = dataset["fields"].get("Duplicates", [])

                if set(dataset_duplicates) == set(duplicates[dataset_id]):
                    # All the duplicates are already on airtable
                    continue

                for duplicate in duplicates[dataset_id]:
                    if duplicate not in dataset_duplicates:
                        dataset_duplicates.append(duplicate)
                payload = {"Possible Duplicates": dataset_duplicates}
                self.airtable.update_record("Datasets", dataset_id, payload)

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

    @requires_services("asana", "airtable")
    def sync(self):
        self.sync_airtable_and_asana()  # HACK: need to update status first
        any_datasets_updated = self.updated_datasets_scores()
        if any_datasets_updated:
            self._log("Datasets updated, syncing Airtable and Asana again.")
            self.sync_airtable_and_asana()
        else:
            self._log("No datasets updated, skipping second sync.")

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

    @requires_services("openai", "rayyan", "tracker")
    def create_extraction_batch(self, articles: list[dict]):
        assert self.openai and self.rayyan and self.tracker

        self._log(f"Preparing extraction batch for {len(articles)} articles...")
        requests = []

        for article in articles:
            pdf_path = self.rayyan.download_pdf(article)
            if not pdf_path:
                continue

            try:
                file = self.openai.upload_file(pdf_path)
            except Exception as e:
                self._log(f"Failed to upload PDF for {article['id']}: {e}")
                continue

            custom_id = f"extraction-{article['id']}"
            body = self.openai.prepare_extraction_body(file.id)
            request = self.openai.create_batch_row(custom_id, body)
            requests.append(request)
            plan = {config.RAYYAN_LABELS["batch_pending"]: 1}
            self.rayyan.update_article_labels(article["id"], plan)

        if requests:
            self._submit_batch(requests, "extraction")

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

    @requires_services("asana")
    def monitor_asana(self, live: Live, stats: dict) -> dict:
        assert self.asana
        try:
            stats["status"] = "[cyan]Checking Asana...[/cyan]"
            stats["total_polls"]["asana"] += 1
            live.update(utils.create_stats_table(stats))

            self._log("Checking Asana events...")
            events = self.asana.get_events()
            stats["last_check"]["asana"] = datetime.now().strftime("%H:%M:%S")

            if events or stats["total_syncs"]["asana"] == 0:
                stats["consecutive_errors"]["asana"] = 0
                stats["status"] = "[yellow]Syncing Asana...[/yellow]"
                live.update(utils.create_stats_table(stats))

                if events:
                    self._log("Changes detected in Asana, syncing...")
                if stats["total_syncs"]["asana"] == 0:
                    self._log("Performing initial Asana sync...")
                self.sync()
                stats["total_syncs"]["asana"] += 1
                self._log("Sync complete. Clearing Asana events...")
                self.asana.get_events()  # Clear events after sync
                stats["status"] = "[green]✓ Asana sync complete[/green]"
                stats["last_sync"]["asana"] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            else:
                stats["status"] = "[green]Idle[/green]"
                live.update(utils.create_stats_table(stats))

        except Exception as e:
            stats["consecutive_errors"]["asana"] += 1
            stats["status"] = f"[red]Asana Error: {e}[/red]"
            self._log(f"Asana monitoring error: {e}", "error")
            live.update(utils.create_stats_table(stats))

        return stats

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

            stats["status"] = "[cyan]Checking Rayyan for unextracted articles...[/cyan]"
            live.update(utils.create_stats_table(stats))
            self._log("Checking Rayyan for unextracted articles...")
            unextracted_articles = self.rayyan.get_unextracted_articles()

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
                unextracted_articles,
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
        unextracted_articles: list | None,
        max_batch_size_abs: int = 1000,
        max_batch_size_ft: int = 100,
        max_batch_size_ext: int = 100,
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
                for batch in batched(unscreened_abstracts, max_batch_size_abs):
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
                for batch in batched(unscreened_fulltexts, max_batch_size_ft):
                    if batch_count >= max_num_batches_per_type:
                        self._log("Reached max number of batches for this cycle.")
                        break
                    self.create_fulltext_screening_batch(list(batch))
                    stats["pending_batches"]["fulltext_screen"] += 1
                    live.update(utils.create_stats_table(stats))
                    batch_count += 1
            if unextracted_articles:
                stats["status"] = "[yellow]Creating extraction batches...[/yellow]"
                live.update(utils.create_stats_table(stats))
                self._log(
                    f"Creating batches for {len(unextracted_articles)} extractions..."
                )
                batch_count = 0
                for batch in batched(unextracted_articles, max_batch_size_ext):
                    if batch_count >= max_num_batches_per_type:
                        self._log("Reached max number of batches for this cycle.")
                        break
                    self.create_extraction_batch(list(batch))
                    stats["pending_batches"]["extraction"] += 1
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
            "extraction": sum(1 for b in pending.values() if b["type"] == "extraction"),
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
        elif batch_type == "extraction":
            self._process_extraction_results(results)

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
                    config.RAYYAN_LABELS["unextracted"]: 1,
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

    @requires_services("rayyan", "airtable", "asana", "openai")
    def _process_extraction_results(self, results: list):
        assert self.rayyan and self.airtable and self.asana and self.openai

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
                llm_extraction = self.openai.parse_extraction_result(content_str)

                article = self.rayyan.get_article_by_id(article_id)
                article_metadata = self.rayyan.extract_article_metadata(article)
                pdf_path = self.rayyan.download_pdf(article)

                dataset = self.upload_extraction_to_airtable(
                    llm_extraction, article_metadata, pdf_path=pdf_path
                )

                self.create_task_from_dataset(dataset)

                plan = {
                    self.rayyan.unextracted_label: -1,
                    self.rayyan.extracted_label: 1,
                    config.RAYYAN_LABELS["batch_pending"]: -1,
                }
                self.rayyan.update_article_labels(article_id, plan)
                self._log(f"Successfully processed extraction for {article_id}")

            except Exception as e:
                self._log(f"Failed to process extraction result: {e}")

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
