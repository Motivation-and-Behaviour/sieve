import json
import os
import tempfile
from functools import partial
from itertools import batched

import requests
from rayyan import Rayyan
from rayyan.notes import Notes
from rayyan.reviews import Review

import sieve.config as config
from sieve.credentials import load_rayyan_credentials


class RayyanManager:
    def __init__(
        self,
        rayyan_creds_path: str | None = None,
        review_id: int = config.RAYYAN_REVIEW_ID,
    ):
        if rayyan_creds_path is None:
            rayyan_creds_path = load_rayyan_credentials()

        self._rayyan_creds_path = rayyan_creds_path
        self.rayyan_instance = Rayyan(rayyan_creds_path)
        self.review = Review(self.rayyan_instance)
        self.review_id = review_id
        self.notes_instance = Notes(self.rayyan_instance)

    def get_unscreened_abstracts(
        self, max_articles: int | None = None, batch_size: int = 1000
    ) -> list[dict]:
        labels_to_check = (
            list(config.RAYYAN_LABELS.values()) + config.RAYYAN_EXCLUSION_LABELS
        )

        # First just to get the total number of results
        results_params = {"start": 0, "length": 10, "extra[mode]": "undecided"}
        results = self._retry_on_auth_error(
            lambda: self.review.results(self.review_id, results_params)  # type: ignore
        )
        total_articles = results["recordsFiltered"]  # type: ignore
        batches = batched(range(0, total_articles), batch_size, strict=False)

        unscreened = []

        for batch in batches:
            results_params = {
                "start": batch[0],
                "length": len(batch),
                "extra[mode]": "undecided",
            }
            results = self._retry_on_auth_error(
                partial(self.review.results, self.review_id, results_params)  # type: ignore
            )

            for article in results["data"]:  # type: ignore
                article_labels = article.get("customizations", {}).get("labels", {})  # type: ignore
                if any(label in labels_to_check for label in article_labels):
                    continue

                unscreened.append(article)

            if max_articles is not None:
                if len(unscreened) >= max_articles:
                    break

        if max_articles is not None:
            unscreened = unscreened[:max_articles]

        return unscreened

    def get_unscreened_fulltexts(self, max_articles: int | None = None) -> list[dict]:
        # TODO: this should include batching like get_unscreened_abstracts
        results_params = {"extra[mode]": "included"}
        labels_to_check = (
            list(config.RAYYAN_LABELS.values()) + config.RAYYAN_EXCLUSION_LABELS
        )
        labels_to_check.remove(config.RAYYAN_LABELS["abstract_included"])
        labels_to_check.remove(config.RAYYAN_LABELS["abstract_excluded"])

        results = self._retry_on_auth_error(
            lambda: self.review.results(self.review_id, results_params)  # type: ignore
        )

        unscreened = []

        for article in results["data"]:  # type: ignore
            fulltext_id = self._get_fulltext_id(article)  # type: ignore
            if fulltext_id is None:
                continue

            article_labels = article.get("customizations", {}).get("labels", {})  # type: ignore
            if any(label in labels_to_check for label in article_labels):
                continue

            unscreened.append(article)

            if max_articles is not None:
                if len(unscreened) >= max_articles:
                    break

        if max_articles is not None:
            unscreened = unscreened[:max_articles]

        return unscreened

    def get_article_by_id(self, article_id: int) -> dict:
        results_params = {"extra[article_ids][]": str(article_id)}

        results = self._retry_on_auth_error(
            lambda: self.review.results(self.review_id, results_params)  # type: ignore
        )

        articles = results["data"]  # type: ignore

        if not articles:
            raise ValueError(f"Article with ID {article_id} not found.")

        return articles[0]  # type: ignore

    def update_article_labels(self, article_id: int, plan: dict) -> None:
        self._retry_on_auth_error(
            lambda: self.review.customize(self.review_id, article_id, plan)
        )

    def create_article_note(self, article_id: int, note_content: str) -> None:
        self._retry_on_auth_error(
            lambda: self.notes_instance.create_note(
                self.review_id, article_id, note_content
            )
        )

    def download_pdf(self, article: dict) -> str:
        fulltext_id = self._get_fulltext_id(article)

        if fulltext_id is None:
            raise ValueError("No fulltext found in the article.")

        fulltext_details = self.rayyan_instance.request.request_handler(
            method="GET", path=f"/api/v1/fulltexts/{fulltext_id}"
        )

        fulltext_url = fulltext_details.get("url", None)

        if fulltext_url is None:
            raise ValueError("No URL found for the fulltext.")

        temp_dir = tempfile.mkdtemp()

        filename = f"{article['id']}.pdf"

        file_path = os.path.join(temp_dir, filename)
        response = requests.get(str(fulltext_url))
        response.raise_for_status()

        with open(file_path, "wb") as f:
            f.write(response.content)

        return file_path

    def _retry_on_auth_error(self, operation, max_retries=3):
        for attempt in range(max_retries):
            try:
                return operation()
            except requests.HTTPError as e:
                if e.response.status_code == 401:
                    # If we get a 401 error, refresh the tokens and retry
                    self._refresh_tokens()

                # If this was the last attempt, raise the error
                if attempt == max_retries - 1:
                    raise e

    def _refresh_tokens(self, update_local: bool = True):
        with open(self._rayyan_creds_path) as f:
            api_tokens = json.load(f)

        url = "https://rayyan.ai/oauth/token"
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": api_tokens["refresh_token"],
            "client_id": "rayyan.ai",
        }

        response = requests.post(url, data=payload)
        if not response.ok:
            raise Exception(f"Token refresh failed: {response.text}")

        api_tokens_fresh = response.json()

        if update_local:
            with open(self._rayyan_creds_path, "w") as f:
                json.dump(api_tokens_fresh, f, indent=2)
            self.rayyan_instance = Rayyan(self._rayyan_creds_path)
        else:
            temp_creds_path = tempfile.NamedTemporaryFile(
                mode="w+", delete=False, suffix=".json"
            )
            with open(temp_creds_path.name, "w") as f:
                json.dump(api_tokens_fresh, f, indent=2)
            self.rayyan_instance = Rayyan(temp_creds_path.name)

        self.review = Review(self.rayyan_instance)

    @staticmethod
    def extract_article_metadata(rayyan_article: dict) -> dict:
        extracted_info = {
            "Rayyan ID": rayyan_article["id"],
            "Article Title": rayyan_article.get("title", ""),
            "Authors": RayyanManager._join_names(rayyan_article.get("authors", "")),
            "Journal": RayyanManager._extract_journal(
                rayyan_article.get("citation", "")
            ),
            "DOI": rayyan_article.get("doi", ""),
            "Year": rayyan_article.get("year", ""),
        }

        return extracted_info

    @staticmethod
    def _join_names(names: list[str]) -> str:
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        return ", ".join(names[:-1]) + " and " + names[-1]

    @staticmethod
    def _extract_journal(citation_str: str) -> str:
        parts = citation_str.split("-")
        return parts[0].strip() if parts else ""

    @staticmethod
    def _get_fulltext_id(article: dict) -> str | None:
        fulltext_id = None
        for fulltext in article["fulltexts"]:
            if fulltext["marked_as_deleted"]:
                # Skip deleted files
                continue
            fulltext_id = fulltext.get("id", None)
            if fulltext_id:
                break

        return fulltext_id
