import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

import sieve.config as config
from sieve.rayyan import RayyanManager


# Sample config values
def test_init_uses_default_path_and_labels(monkeypatch, tmp_path):
    # Create a fake credentials file path in env
    fake_path = tmp_path / "creds.json"
    fake_path.write_text(json.dumps({"refresh_token": "old"}))
    monkeypatch.setenv("RAYYAN_JSON_PATH", str(fake_path))
    # Patch Rayyan and Review to dummy constructors
    DummyRayyan = MagicMock()
    DummyReview = MagicMock()
    monkeypatch.setattr("sieve.rayyan.Rayyan", lambda p: DummyRayyan)
    monkeypatch.setattr("sieve.rayyan.Review", lambda inst: DummyReview)
    # Instantiate without args
    mgr = RayyanManager()
    # Should load creds path from env
    assert mgr._rayyan_creds_path == str(fake_path)
    # Instance attributes
    assert mgr.rayyan_instance is DummyRayyan
    assert mgr.review is DummyReview
    assert mgr.review_id == config.RAYYAN_REVIEW_ID


def test_extract_article_metadata_and_helpers():
    art = {
        "id": 123,
        "title": "Test Article",
        "authors": ["Alice"],
        "citation": "Journal Name - Some other info",
        "doi": "10.1000/xyz",
        "year": "2022",
        "customizations": {"labels": {"SDQ": 1}},
    }
    meta = RayyanManager.extract_article_metadata(art)
    assert meta == {
        "Rayyan ID": 123,
        "Article Title": "Test Article",
        "Authors": "Alice",
        "Journal": "Journal Name",
        "DOI": "10.1000/xyz",
        "Year": "2022",
    }

    # Test join_names multiline
    assert RayyanManager._join_names([]) == ""
    assert RayyanManager._join_names(["A"]) == "A"
    assert RayyanManager._join_names(["A", "B"]) == "A and B"
    assert RayyanManager._join_names(["A", "B", "C"]) == "A, B and C"
    # Test extract_journal edge
    assert RayyanManager._extract_journal("") == ""
    assert RayyanManager._extract_journal("OnlyJournal") == "OnlyJournal"


def test_download_pdf_success_and_no_url(monkeypatch, tmp_path):
    # Create dummy article dict with two fulltexts, one deleted, one valid
    article = {
        "id": 99,
        "fulltexts": [
            {"marked_as_deleted": True, "id": "bad_id"},
            {"marked_as_deleted": False, "id": "good_id"},
        ],
    }

    # Mock the RayyanManager instance and its dependencies
    mock_rayyan_instance = MagicMock()
    mock_rayyan_instance.request.request_handler.return_value = {
        "url": "http://example.com/pdf99"
    }

    # Mock requests.get
    class DummyResponse:
        def __init__(self, content, ok=True):
            self.content = content
            self.ok = ok
            self.status_code = 200

        def raise_for_status(self):
            if not self.ok:
                raise requests.HTTPError()

    dummy = DummyResponse(b"binarypdf")
    monkeypatch.setattr("sieve.rayyan.requests.get", lambda url: dummy)

    # Mock tempfile.mkdtemp to use tmp_path
    monkeypatch.setattr("sieve.rayyan.tempfile.mkdtemp", lambda: str(tmp_path))

    # Create RayyanManager instance with mocked rayyan_instance
    fake_path = tmp_path / "creds.json"
    fake_path.write_text(
        json.dumps({"refresh_token": "old", "access_token": "faketoken"})
    )
    monkeypatch.setenv("RAYYAN_JSON_PATH", str(fake_path))
    manager = RayyanManager()
    manager.rayyan_instance = mock_rayyan_instance

    path = manager.download_pdf(article)

    # Verify API was called with correct fulltext ID
    mock_rayyan_instance.request.request_handler.assert_called_once_with(
        method="GET", path="/api/v1/fulltexts/good_id"
    )

    # File exists and matches id
    assert Path(path).exists()
    assert Path(path).name == "99.pdf"
    assert Path(path).read_bytes() == b"binarypdf"

    # Now test no valid fulltext ID
    bad_article = {"id": 100, "fulltexts": []}
    with pytest.raises(ValueError, match="No fulltext found in the article."):
        manager.download_pdf(bad_article)

    # Test no URL in fulltext details
    mock_rayyan_instance.request.request_handler.return_value = {}
    article_no_url = {
        "id": 101,
        "fulltexts": [{"marked_as_deleted": False, "id": "no_url_id"}],
    }
    with pytest.raises(ValueError, match="No URL found for the fulltext."):
        manager.download_pdf(article_no_url)


@pytest.fixture
def mock_manager(monkeypatch, tmp_path):
    """Create a RayyanManager with mocked external dependencies."""
    fake_path = tmp_path / "creds.json"
    fake_path.write_text(
        json.dumps({"refresh_token": "test", "access_token": "test_token"})
    )
    monkeypatch.setenv("RAYYAN_JSON_PATH", str(fake_path))

    mock_rayyan = MagicMock()
    mock_review = MagicMock()
    mock_notes = MagicMock()
    monkeypatch.setattr("sieve.rayyan.Rayyan", lambda p: mock_rayyan)
    monkeypatch.setattr("sieve.rayyan.Review", lambda inst: mock_review)
    monkeypatch.setattr("sieve.rayyan.Notes", lambda inst: mock_notes)

    manager = RayyanManager()
    manager.rayyan_instance = mock_rayyan
    manager.review = mock_review
    manager.notes_instance = mock_notes
    return manager


class TestGetUnscreenedAbstracts:
    def test_filters_already_labeled_articles(self, mock_manager):
        mock_manager.review.results.side_effect = [
            {"recordsFiltered": 3, "data": []},
            {
                "data": [
                    {"id": 1, "customizations": {"labels": {}}},
                    {
                        "id": 2,
                        "customizations": {
                            "labels": {config.RAYYAN_LABELS["abstract_included"]: 1}
                        },
                    },
                    {"id": 3, "customizations": {"labels": {"SDQ": 1}}},
                ]
            },
        ]

        articles = mock_manager.get_unscreened_abstracts(max_articles=10)

        # Should include id 1 and 3 (priority), exclude id 2 (already labeled)
        assert len(articles) == 2
        assert articles[0]["id"] == 1
        assert articles[1]["id"] == 3

    def test_respects_max_articles(self, mock_manager):
        mock_manager.review.results.side_effect = [
            {"recordsFiltered": 100, "data": []},
            {"data": [{"id": i, "customizations": {"labels": {}}} for i in range(50)]},
        ]

        articles = mock_manager.get_unscreened_abstracts(max_articles=5)
        assert len(articles) == 5


class TestGetUnscreenedFulltexts:
    def test_filters_articles_without_fulltext(self, mock_manager):
        mock_manager.review.results.return_value = {
            "data": [
                {"id": 1, "customizations": {"labels": {}}, "fulltexts": []},
                {
                    "id": 2,
                    "customizations": {"labels": {}},
                    "fulltexts": [{"marked_as_deleted": False, "id": "ft1"}],
                },
            ]
        }

        articles = mock_manager.get_unscreened_fulltexts()

        # Only article 2 has a valid fulltext
        assert len(articles) == 1
        assert articles[0]["id"] == 2

    def test_respects_max_articles(self, mock_manager):
        mock_manager.review.results.return_value = {
            "data": [
                {
                    "id": i,
                    "customizations": {"labels": {}},
                    "fulltexts": [{"marked_as_deleted": False, "id": f"ft{i}"}],
                }
                for i in range(10)
            ]
        }

        articles = mock_manager.get_unscreened_fulltexts(max_articles=3)
        assert len(articles) == 3


class TestGetArticleById:
    def test_returns_article(self, mock_manager):
        mock_manager.review.results.return_value = {
            "data": [{"id": 123, "title": "Test Article"}]
        }

        article = mock_manager.get_article_by_id(123)
        assert article["id"] == 123
        assert article["title"] == "Test Article"

    def test_raises_error_when_not_found(self, mock_manager):
        mock_manager.review.results.return_value = {"data": []}

        with pytest.raises(ValueError, match="Article with ID 999 not found"):
            mock_manager.get_article_by_id(999)


class TestUpdateArticleLabels:
    def test_calls_customize(self, mock_manager):
        plan = {"label1": 1, "label2": -1}
        mock_manager.update_article_labels(123, plan)

        mock_manager.review.customize.assert_called_once_with(
            mock_manager.review_id, 123, plan
        )


class TestCreateArticleNote:
    def test_calls_create_note(self, mock_manager):
        mock_manager.create_article_note(123, "Test note content")

        mock_manager.notes_instance.create_note.assert_called_once_with(
            mock_manager.review_id, 123, "Test note content"
        )


class TestGetFulltextId:
    def test_returns_first_valid_id(self):
        article = {
            "fulltexts": [
                {"marked_as_deleted": True, "id": "deleted_id"},
                {"marked_as_deleted": False, "id": "valid_id"},
                {"marked_as_deleted": False, "id": "another_valid"},
            ]
        }
        assert RayyanManager._get_fulltext_id(article) == "valid_id"

    def test_returns_none_when_no_valid_fulltext(self):
        article = {"fulltexts": [{"marked_as_deleted": True, "id": "deleted"}]}
        assert RayyanManager._get_fulltext_id(article) is None

    def test_returns_none_for_empty_fulltexts(self):
        article = {"fulltexts": []}
        assert RayyanManager._get_fulltext_id(article) is None


class TestRetryOnAuthError:
    def test_retries_on_401(self, mock_manager, monkeypatch):
        call_count = {"count": 0}

        def operation():
            call_count["count"] += 1
            if call_count["count"] < 3:
                response = MagicMock()
                response.status_code = 401
                raise requests.HTTPError(response=response)
            return "success"

        # Mock token refresh
        monkeypatch.setattr(
            "sieve.rayyan.requests.post",
            lambda url, data: MagicMock(
                ok=True, json=lambda: {"refresh_token": "new", "access_token": "new"}
            ),
        )

        result = mock_manager._retry_on_auth_error(operation)
        assert result == "success"
        assert call_count["count"] == 3

    def test_raises_after_max_retries(self, mock_manager, monkeypatch):
        def always_fail():
            response = MagicMock()
            response.status_code = 401
            raise requests.HTTPError(response=response)

        monkeypatch.setattr(
            "sieve.rayyan.requests.post",
            lambda url, data: MagicMock(
                ok=True, json=lambda: {"refresh_token": "new", "access_token": "new"}
            ),
        )

        with pytest.raises(requests.HTTPError):
            mock_manager._retry_on_auth_error(always_fail, max_retries=3)
