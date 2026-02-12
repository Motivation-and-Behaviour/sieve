"""Tests for IntegrationManager class."""

from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

import sieve.config as config
from sieve.datamodels import ScreeningDecision
from sieve.integration import IntegrationManager, requires_services


class TestRequiresServicesDecorator:
    def test_allows_when_all_services_present(self):
        class MockManager:
            asana = MagicMock()
            airtable = MagicMock()

            @requires_services("asana", "airtable")
            def method(self):
                return "success"

        mgr = MockManager()
        assert mgr.method() == "success"

    def test_raises_when_service_missing(self):
        class MockManager:
            asana = MagicMock()
            airtable = None

            @requires_services("asana", "airtable")
            def method(self):
                return "success"

        mgr = MockManager()
        with pytest.raises(RuntimeError, match="Missing required services: airtable"):
            mgr.method()

    def test_raises_with_multiple_missing_services(self):
        class MockManager:
            asana = None
            airtable = None
            rayyan = MagicMock()

            @requires_services("asana", "airtable", "rayyan")
            def method(self):
                return "success"

        mgr = MockManager()
        with pytest.raises(RuntimeError, match="asana"):
            mgr.method()


class TestIntegrationManagerInit:
    def test_init_with_no_services(self):
        mgr = IntegrationManager()
        assert mgr.rayyan is None
        assert mgr.openai is None
        assert mgr.tracker is None
        assert mgr.debug is False

    def test_init_with_all_services(self):
        rayyan = MagicMock()
        openai = MagicMock()
        tracker = MagicMock()
        console = Console()

        mgr = IntegrationManager(
            rayyan_manager=rayyan,
            openai_manager=openai,
            batch_tracker=tracker,
            console=console,
            debug=True,
        )

        assert mgr.rayyan is rayyan
        assert mgr.openai is openai
        assert mgr.tracker is tracker
        assert mgr.console is console
        assert mgr.debug is True


@pytest.fixture
def mock_rayyan():
    rayyan = MagicMock()
    rayyan.unextracted_label = "Unextracted"
    rayyan.extracted_label = "Extracted"
    return rayyan


@pytest.fixture
def mock_openai():
    openai = MagicMock()
    return openai


@pytest.fixture
def mock_tracker():
    tracker = MagicMock()
    return tracker


@pytest.fixture
def integration_manager(mock_rayyan, mock_openai, mock_tracker):
    return IntegrationManager(
        rayyan_manager=mock_rayyan,
        openai_manager=mock_openai,
        batch_tracker=mock_tracker,
        debug=True,
    )


class TestScreenAbstract:
    def test_screens_and_actions_decision(
        self, integration_manager, mock_openai, mock_rayyan
    ):
        article = {"id": 123, "abstracts": [{"content": "This is the abstract text"}]}

        decision = ScreeningDecision(
            vote="include",
            matched_inclusion=[1, 2],
            failed_inclusion=None,
            triggered_exclusion=None,
            exclusion_reasons=None,
            rationale="Meets criteria",
        )
        mock_openai.screen_record_abstract.return_value = decision

        integration_manager.screen_abstract(article)

        mock_openai.screen_record_abstract.assert_called_once_with(
            "This is the abstract text"
        )
        mock_rayyan.update_article_labels.assert_called_once()

    def test_skips_when_no_abstract(self, integration_manager, mock_openai):
        article = {"id": 123, "abstracts": []}

        integration_manager.screen_abstract(article)

        mock_openai.screen_record_abstract.assert_not_called()


class TestScreenFulltext:
    def test_screens_fulltext(
        self, integration_manager, mock_openai, mock_rayyan, tmp_path
    ):
        pdf_path = str(tmp_path / "article.pdf")
        article = {"id": 123}

        mock_rayyan.download_pdf.return_value = pdf_path
        decision = ScreeningDecision(
            vote="exclude",
            matched_inclusion=None,
            failed_inclusion=None,
            triggered_exclusion=[1],
            exclusion_reasons=None,
            rationale="Wrong population",
        )
        mock_openai.screen_record_fulltext.return_value = decision

        integration_manager.screen_fulltext(article)

        mock_rayyan.download_pdf.assert_called_once_with(article)
        mock_openai.screen_record_fulltext.assert_called_once_with(pdf_path)

    def test_skips_when_no_pdf(self, integration_manager, mock_rayyan, mock_openai):
        article = {"id": 123}
        mock_rayyan.download_pdf.return_value = None

        integration_manager.screen_fulltext(article)

        mock_openai.screen_record_fulltext.assert_not_called()


class TestActionScreeningDecision:
    def test_include_abstract_decision(self, integration_manager, mock_rayyan):
        decision = {"vote": "include", "rationale": "Meets criteria"}

        integration_manager._action_screening_decision(
            decision, article_id=123, is_abstract=True
        )

        mock_rayyan.update_article_labels.assert_called_once()
        call_args = mock_rayyan.update_article_labels.call_args
        assert config.RAYYAN_LABELS["abstract_included"] in call_args[0][1]

    def test_exclude_fulltext_decision_with_reason(
        self, integration_manager, mock_rayyan
    ):
        decision = {
            "vote": "exclude",
            "triggered_exclusion": [1],
            "rationale": "Wrong population",
        }

        integration_manager._action_screening_decision(
            decision, article_id=123, is_abstract=False
        )

        mock_rayyan.update_article_labels.assert_called()
        mock_rayyan.create_article_note.assert_called_once()

    def test_invalid_vote_skipped(self, integration_manager, mock_rayyan):
        decision = {"vote": "maybe", "rationale": "Unsure"}

        integration_manager._action_screening_decision(
            decision, article_id=123, is_abstract=True
        )

        mock_rayyan.update_article_labels.assert_not_called()

    def test_batch_removes_pending_label(self, integration_manager, mock_rayyan):
        decision = {"vote": "include", "rationale": "Good"}

        integration_manager._action_screening_decision(
            decision, article_id=123, is_abstract=True, is_batch=True
        )

        call_args = mock_rayyan.update_article_labels.call_args
        plan = call_args[0][1]
        assert config.RAYYAN_LABELS["batch_pending"] in plan
        assert plan[config.RAYYAN_LABELS["batch_pending"]] == -1


class TestCreateAbstractScreeningBatch:
    def test_prepares_and_submits_batch(
        self, integration_manager, mock_openai, mock_rayyan, mock_tracker, tmp_path
    ):
        articles = [
            {"id": 1, "abstracts": [{"content": "Abstract 1"}]},
            {"id": 2, "abstracts": [{"content": "Abstract 2"}]},
            {"id": 3, "abstracts": []},  # Should be skipped
        ]

        mock_openai.prepare_abstract_body.return_value = {"model": "test", "input": []}

        with patch.object(integration_manager, "_submit_batch") as mock_submit:
            integration_manager.create_abstract_screening_batch(articles)

        # Should have prepared 2 requests (article 3 has no abstract)
        assert mock_openai.prepare_abstract_body.call_count == 2
        mock_submit.assert_called_once()


class TestCreateFulltextScreeningBatch:
    def test_prepares_and_submits_batch(
        self, integration_manager, mock_openai, mock_rayyan, mock_tracker, tmp_path
    ):
        articles = [{"id": 1}, {"id": 2}]

        mock_rayyan.download_pdf.return_value = "/path/to/pdf"
        mock_file = MagicMock()
        mock_file.id = "file_123"
        mock_openai.upload_file.return_value = mock_file
        mock_openai.prepare_fulltext_body.return_value = {
            "model": "test",
            "messages": [],
        }

        with patch.object(integration_manager, "_submit_batch") as mock_submit:
            integration_manager.create_fulltext_screening_batch(articles)

        assert mock_openai.upload_file.call_count == 2
        mock_submit.assert_called_once()

    def test_skips_articles_without_pdf(
        self, integration_manager, mock_openai, mock_rayyan
    ):
        articles = [{"id": 1}]
        mock_rayyan.download_pdf.return_value = None

        with patch.object(integration_manager, "_submit_batch") as mock_submit:
            integration_manager.create_fulltext_screening_batch(articles)

        mock_openai.upload_file.assert_not_called()
        mock_submit.assert_not_called()


class TestSubmitBatch:
    def test_writes_jsonl_and_creates_batch(
        self, integration_manager, mock_openai, mock_tracker, tmp_path, monkeypatch
    ):
        # Change to tmp_path for file operations
        monkeypatch.chdir(tmp_path)

        requests = [
            {"custom_id": "abstract-1", "method": "POST", "body": {}},
            {"custom_id": "abstract-2", "method": "POST", "body": {}},
        ]

        mock_batch = MagicMock()
        mock_batch.id = "batch_123"
        mock_openai.create_batch.return_value = mock_batch

        integration_manager._submit_batch(requests, "abstract_screen")

        mock_openai.create_batch.assert_called_once()
        mock_tracker.add_batch.assert_called_once_with("batch_123", "abstract_screen")


class TestProcessPendingBatches:
    def test_processes_completed_batch(self, integration_manager, mock_openai):
        pending = {"batch_1": {"type": "abstract_screen"}}

        mock_batch = MagicMock()
        mock_batch.status = "completed"
        mock_batch.output_file_id = "output_123"
        mock_openai.retrieve_batch.return_value = mock_batch

        with patch.object(
            integration_manager, "_handle_completed_batch"
        ) as mock_handle:
            integration_manager.process_pending_batches(pending)

        mock_handle.assert_called_once_with("output_123", "abstract_screen", "batch_1")

    def test_handles_failed_batch(self, integration_manager, mock_openai):
        pending = {"batch_1": {"type": "abstract_screen"}}

        mock_batch = MagicMock()
        mock_batch.status = "failed"
        mock_openai.retrieve_batch.return_value = mock_batch

        # Should not raise, just log
        integration_manager.process_pending_batches(pending)

    def test_handles_in_progress_batch(self, integration_manager, mock_openai):
        pending = {"batch_1": {"type": "abstract_screen"}}

        mock_batch = MagicMock()
        mock_batch.status = "in_progress"
        mock_openai.retrieve_batch.return_value = mock_batch

        # Should not call handle
        with patch.object(
            integration_manager, "_handle_completed_batch"
        ) as mock_handle:
            integration_manager.process_pending_batches(pending)

        mock_handle.assert_not_called()


class TestHandleCompletedBatch:
    def test_processes_abstract_results(
        self, integration_manager, mock_openai, mock_tracker
    ):
        output_file_id = "output_123"
        batch_type = "abstract_screen"
        batch_id = "batch_1"

        # Mock file content response
        mock_content = MagicMock()
        mock_content.text = '{"custom_id": "abstract-123", "response": {"status_code": 200, "body": {"choices": [{"message": {"content": "{\\"vote\\": \\"include\\", \\"rationale\\": \\"Good\\"}"}}]}}}'  # noqa: E501
        mock_openai.client.files.content.return_value = mock_content

        decision = ScreeningDecision(
            vote="include",
            matched_inclusion=None,
            failed_inclusion=None,
            triggered_exclusion=None,
            exclusion_reasons=None,
            rationale="Good",
        )
        mock_openai.parse_screening_decision.return_value = decision

        with patch.object(integration_manager, "_action_screening_decision"):
            integration_manager._handle_completed_batch(
                output_file_id, batch_type, batch_id
            )

        mock_tracker.mark_completed.assert_called_once_with(batch_id)


class TestLog:
    def test_logs_when_debug_enabled(self, integration_manager):
        integration_manager.debug = True
        integration_manager.console = MagicMock()

        integration_manager._log("Test message")

        integration_manager.console.log.assert_called_once_with("Test message")

    def test_does_not_log_when_debug_disabled(self, integration_manager):
        integration_manager.debug = False
        integration_manager.console = MagicMock()

        integration_manager._log("Test message")

        integration_manager.console.log.assert_not_called()


class TestProcessAbstractResults:
    def test_processes_successful_results(
        self, integration_manager, mock_openai, mock_rayyan
    ):
        results = [
            {
                "custom_id": "abstract-123",
                "response": {
                    "status_code": 200,
                    "body": {
                        "output": [
                            {
                                "content": [
                                    {"text": '{"vote": "include", "rationale": "Good"}'}
                                ]
                            }
                        ]
                    },
                },
            }
        ]

        decision = ScreeningDecision(
            vote="include",
            matched_inclusion=None,
            failed_inclusion=None,
            triggered_exclusion=None,
            exclusion_reasons=None,
            rationale="Good",
        )
        mock_openai.parse_screening_decision.return_value = decision

        integration_manager._process_abstract_results(results)

        mock_rayyan.update_article_labels.assert_called_once()

    def test_handles_error_results(self, integration_manager, mock_openai, mock_rayyan):
        results = [
            {
                "custom_id": "abstract-123",
                "response": {
                    "status_code": 500,
                    "body": {"error": "Internal error"},
                },
            }
        ]

        integration_manager._process_abstract_results(results)

        # Should update labels to remove batch_pending
        mock_rayyan.update_article_labels.assert_called_once()

    def test_handles_parsing_errors(
        self, integration_manager, mock_openai, mock_rayyan
    ):
        results = [
            {
                "custom_id": "abstract-123",
                "response": {
                    "status_code": 200,
                    "body": {"output": [{"content": [{"text": "invalid json"}]}]},
                },
            }
        ]

        mock_openai.parse_screening_decision.side_effect = Exception("Parse error")

        # Should not raise, just log
        integration_manager._process_abstract_results(results)


class TestProcessFulltextResults:
    def test_processes_successful_fulltext_results(
        self, integration_manager, mock_openai, mock_rayyan
    ):
        results = [
            {
                "custom_id": "fulltext-456",
                "response": {
                    "status_code": 200,
                    "body": {
                        "output": [
                            {
                                "content": [
                                    {
                                        "text": (
                                            '{"vote": "exclude", '
                                            '"triggered_exclusion": [1], '
                                            '"rationale": "Wrong"}'
                                        )
                                    }
                                ]
                            }
                        ]
                    },
                },
            }
        ]

        decision = ScreeningDecision(
            vote="exclude",
            matched_inclusion=None,
            failed_inclusion=None,
            triggered_exclusion=[1],
            exclusion_reasons=None,
            rationale="Wrong",
        )
        mock_openai.parse_screening_decision.return_value = decision

        integration_manager._process_fulltext_results(results)

        mock_rayyan.update_article_labels.assert_called_once()


class TestActionScreeningDecisionEdgeCases:
    def test_exclude_with_failed_inclusion(self, integration_manager, mock_rayyan):
        decision = {
            "vote": "exclude",
            "failed_inclusion": [2],
            "rationale": "Failed inclusion criteria",
        }

        integration_manager._action_screening_decision(
            decision, article_id=123, is_abstract=False
        )

        mock_rayyan.create_article_note.assert_called_once()
        mock_rayyan.update_article_labels.assert_called_once()

    def test_long_rationale_truncated(self, integration_manager, mock_rayyan):
        decision = {
            "vote": "exclude",
            "triggered_exclusion": [1],
            "rationale": "x" * 1500,  # Very long rationale
        }

        integration_manager._action_screening_decision(
            decision, article_id=123, is_abstract=False
        )

        # Should truncate to ~1000 chars
        call_args = mock_rayyan.create_article_note.call_args
        note_text = call_args[0][1]
        assert len(note_text) <= 1020
        assert "..." in note_text

    def test_note_creation_failure(self, integration_manager, mock_rayyan):
        decision = {
            "vote": "exclude",
            "triggered_exclusion": [1],
            "rationale": "Excluded",
        }

        mock_rayyan.create_article_note.side_effect = Exception("Note failed")

        # Should not raise
        integration_manager._action_screening_decision(
            decision, article_id=123, is_abstract=False
        )

        mock_rayyan.update_article_labels.assert_called_once()

    def test_include_fulltext_adds_unextracted(self, integration_manager, mock_rayyan):
        decision = {
            "vote": "include",
            "rationale": "Good",
        }

        integration_manager._action_screening_decision(
            decision, article_id=123, is_abstract=False
        )

        call_args = mock_rayyan.update_article_labels.call_args
        plan = call_args[0][1]
        assert config.RAYYAN_LABELS["included"] in plan


class TestScreenAbstractEdgeCases:
    def test_skips_empty_abstract_content(self, integration_manager, mock_openai):
        article = {"id": 123, "abstracts": [{"content": ""}]}

        integration_manager.screen_abstract(article)

        mock_openai.screen_record_abstract.assert_not_called()

    def test_handles_llm_failure(self, integration_manager, mock_openai, mock_rayyan):
        article = {"id": 123, "abstracts": [{"content": "Test abstract"}]}
        mock_openai.screen_record_abstract.return_value = None

        integration_manager.screen_abstract(article)

        mock_rayyan.update_article_labels.assert_not_called()


class TestScreenFulltextEdgeCases:
    def test_handles_llm_failure_fulltext(
        self, integration_manager, mock_openai, mock_rayyan
    ):
        article = {"id": 123}
        mock_rayyan.download_pdf.return_value = "/path/to/pdf"
        mock_openai.screen_record_fulltext.return_value = None

        integration_manager.screen_fulltext(article)

        mock_rayyan.update_article_labels.assert_not_called()


class TestCreateAbstractBatchEdgeCases:
    def test_marks_missing_abstracts(
        self, integration_manager, mock_openai, mock_rayyan
    ):
        articles = [
            {"id": 1, "abstracts": []},
            {"id": 2, "abstracts": [{"content": ""}]},
        ]

        with patch.object(integration_manager, "_submit_batch") as mock_submit:
            integration_manager.create_abstract_screening_batch(articles)

        # Should update labels for both articles with missing abstract
        assert mock_rayyan.update_article_labels.call_count == 2
        mock_submit.assert_not_called()

    def test_submits_batch_when_no_requests(self, integration_manager):
        articles = []

        with patch.object(integration_manager, "_submit_batch") as mock_submit:
            integration_manager.create_abstract_screening_batch(articles)

        mock_submit.assert_not_called()
