import os

import pytest

import sieve.credentials as credentials
import sieve.openai as openai


class DummyFile:
    def __init__(self, file_id):
        self.id = file_id


class DummyResponse:
    def __init__(self, parsed):
        self.output_parsed = parsed


@pytest.fixture(autouse=True)
def patch_env_and_token(monkeypatch):
    # Ensure default token used if api_key None
    monkeypatch.setenv("OPENAI_TOKEN", "env_test_key")
    # Monkey-patch load_token to return the env var
    monkeypatch.setattr(credentials, "load_token", lambda name: os.getenv(name))


@pytest.fixture
def dummy_openai(monkeypatch):
    # Replace OpenAI constructor to capture api_key
    created = {}

    class FakeOpenAIClient:
        def __init__(self, api_key):
            created["api_key"] = api_key
            # prepare nested files and responses attrs
            self.files = type(
                "F", (), {"create": lambda self, file, purpose: DummyFile("file123")}
            )()
            self.responses = type(
                "R",
                (),
                {
                    "parse": lambda self, model, input, text_format: DummyResponse(
                        "parsed_output"
                    )
                },
            )()

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAIClient)
    return created


def test_init_with_provided_key(dummy_openai):
    mgr = openai.OpenAIManager(api_key="provided_key", model="gpt-test")
    # Should pass provided_key to underlying client
    assert dummy_openai["api_key"] == "provided_key"
    assert mgr.model == "gpt-test"


def test_init_without_key_uses_load_token(monkeypatch, dummy_openai):
    # Remove direct call, use default None
    mgr = openai.OpenAIManager(api_key=None)  # noqa: F841
    assert dummy_openai["api_key"] == os.getenv("OPENAI_TOKEN")


class DummyBatch:
    """Mock batch object for testing."""

    def __init__(self, batch_id, status="completed", output_file_id=None):
        self.id = batch_id
        self.status = status
        self.output_file_id = output_file_id
        self.error_file_id = None


@pytest.fixture
def mock_openai_manager(monkeypatch):
    """Create an OpenAIManager with a fully mocked client."""
    monkeypatch.setenv("OPENAI_TOKEN", "test_key")

    class MockFiles:
        def create(self, file, purpose):
            return DummyFile("mock_file_id")

    class MockResponses:
        def parse(self, model, input, text_format):
            return DummyResponse("mocked_parse_result")

    class MockBatches:
        def create(self, input_file_id, endpoint, completion_window, metadata):
            return DummyBatch("batch_123")

        def retrieve(self, batch_id):
            return DummyBatch(batch_id, status="completed", output_file_id="output_123")

    class FakeClient:
        def __init__(self, api_key):
            self.files = MockFiles()
            self.responses = MockResponses()
            self.batches = MockBatches()

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    return openai.OpenAIManager(api_key="test_key")


class TestScreenRecordAbstract:
    def test_returns_parsed_output(self, mock_openai_manager):
        result = mock_openai_manager.screen_record_abstract("This is an abstract")
        assert result == "mocked_parse_result"


class TestScreenRecordFulltext:
    def test_returns_parsed_output(self, mock_openai_manager, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy content")

        result = mock_openai_manager.screen_record_fulltext(str(pdf))
        assert result == "mocked_parse_result"


class TestPrepareAbstractBody:
    def test_returns_structured_payload(self, mock_openai_manager):
        body = mock_openai_manager.prepare_abstract_body("Test abstract content")

        assert "model" in body
        assert "input" in body
        assert "text" in body
        assert body["text"]["format"]["type"] == "json_schema"
        assert body["model"] == mock_openai_manager.model


class TestPrepareFulltextBody:
    def test_returns_structured_payload(self, mock_openai_manager):
        body = mock_openai_manager.prepare_fulltext_body("file_123")

        assert "model" in body
        assert "input" in body
        assert "text" in body


class TestParseScreeningDecision:
    def test_parses_valid_json(self, mock_openai_manager):
        json_str = """{
            "vote": "include",
            "matched_inclusion": null,
            "failed_inclusion": null,
            "triggered_exclusion": null,
            "exclusion_reasons": null,
            "rationale": "Meets all criteria"
        }"""
        decision = mock_openai_manager.parse_screening_decision(json_str)

        assert decision.vote == "include"
        assert decision.rationale == "Meets all criteria"

    def test_parses_exclude_decision(self, mock_openai_manager):
        json_str = """{
            "vote": "exclude",
            "matched_inclusion": null,
            "failed_inclusion": null,
            "triggered_exclusion": [1],
            "exclusion_reasons": null,
            "rationale": "Exclusion criterion 1 triggered"
        }"""
        decision = mock_openai_manager.parse_screening_decision(json_str)

        assert decision.vote == "exclude"
        assert decision.triggered_exclusion == [1]


class TestUploadFile:
    def test_returns_file_object(self, mock_openai_manager, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"test content")

        result = mock_openai_manager.upload_file(str(test_file))
        assert result.id == "mock_file_id"


class TestCreateBatch:
    def test_creates_batch_job(self, mock_openai_manager, tmp_path):
        # Create a dummy JSONL file
        jsonl_file = tmp_path / "batch.jsonl"
        jsonl_file.write_text('{"custom_id": "1", "method": "POST"}\n')

        result = mock_openai_manager.create_batch(str(jsonl_file), "abstract_screen")
        assert result.id == "batch_123"


class TestRetrieveBatch:
    def test_retrieves_batch_status(self, mock_openai_manager):
        result = mock_openai_manager.retrieve_batch("batch_456")

        assert result.id == "batch_456"
        assert result.status == "completed"


class TestBuildAbstractPrompt:
    def test_includes_criteria(self, mock_openai_manager):
        messages = mock_openai_manager._build_abstract_prompt("Test abstract")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Test abstract" in messages[1]["content"]
        assert "Inclusion criteria" in messages[0]["content"]
        assert "Exclusion criteria" in messages[0]["content"]


class TestBuildFulltextPrompt:
    def test_includes_file_reference(self, mock_openai_manager):
        messages = mock_openai_manager._build_fulltext_prompt("file_abc")

        # Should have 3 messages (system, user with file, system again)
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # User message should contain file reference
        assert messages[1]["content"][0]["type"] == "input_file"
        assert messages[1]["content"][0]["file_id"] == "file_abc"


class TestNumberCriteria:
    def test_numbers_criteria_list(self):
        criteria = ["First criterion", "Second criterion", "Third criterion"]
        result = openai.OpenAIManager._number_criteria(criteria)

        assert "1. First criterion" in result
        assert "2. Second criterion" in result
        assert "3. Third criterion" in result

    def test_empty_list(self):
        result = openai.OpenAIManager._number_criteria([])
        assert result == ""
