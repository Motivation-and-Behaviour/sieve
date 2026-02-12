"""Tests for BatchTracker class."""

import json
from datetime import datetime

import pytest

from sieve.batchtracker import BatchTracker


@pytest.fixture
def tracker(tmp_path):
    """Create a BatchTracker with a temporary file."""
    filepath = tmp_path / "batches.json"
    return BatchTracker(filepath=str(filepath))


class TestBatchTrackerInit:
    def test_creates_empty_file_if_not_exists(self, tmp_path):
        filepath = tmp_path / "new_batches.json"
        assert not filepath.exists()

        BatchTracker(filepath=str(filepath))

        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
        assert data == {}

    def test_uses_existing_file(self, tmp_path):
        filepath = tmp_path / "existing.json"
        existing_data = {"batch_123": {"type": "abstract", "status": "completed"}}
        filepath.write_text(json.dumps(existing_data))

        tracker = BatchTracker(filepath=str(filepath))

        # Verify data is preserved
        data = tracker._load()
        assert data == existing_data


class TestAddBatch:
    def test_add_batch_creates_entry(self, tracker):
        tracker.add_batch("batch_abc", "abstract")

        data = tracker._load()
        assert "batch_abc" in data
        assert data["batch_abc"]["type"] == "abstract"
        assert data["batch_abc"]["status"] == "in_progress"
        assert "created_at" in data["batch_abc"]

    def test_add_batch_with_different_types(self, tracker):
        tracker.add_batch("batch_1", "abstract")
        tracker.add_batch("batch_2", "fulltext")
        tracker.add_batch("batch_3", "extraction")

        data = tracker._load()
        assert data["batch_1"]["type"] == "abstract"
        assert data["batch_2"]["type"] == "fulltext"
        assert data["batch_3"]["type"] == "extraction"

    def test_add_batch_stores_valid_iso_timestamp(self, tracker):
        tracker.add_batch("batch_time", "abstract")

        data = tracker._load()
        created_at = data["batch_time"]["created_at"]
        # Should be parseable as ISO format
        parsed = datetime.fromisoformat(created_at)
        assert isinstance(parsed, datetime)

    def test_add_batch_overwrites_existing(self, tracker):
        tracker.add_batch("batch_dup", "abstract")
        tracker.add_batch("batch_dup", "fulltext")

        data = tracker._load()
        assert data["batch_dup"]["type"] == "fulltext"


class TestGetPendingBatches:
    def test_returns_empty_dict_when_no_batches(self, tracker):
        pending = tracker.get_pending_batches()
        assert pending == {}

    def test_returns_in_progress_batches(self, tracker):
        tracker.add_batch("batch_1", "abstract")
        tracker.add_batch("batch_2", "fulltext")

        pending = tracker.get_pending_batches()
        assert len(pending) == 2
        assert "batch_1" in pending
        assert "batch_2" in pending

    def test_excludes_completed_batches(self, tracker):
        tracker.add_batch("batch_1", "abstract")
        tracker.add_batch("batch_2", "fulltext")
        tracker.mark_completed("batch_1")

        pending = tracker.get_pending_batches()
        assert len(pending) == 1
        assert "batch_1" not in pending
        assert "batch_2" in pending

    def test_returns_empty_when_all_completed(self, tracker):
        tracker.add_batch("batch_1", "abstract")
        tracker.mark_completed("batch_1")

        pending = tracker.get_pending_batches()
        assert pending == {}


class TestMarkCompleted:
    def test_marks_batch_as_completed(self, tracker):
        tracker.add_batch("batch_1", "abstract")
        tracker.mark_completed("batch_1")

        data = tracker._load()
        assert data["batch_1"]["status"] == "completed"

    def test_mark_nonexistent_batch_does_nothing(self, tracker):
        # Should not raise an error
        tracker.mark_completed("nonexistent")

        data = tracker._load()
        assert "nonexistent" not in data

    def test_preserves_other_fields_on_completion(self, tracker):
        tracker.add_batch("batch_1", "extraction")
        original_data = tracker._load()
        original_created = original_data["batch_1"]["created_at"]
        original_type = original_data["batch_1"]["type"]

        tracker.mark_completed("batch_1")

        data = tracker._load()
        assert data["batch_1"]["created_at"] == original_created
        assert data["batch_1"]["type"] == original_type
        assert data["batch_1"]["status"] == "completed"


class TestLoadSave:
    def test_load_returns_dict(self, tracker):
        data = tracker._load()
        assert isinstance(data, dict)

    def test_save_persists_data(self, tracker):
        test_data = {"test_key": {"value": 123}}
        tracker._save(test_data)

        loaded = tracker._load()
        assert loaded == test_data
