import json

import pytest

import sieve.credentials as creds

ignore_token = "TESTTOKEN"


def test_load_token_success(monkeypatch):
    # Given an environment variable set
    monkeypatch.setenv("MY_TOKEN", ignore_token)
    # When loading that token
    result = creds.load_token("MY_TOKEN")
    # Then we get its value
    assert result == ignore_token


def test_load_token_missing(monkeypatch):
    # Ensure the variable is not set
    monkeypatch.delenv("NO_SUCH_TOKEN", raising=False)
    # Expect a ValueError mentioning the missing var name
    with pytest.raises(ValueError) as exc:
        creds.load_token("NO_SUCH_TOKEN")
    assert "Missing required environment variable: NO_SUCH_TOKEN" in str(exc.value)


def test_load_rayyan_credentials_missing_env(monkeypatch):
    # Ensure RAYYAN_JSON_PATH is not set
    monkeypatch.delenv("RAYYAN_JSON_PATH", raising=False)
    monkeypatch.delenv("RAYYAN_CREDS_JSON", raising=False)
    # Expect a ValueError
    with pytest.raises(ValueError) as exc:
        creds.load_rayyan_credentials()
    assert (
        "Missing required environment variable: RAYYAN_JSON_PATH or RAYYAN_CREDS_JSON"
        in str(exc.value)
    )


def test_load_rayyan_credentials_not_found(monkeypatch, tmp_path):
    # Point to a non-existent file path
    fake_path = str(tmp_path / "no_file.json")
    monkeypatch.setenv("RAYYAN_JSON_PATH", fake_path)
    with pytest.raises(FileNotFoundError) as exc:
        creds.load_rayyan_credentials()
    assert f"JSON credentials file not found at: {fake_path}" in str(exc.value)


def test_load_rayyan_credentials_invalid_json(monkeypatch, tmp_path):
    # Create a file with invalid JSON
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{invalid-json!}")
    monkeypatch.setenv("RAYYAN_JSON_PATH", str(bad_file))
    # Expect a ValueError wrapping JSONDecodeError
    with pytest.raises(ValueError) as exc:
        creds.load_rayyan_credentials()
    msg = str(exc.value)
    assert "Error parsing JSON credentials file" in msg
    assert (
        "Expecting property name enclosed in double quotes" in msg or "invalid" in msg
    )


def test_load_rayyan_credentials_success(monkeypatch, tmp_path):
    # Create a file with valid JSON
    good_file = tmp_path / "good.json"
    data = {"key": "value"}
    good_file.write_text(json.dumps(data))
    monkeypatch.setenv("RAYYAN_JSON_PATH", str(good_file))
    # Should return the path unchanged
    result = creds.load_rayyan_credentials()
    assert result == str(good_file)


def test_load_rayyan_credentials_from_json_env(monkeypatch):
    # Test loading credentials from RAYYAN_CREDS_JSON env variable
    monkeypatch.delenv("RAYYAN_JSON_PATH", raising=False)
    test_json = '{"key": "value"}'
    monkeypatch.setenv("RAYYAN_CREDS_JSON", test_json)

    result = creds.load_rayyan_credentials()

    # Should create temp file and return its path
    assert result == "/tmp/rayyan_tokens.json"

    # Verify the file was created with correct content
    with open("/tmp/rayyan_tokens.json") as f:
        assert f.read() == test_json
