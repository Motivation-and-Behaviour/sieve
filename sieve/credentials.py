import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def load_rayyan_credentials() -> str:
    json_path = os.getenv("RAYYAN_JSON_PATH")
    rayyan_creds_json = os.getenv("RAYYAN_CREDS_JSON")

    if json_path:
        json_file = Path(json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"JSON credentials file not found at: {json_path}")

        try:
            with open(json_file) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON credentials file: {e}") from e

        return json_path

    if rayyan_creds_json:
        rayyan_creds_path = "/tmp/rayyan_tokens.json"
        with open(rayyan_creds_path, "w") as f:
            f.write(rayyan_creds_json)
        return rayyan_creds_path

    raise ValueError(
        "Missing required environment variable: RAYYAN_JSON_PATH or RAYYAN_CREDS_JSON"  # noqa: E501
    )


def load_token(token_name: str) -> str:
    token = os.getenv(token_name)
    if not token:
        raise ValueError(f"Missing required environment variable: {token_name}")

    return token
