# Copyright 2026
#
# TensorFlow PR Review Agent - Utility Functions

import sys
from typing import Any

from agent.settings import GITHUB_GRAPHQL_URL
from agent.settings import GITHUB_TOKEN
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.genai import types
import requests

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

diff_headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3.diff",
}


def run_graphql_query(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    """Executes a GitHub GraphQL query."""
    payload = {"query": query, "variables": variables}
    response = requests.post(
        GITHUB_GRAPHQL_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def get_request(
    url: str,
    params: dict[str, Any] | None = None,
) -> Any:
    """Executes a GitHub GET request."""
    if params is None:
        params = {}

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def get_diff(url: str) -> str:
    """Retrieves PR diff content."""
    response = requests.get(
        url,
        headers=diff_headers,
        timeout=60,
    )
    response.raise_for_status()
    return response.text


def post_request(url: str, payload: Any) -> dict[str, Any]:
    """Executes a GitHub POST request."""
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def error_response(error_message: str) -> dict[str, Any]:
    """Returns a standardized error response."""
    return {
        "status": "error",
        "error_message": error_message,
    }


def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return ""


def parse_number_string(
    number_str: str | None,
    default_value: int = 0,
) -> int:
    """Parse integer values safely."""
    if not number_str:
        return default_value

    try:
        return int(number_str)
    except ValueError:
        print(
            f"Warning: Invalid number string: {number_str}. "
            f"Defaulting to {default_value}.",
            file=sys.stderr,
        )
        return default_value


async def call_agent_async(
    runner: Runner,
    user_id: str,
    session_id: str,
    prompt: str,
) -> str:
    """Execute an ADK agent asynchronously."""
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)],
    )

    final_response_text = ""

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
        run_config=RunConfig(
            save_input_blobs_as_artifacts=False,
        ),
    ):
        if event.content and event.content.parts:
            text = "".join(
                part.text or ""
                for part in event.content.parts
            )

            if text and event.author != "user":
                final_response_text += text

    return final_response_text
