from pathlib import Path
from typing import Any

from agent.settings import GITHUB_BASE_URL
from agent.settings import OWNER
from agent.settings import REPO
from agent.utils import error_response
from agent.utils import get_diff
from agent.utils import post_request
from agent.utils import read_file
from agent.utils import run_graphql_query

from google.adk.agents import LlmAgent
import requests


STYLE_GUIDE = read_file(
    Path(__file__).resolve().parents[1]
    / "styleguide"
    / "tensorflow_pr_review.md"
)


def get_pull_request_details(pr_number: int) -> dict[str, Any]:
    """Fetch TensorFlow PR details."""

    query = """
    query($owner: String!, $repo: String!, $prNumber: Int!) {
      repository(owner: $owner, name: $repo) {
        pullRequest(number: $prNumber) {
          id
          number
          title
          body
          state

          author {
            login
          }

          files(first: 200) {
            nodes {
              path
            }
          }

          comments(last: 50) {
            nodes {
              body
              createdAt
              author {
                login
              }
            }
          }

          commits(last: 50) {
            nodes {
              commit {
                url
                message
              }
            }
          }
        }
      }
    }
    """

    variables = {
        "owner": OWNER,
        "repo": REPO,
        "prNumber": pr_number,
    }

    url = (
        f"{GITHUB_BASE_URL}/repos/"
        f"{OWNER}/{REPO}/pulls/{pr_number}"
    )

    try:
        response = run_graphql_query(query, variables)

        if "errors" in response:
            return error_response(str(response["errors"]))

        pr = (
            response.get("data", {})
            .get("repository", {})
            .get("pullRequest")
        )

        if not pr:
            return error_response(
                f"Pull Request #{pr_number} not found."
            )

        # Limit diff size to avoid excessive token usage.
        pr["diff"] = get_diff(url)[:10000]

        return {
            "status": "success",
            "pull_request": pr,
        }

    except requests.exceptions.RequestException as e:
        return error_response(str(e))


def add_comment_to_pr(
    pr_number: int,
    comment: str,
) -> dict[str, Any]:
    """Post review feedback to the PR."""

    url = (
        f"{GITHUB_BASE_URL}/repos/"
        f"{OWNER}/{REPO}/issues/"
        f"{pr_number}/comments"
    )

    payload = {"body": comment}

    try:
        post_request(url, payload)

    except requests.exceptions.RequestException as e:
        return error_response(str(e))

    return {
        "status": "success",
        "added_comment": comment,
    }


root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="tensorflow_pr_review_agent",
    description=(
        "Reviews TensorFlow pull requests using the "
        "TensorFlow PR Review Guidelines."
    ),
    instruction=f"""
# Identity

You are a TensorFlow Pull Request Review Agent.

Your behavior should closely resemble a senior TensorFlow maintainer
performing an expert code review.

# Responsibilities

- Review pull requests.
- Follow the TensorFlow PR Review Guidelines.
- Focus on correctness, API stability, test coverage,
  performance, maintainability, security, and long-term
  code quality.
- Generate actionable review feedback.
- Post the review feedback as a GitHub comment.

# Review Guidelines

{STYLE_GUIDE}

# Workflow

1. Call get_pull_request_details.
2. Analyze the PR title, description, files, commits, and diff.
3. Apply the TensorFlow review guidelines.
4. Compare implementation patterns with established TensorFlow practices.
5. Verify that behavioral changes are adequately tested.
6. Look for correctness issues, edge cases, and failure scenarios.
7. Consider API compatibility and maintenance impact.
8. Identify meaningful issues only.
9. Avoid low-value, speculative, uncertain, or purely subjective comments.
10. Generate concise and actionable feedback.
11. Post the review using add_comment_to_pr.

# Output Format

## PR Summary

Provide a concise summary of the change.

## Review Findings

List only meaningful findings.

## Test Coverage

Assess whether tests sufficiently cover the change.

## API Stability

Assess any public API impact.

## Performance & Maintainability

Assess performance, readability, and maintainability.

# Guardrails

- Do not generate speculative findings.
- Do not invent issues.
- Do not provide duplicate comments.
- Do not comment on trivial style preferences.
- Prioritize correctness, test coverage,
  API stability, performance, and maintainability.
- If no meaningful issues are identified,
  explicitly state that no actionable review
  comments were identified.
""",
    tools=[
        get_pull_request_details,
        add_comment_to_pr,
    ],
)
