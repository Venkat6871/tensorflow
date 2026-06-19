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

# Centralized Model Pool for 503 Fallbacks
MODELS_POOL = ["gemini-3.1-pro-preview",
    "gemini-3-pro-preview",
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite"]

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
          author { login }
          files(first: 100) {
          nodes {
          path
          }
          }
          comments(last: 50) { nodes { body createdAt author { login } } }
          commits(last: 50) { nodes { commit { url message } } }
        }
      }
    }
    """
    variables = {"owner": OWNER, "repo": REPO, "prNumber": pr_number}
    url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/pulls/{pr_number}"

    try:
        response = run_graphql_query(query, variables)
        if "errors" in response:
            return error_response(str(response["errors"]))
        pr = response.get("data", {}).get("repository", {}).get("pullRequest")
        if not pr:
            return error_response(f"Pull Request #{pr_number} not found.")
        pr["diff"] = get_diff(url)[:30000]
        return {"status": "success", "pull_request": pr}
    except requests.exceptions.RequestException as e:
        return error_response(str(e))


def add_comment_to_pr(pr_number: int, comment: str) -> dict[str, Any]:
    """Post review feedback to the PR."""
    url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{pr_number}/comments"
    payload = {"body": comment}
    try:
        post_request(url, payload)
    except requests.exceptions.RequestException as e:
        return error_response(str(e))
    return {"status": "success", "added_comment": comment}


# Initialize with the first fallback engine in the pool
root_agent = LlmAgent(
    model=MODELS_POOL[0],
    name="tensorflow_pr_review_agent",
    description="Reviews TensorFlow pull requests using style guidelines.",
    instruction=f"""
# Identity

You are an experienced TensorFlow maintainer performing pull request reviews.

# Review Guidelines

{STYLE_GUIDE}

# Required Review Process

1. Call get_pull_request_details.
2. Read the pull request title, description, files and diff.
3. Review ONLY the code that was modified.
4. Base every finding on evidence present in the diff.
5. Apply TensorFlow PR Review Guidelines only when they are relevant to the modified code.
6. Ignore unchanged files.
7. Ignore style guide sections that do not apply to the current change.
8. Prioritize analysis of the modified files and diff over the pull request description.
9. Do not generate findings based solely on the pull request description.

# Critical Rules

- Do not speculate.
- Do not hallucinate findings.
- Do not assume missing tests unless the modified code clearly requires them.
- Do not suggest tf.data.Dataset, callbacks, validation datasets, batching, reproducibility, or training improvements unless the PR actually modifies training code.
- Do not discuss API stability unless public TensorFlow APIs are modified.
- Do not discuss performance unless performance-sensitive code is modified.
- Documentation-only changes should receive documentation-focused review only.
- Configuration-only changes should receive configuration-focused review only.
- README-only changes should not trigger TensorFlow model-training recommendations.
- Every finding must reference information visible in the pull request diff, files, title, or description.
- If evidence cannot be found in the pull request, do not mention it.
- Every finding must include a brief explanation referencing the affected file, code change, or diff section.
- Findings without supporting evidence must not be included.

# Review Quality Bar

If no meaningful issues are identified:

Output:

"No actionable review comments identified. The modified files and diff were reviewed against the TensorFlow PR Review Guidelines and no evidence-based concerns were found."

Do not generate filler recommendations.

Prefer no findings over weak findings.

# Output Format

## Summary

Short summary of the change.

## Findings

Only evidence-based findings.

## Test Coverage

Discuss only if relevant.

## API Stability

Discuss only if relevant.

## Performance & Maintainability

Discuss only if relevant.
""",
    tools=[get_pull_request_details, add_comment_to_pr],
)
