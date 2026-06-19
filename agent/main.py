# Copyright 2026
# TensorFlow PR Review Agent

import asyncio
import logging
import time
import requests
from os import environ

from agent import agent
from agent.settings import OWNER, REPO, PULL_REQUEST_NUMBER, GITHUB_BASE_URL
from agent.utils import call_agent_async, parse_number_string

from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner
from google.genai.errors import ServerError

APP_NAME = "tensorflow_pr_review_app"
USER_ID = "tensorflow_pr_review_user"

logs.setup_adk_logger(level=logging.DEBUG)


def get_first_comment_id(pr_number: int) -> int | None:
    """Fetches the ID of the very first issue comment to attach reactions to."""
    token = environ.get("GITHUB_TOKEN")
    if not token or not pr_number:
        return None

    # Pull requests are treated as issues for top-level comment threads
    url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{pr_number}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.json().get("id")
    except Exception as e:
        print(f"Failed to fetch issue metadata: {e}")
    return None


def clear_and_set_reaction(pr_number: int, add_content: str = "eyes"):
    """Cleans up previous runtime reactions and establishes the new active emoji."""
    token = environ.get("GITHUB_TOKEN")
    if not token or not pr_number:
        return

    # Use the specific issue ID endpoint to isolate the root comment reaction block
    url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{pr_number}/reactions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.squirrel-girl-preview+json"
    }

    try:
        # Step 1: Read all existing reactions on this thread
        existing_res = requests.get(url, headers=headers, timeout=10)
        if existing_res.status_code == 200:
            reactions_list = existing_res.json()
            # Loop through and remove any active 'eyes' reactions posted by this agent integration
            for reaction in reactions_list:
                if reaction.get("content") == "eyes":
                    reaction_id = reaction.get("id")
                    delete_url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/reactions/{reaction_id}"
                    requests.delete(delete_url, headers=headers, timeout=10)
                    print(f"Cleared stale 'eyes' reaction ID: {reaction_id}")

        # Step 2: Post the fresh structural reaction status
        requests.post(url, headers=headers, json={"content": add_content}, timeout=10)
        print(f"Successfully posted final state reaction: {add_content}")
    except Exception as e:
        print(f"Failed to balance PR reaction status lifecycle: {e}")


async def main():
    pr_number = parse_number_string(PULL_REQUEST_NUMBER)
    if not pr_number:
        print(f"Error: Invalid pull request number received: {PULL_REQUEST_NUMBER}")
        return

    # 1. Clean old states and put down the looking eyes emoji
    clear_and_set_reaction(pr_number, add_content="eyes")

    prompt = (
        f"Execute your full workflow for pull request #{pr_number}:\n"
        "1. Run the `get_pull_request_details` tool to fetch the diff and content.\n"
        "2. Analyze the code changes using the TensorFlow PR Review Guidelines style guide.\n"
        "3. Generate your architectural feedback review findings.\n"
        "4. Run the `add_comment_to_pr` tool to post your final review feedback comment onto the PR."
    )

    # Reference the custom pool dynamically from agent.py
    for model_name in agent.MODELS_POOL:
        print(f"Attempting execution using engine target: {model_name}...")
        
        # Dynamically overwrite the model attribute on the agent instance
        agent.root_agent.model = model_name

        runner = InMemoryRunner(agent=agent.root_agent, app_name=APP_NAME)
        session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)

        try:
            response = await call_agent_async(runner, USER_ID, session.id, prompt)
            print(f"<<<< Agent Final Output: {response}\n")
            print(f"Processing complete successfully with {model_name}!")
            
            # 2. Execution complete! Swap the eyes emoji out for a final rocket emoji
            clear_and_set_reaction(pr_number, add_content="rocket")
            return 

        except ServerError as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e).upper():
                print(f"⚠️ Target {model_name} is overloaded (503). Attempting alternative fallback...")
                continue 
            else:
                raise e
        except Exception as e:
            raise e

    print("Error: All model targets within the fallback pool returned high demand load spikes.")


if __name__ == "__main__":
    start_time = time.time()
    print(f"Start reviewing {OWNER}/{REPO} pull request #{PULL_REQUEST_NUMBER}")
    print("-" * 80)
    asyncio.run(main())
    print("-" * 80)
