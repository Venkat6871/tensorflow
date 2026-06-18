# Copyright 2026

#

# TensorFlow PR Review Agent

import asyncio
import logging
import time

from agent import agent
from agent.settings import OWNER
from agent.settings import PULL_REQUEST_NUMBER
from agent.settings import REPO
from agent.utils import call_agent_async
from agent.utils import parse_number_string

from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner

APP_NAME = "tensorflow_pr_review_app"
USER_ID = "tensorflow_pr_review_user"

logs.setup_adk_logger(level=logging.DEBUG)

async def main():
runner = InMemoryRunner(
agent=agent.root_agent,
app_name=APP_NAME,
)

```
session = await runner.session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
)

pr_number = parse_number_string(PULL_REQUEST_NUMBER)

if not pr_number:
    print(
        f"Error: Invalid pull request number received: "
        f"{PULL_REQUEST_NUMBER}."
    )
    return

prompt = (
    f"Review pull request #{pr_number} using the "
    "TensorFlow PR Review Guidelines. "
    "Analyze the pull request details, evaluate correctness, "
    "test coverage, API stability, performance, maintainability, "
    "and security considerations. "
    "Generate actionable review feedback and post the review "
    "as a GitHub comment."
)

response = await call_agent_async(
    runner,
    USER_ID,
    session.id,
    prompt,
)

print(f"<<<< Agent Final Output: {response}\n")
```

if **name** == "**main**":
start_time = time.time()

```
print(
    f"Start reviewing {OWNER}/{REPO} "
    f"pull request #{PULL_REQUEST_NUMBER} at "
    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_time))}"
)

print("-" * 80)

asyncio.run(main())

print("-" * 80)

end_time = time.time()

print(
    "Review finished at "
    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(end_time))}"
)

print(
    "Total script execution time:",
    f"{end_time - start_time:.2f} seconds",
)
```
