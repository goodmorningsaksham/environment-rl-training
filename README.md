👇

🚀 Submission Guide (Final)

This document contains everything required before submitting:

✅ Pre-Submission Checklist
🧠 Inference Script Requirements + Code
🧪 Pre-Validation Checklist + Script
✅ 1. Pre-Submission Checklist

Confirm all before submitting:

I’ve read and followed the sample inference.py strictly
Environment variables are correctly used:
API_BASE_URL
MODEL_NAME
HF_TOKEN
(optional) LOCAL_IMAGE_NAME
Defaults are set only for:
API_BASE_URL
MODEL_NAME
❌ NOT for HF_TOKEN
API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-url>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
All LLM calls use:
from openai import OpenAI
Stdout logs strictly follow:
[START]
[STEP]
[END]
🧠 2. Inference Script
📌 Requirements
File must be named: inference.py
Must be in root directory
Must use OpenAI client
Must follow strict logging format
📤 STDOUT FORMAT (STRICT)
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
Rules:
One [START]
One [STEP] per step
One [END] always
Rewards → 2 decimal
Booleans → lowercase
error=null if none
📜 Inference Script Code

Below is your exact working script:

# Source: User provided script :contentReference[oaicite:0]{index=0}

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from my_env_v4 import MyEnvV4Action, MyEnvV4Env

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASK_NAME = os.getenv("MY_ENV_V4_TASK", "echo")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")

MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1

SYSTEM_PROMPT = """
You are interacting with an echo environment.
Maximize reward by sending meaningful messages.
"""

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}")

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}")

def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}")

def get_model_message(client):
    res = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=MAX_TOKENS,
    )
    return res.choices[0].message.content.strip()

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)

    rewards = []
    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    result = await env.reset()

    for step in range(1, MAX_STEPS + 1):
        msg = get_model_message(client)
        result = await env.step(MyEnvV4Action(message=msg))

        reward = result.reward or 0.0
        rewards.append(reward)

        log_step(step, msg, reward, result.done, None)

        if result.done:
            break

    success = sum(rewards) > 0

    await env.close()
    log_end(success, step, rewards)

if __name__ == "__main__":
    asyncio.run(main())
🧪 3. Pre-Validation Script
📌 Checklist
HF Space is live (/reset works)
Docker builds successfully
openenv validate passes
📜 Validation Script Code
# Source: User provided script :contentReference[oaicite:1]{index=1}

#!/usr/bin/env bash

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    wait "$pid" 2>/dev/null
  fi
}

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  echo "Usage: $0 <ping_url> [repo_dir]"
  exit 1
fi

PING_URL="${PING_URL%/}"

echo "Checking HF Space..."

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 || echo "000")

[ "$HTTP_CODE" = "200" ] || { echo "HF Space failed"; exit 1; }

echo "Docker build..."

docker build "$REPO_DIR" || exit 1

echo "Running openenv validate..."

(cd "$REPO_DIR" && openenv validate) || exit 1

echo "All checks passed!"
🎯 Final Submission Flow
✅ Write inference.py
✅ Set environment variables
✅ Run validation script
✅ Ensure all checks pass
🚀 Submit
🔥 Pro Tip (Important)

Most rejections happen because of:

❌ Wrong stdout format
❌ Missing env variables
❌ Docker build failing
❌ Not using OpenAI client
