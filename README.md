# MASTER BUILD PROMPT — GlucoRL OpenEnv Environment
# For Claude Code / Claude Opus 4.6
# Attach the Kube SRE Gym codebase zip alongside this prompt

---
OPERATING INSTRUCTIONS — READ BEFORE DOING ANYTHING ELSE

You are about to build a complete, production-ready project.
This is not a quick task. Take as long as needed.

STRICT RULES you must follow throughout this entire session:

1. NEVER summarise code with comments like "# ... rest of implementation"
   or "# similar to above" or "# add error handling here".
   Write every single line of every file completely. No placeholders.

2. NEVER skip a file. Every file listed in Section 4 must be fully
   written out, not described.

3. NEVER move to the next file until the current file is 100% complete
   and you have shown the test command and expected output for it.

4. If your response is getting long, do NOT compress or rush the
   remaining content. Instead, stop at a clean boundary (end of a
   complete file), write "CHECKPOINT — respond with 'continue' to
   proceed to the next file", and wait.

5. When I say 'continue', pick up exactly where you left off.
   Start your response with "Continuing from [filename]..." so I
   know you have not lost context.

6. After every file, explicitly state:
   "File [name] complete. Test with: [exact command]"

7. If you are ever unsure about something — simglucose API,
   OpenEnv spec detail, Pydantic version — STOP and ask me.
   Do not assume and implement something wrong.

8. Keep track of what has been built. At each checkpoint, output
   a short checklist showing which files are done and which remain.

9. Do not write README.md until ALL code files are complete and tested.
   The README needs real baseline scores which don't exist yet.

10. Your goal is a project that runs perfectly, not a project
    that looks complete but has hidden bugs.

HARD STOP AFTER READING THIS PREAMBLE:

Do NOT start building. Do NOT write any code. Do NOT analyse the
codebase yet.

After reading everything I send you in this message, your ONLY
response should be exactly this:

"I have read the preamble, the full GlucoRL build prompt, and the
Kube SRE Gym codebase. I am ready to begin. Awaiting your first
instruction."

Nothing else. No summaries. No "here's my plan". No "I'll start with".
Just that one sentence. Wait for me to give you the first instruction.

---

## YOUR ROLE
You are an expert Python engineer building a complete, production-ready OpenEnv
reinforcement learning environment called **GlucoRL** for a hackathon. You have
been given the complete codebase of **Kube SRE Gym** — a prior winning submission
to the same hackathon — as a reference. Your job is to:

1. Study the Kube SRE Gym codebase carefully to understand the exact OpenEnv
   spec, FastAPI server structure, Pydantic models, Dockerfile pattern, and
   openenv.yaml format.
2. Build GlucoRL from scratch using those patterns — but with a completely
   different domain (insulin dosing for diabetic patients), a different
   simulator (simglucose), and no Kubernetes infrastructure whatsoever.
3. Ensure every item on the hackathon Round 1 checklist is satisfied.

Do NOT copy Kubernetes-specific code. Do NOT include a curriculum controller,
adversarial designer, GKE backend, or LLM judge. Those are Kube SRE Gym-specific
components that do not belong in GlucoRL. What you DO copy is the structural
pattern: the OpenEnv FastAPI server, Pydantic model layout, openenv.yaml format,
Dockerfile base image, and client/inference.py pattern.

---

## SECTION 1 — STUDY THE REFERENCE CODEBASE FIRST

Before writing a single line of GlucoRL code, read and understand these files
from the attached Kube SRE Gym zip:

### Files to read and understand deeply:
- `models.py` — how Action, Observation, State, and Reward are typed as Pydantic models
- `server/app.py` — the FastAPI server with /reset, /step, /state endpoints and WebSocket
- `server/kube_sre_gym_environment.py` — the core reset()/step()/state() implementation pattern
- `openenv.yaml` — the exact spec format required by `openenv validate`
- `Dockerfile` — the openenv-base image, uvicorn startup, port 8000
- `client.py` — the sync client that wraps the HTTP API
- `server/judge.py` — how a grader returns a 0.0–1.0 score with partial credit

### What to extract from your reading:
- The exact fields in openenv.yaml and what they mean
- How Pydantic models are structured for the OpenEnv spec (what base classes, if any)
- How reset() initialises and returns an observation
- How step(action) processes the action, advances state, computes reward, and returns (observation, reward, done, info)
- How state() returns the current full environment state
- How the graders score partial progress rather than binary pass/fail
- How the Dockerfile uses the openenv-base image and starts the server
- How the client wraps reset() and step() for use in inference.py

Only after reading and confirming you understand all of the above should you
begin building GlucoRL.

---

## SECTION 2 — GLUCORL PROJECT DESCRIPTION

### What is GlucoRL?

GlucoRL is an OpenEnv reinforcement learning environment that trains AI agents
to make insulin dosing decisions for Type 1 Diabetic (T1D) patients. It simulates
a continuous glucose monitoring (CGM) system where an agent observes blood glucose
readings and decides how much insulin to deliver every 3 minutes over a simulated day.

### Why does this matter?

Type 1 Diabetes affects over 9 million people worldwide. These patients require
continuous insulin management — too little insulin and glucose spikes dangerously
high (hyperglycemia, causes organ damage over time). Too much insulin and glucose
crashes dangerously low (hypoglycemia, can cause seizures and death within minutes).

Current clinical systems use PID (Proportional-Integral-Derivative) controllers
or Model Predictive Control — rule-based systems that cannot adapt to individual
patient variability, meal timing, or exercise. RL agents have the potential to
learn personalised policies that outperform static controllers.

GlucoRL provides the training environment to develop and evaluate such agents,
benchmarked against a PID baseline, using the medically-validated simglucose
simulator (based on the FDA-accepted UVa/Padova T1D metabolic simulator).

### The clinical framing to use in README and comments:
- Time-in-Range (TIR) is the clinical gold standard metric: % of time glucose
  stays in the 70–180 mg/dL target range. Clinical target is ≥70% TIR.
- Hypoglycemia: glucose < 70 mg/dL (dangerous), < 54 mg/dL (severe/life-threatening)
- Hyperglycemia: glucose > 180 mg/dL (causes damage over time), > 250 mg/dL (severe)
- Hypo events are 2–4x more dangerous than hyper events in the short term
- One episode = one simulated day (24 hours = 480 steps at 3-minute intervals)

---

## SECTION 3 — TECHNICAL ARCHITECTURE

### Simulator: simglucose

Use the `simglucose` Python library. Install with: `pip install simglucose`

Key simglucose concepts:
- `T1DPatient.withName(name)` — creates a patient from the built-in pool
- Available patients: 'adolescent#001' through 'adolescent#010',
  'adult#001' through 'adult#010', 'child#001' through 'child#010' (30 total)
- `patient.observation` — returns namedtuple with `CGM` field (glucose in mg/dL)
- `patient.step(action)` — action is a namedtuple `Action(basal=X, bolus=Y)`
  where X and Y are floats in units/min
- Meal events: use `simglucose.envs.simglucose_gym_env` or inject meals manually
  via the patient step. Meals are specified in grams of CHO (carbohydrates).

The step interval in simglucose is 3 minutes by default. One day = 480 steps.
For our environment, we use 3-minute intervals throughout.

### Important simglucose usage note:
simglucose actions use units/min. Our GlucoAction uses units/hr for basal and
total units for bolus. Convert: `basal_umin = basal_rate_uhr / 60.0` and
`bolus_umin = bolus_dose / 3.0` (spread over one 3-minute step).

### Meal schedule for Task 2 and Task 3:
Use a fixed meal schedule per episode:
- Breakfast: step 100 (5 hours in), 50g CHO
- Lunch: step 200 (10 hours in), 70g CHO  
- Dinner: step 320 (16 hours in), 80g CHO

In Task 2, announce meals 10 steps (30 minutes) before they occur via the
observation's `meal_announced` field and `meal_grams_announced` field.
In Task 3, do NOT announce meals — the agent must learn meal patterns.

---

## SECTION 4 — PROJECT FILE STRUCTURE

Build exactly this structure. Do not add extra files or deviate:

```
glucorl/
├── inference.py                  # Baseline inference script (root level, MANDATORY)
├── models.py                     # Pydantic models: GlucoAction, GlucoObservation, GlucoState, GlucoReward
├── client.py                     # GlucoEnv sync client (mirrors Kube SRE Gym client.py pattern)
├── openenv.yaml                  # OpenEnv spec metadata
├── Dockerfile                    # HF Spaces deployment
├── requirements.txt              # All dependencies
├── README.md                     # Full documentation (see Section 9)
├── eval.py                       # Evaluation script: baseline LLM vs PID controller
├── server/
│   ├── app.py                    # FastAPI server: /reset /step /state /tasks endpoints
│   ├── glucorl_environment.py    # Core environment: reset() step() state()
│   ├── patient_manager.py        # simglucose patient initialisation and stepping
│   ├── reward_calculator.py      # Reward function with all components
│   ├── graders.py                # Task graders: score_task_1, score_task_2, score_task_3
│   ├── pid_controller.py         # PID baseline for benchmark comparison
│   └── constants.py              # Glucose thresholds, patient names, meal schedule
└── tests/
    ├── test_environment.py       # Test reset/step/state work correctly
    ├── test_graders.py           # Test graders return 0.0–1.0 deterministically
    └── test_reward.py            # Test reward function gives expected values
```

---

## SECTION 5 — PYDANTIC MODELS (models.py)

Follow the exact same Pydantic model pattern as Kube SRE Gym's models.py.
Study that file first, then implement these:

```python
# models.py
from pydantic import BaseModel, Field
from typing import Optional, Literal

class GlucoAction(BaseModel):
    """
    Insulin dosing action taken by the agent each step (every 3 minutes).
    basal_rate: continuous background insulin in units/hr (0.0 to 5.0)
    bolus_dose: correction/meal insulin in total units (0.0 to 20.0)
    """
    basal_rate: float = Field(default=1.0, ge=0.0, le=5.0,
        description="Basal insulin rate in units/hr")
    bolus_dose: float = Field(default=0.0, ge=0.0, le=20.0,
        description="Bolus insulin dose in units")

class GlucoObservation(BaseModel):
    """
    What the agent observes at each step.
    """
    glucose_mg_dl: float = Field(description="Current CGM glucose reading in mg/dL")
    glucose_trend: Literal["rapidly_falling", "falling", "stable",
                           "rising", "rapidly_rising"] = Field(
        description="CGM trend arrow based on rate of change")
    meal_announced: bool = Field(default=False,
        description="True if a meal is coming in the next 30 minutes (Task 2 only)")
    meal_grams_announced: float = Field(default=0.0,
        description="Carbohydrate grams in the announced upcoming meal")
    time_of_day_hours: float = Field(
        description="Current time in simulated day (0.0 to 24.0 hours)")
    step: int = Field(description="Current step number (0 to 479)")
    patient_id: Optional[str] = Field(default=None,
        description="Patient identifier (None in Task 3 to force generalisation)")
    last_action_basal: float = Field(default=1.0,
        description="Basal rate from previous step")
    last_action_bolus: float = Field(default=0.0,
        description="Bolus dose from previous step")

class GlucoReward(BaseModel):
    """
    Decomposed reward signal for the current step.
    """
    tir_contribution: float = Field(
        description="Reward for being in target range 70-180 mg/dL: +1.0 if in range")
    hypo_penalty: float = Field(
        description="Penalty for hypoglycemia: -1.0 if <70, -3.0 if <54 mg/dL")
    hyper_penalty: float = Field(
        description="Penalty for hyperglycemia: -0.5 if >180, -1.5 if >250 mg/dL")
    overdose_penalty: float = Field(default=0.0,
        description="Penalty of -3.0 if glucose crashes below 54 within 2 steps of a bolus")
    step_total: float = Field(description="Total reward for this step (sum of components)")

class GlucoState(BaseModel):
    """
    Full environment state returned by state() endpoint.
    """
    task_id: int = Field(description="Current task: 1, 2, or 3")
    patient_name: str = Field(description="simglucose patient identifier")
    step: int = Field(description="Current step in episode")
    done: bool = Field(description="Whether episode has ended")
    glucose_history: list[float] = Field(
        description="Full glucose reading history for this episode")
    reward_history: list[float] = Field(
        description="Step reward history for this episode")
    tir_current: float = Field(
        description="Current Time-in-Range percentage (0.0 to 1.0)")
    hypo_events: int = Field(description="Number of hypoglycemia steps so far")
    severe_hypo_events: int = Field(description="Steps below 54 mg/dL so far")
    hyper_events: int = Field(description="Number of hyperglycemia steps so far")
    episode_reward_total: float = Field(description="Cumulative reward for the episode")
```

---

## SECTION 6 — THREE TASKS WITH GRADERS (server/graders.py)

Each task must have a grader that returns a score between 0.0 and 1.0.
Graders must be deterministic and reproducible — same glucose history always
returns the same score. The graders take the completed episode state and
return a float.

### Task 1 — Basal Rate Control (Easy)

**Objective:** Keep a single stable adult patient's glucose in the 70–180 mg/dL
target range for as much of a simulated day as possible, using only basal rate
adjustments (no meal events, no bolus needed).

**Patient:** Always 'adult#001' (consistent, predictable dynamics)
**Meals:** None
**Steps:** 480 (full day)
**Grader:**
```
tir = (steps where glucose in 70–180) / total_steps
score = tir  # already 0.0–1.0
# Bonus: if no severe hypo events, add 0.05 (capped at 1.0)
# Penalty: subtract 0.1 for each severe hypo event (glucose < 54)
```
Expected scores: random agent ~0.2–0.35, PID ~0.65–0.75, good RL agent ~0.80+

### Task 2 — Meal Bolus Timing (Medium)

**Objective:** Manage the same adult patient through a full day with 3 announced
meals. Agent must deliver correct bolus doses at the right time around meals
to prevent post-meal spikes while avoiding hypoglycemia.

**Patient:** Always 'adult#001'
**Meals:** Announced 10 steps (30 min) in advance via observation
**Steps:** 480 (full day)
**Grader:**
```
tir = (steps in 70–180) / total_steps
post_meal_spike_penalty = 0.0
for each meal:
    peak_glucose = max(glucose_history[meal_step : meal_step + 60])
    if peak_glucose > 250: post_meal_spike_penalty += 0.15
    elif peak_glucose > 200: post_meal_spike_penalty += 0.08
    elif peak_glucose > 180: post_meal_spike_penalty += 0.03

hypo_penalty = min(0.3, severe_hypo_events * 0.1)
score = max(0.0, tir - post_meal_spike_penalty - hypo_penalty)
```
Expected scores: random agent ~0.1–0.2, PID ~0.55–0.65, good RL agent ~0.70+

### Task 3 — Cross-Patient Generalisation (Hard)

**Objective:** The environment samples a RANDOM patient from the full pool of
30 patients (adolescent, adult, child profiles). Meals are NOT announced.
The agent must develop a policy that works across varied patient physiology
without knowing which patient it is treating.

**Patient:** Random from all 30 at each reset() call. patient_id is set to None
in the observation to prevent the agent from trivially memorising per-patient policies.
**Meals:** 3 meals at fixed times (same schedule as Task 2) but NOT announced
**Steps:** 480 (full day)
**Grader:** Run 5 episodes with 5 different randomly sampled patients,
average the TIR-based score:
```
per_episode_score = max(0.0, tir - (severe_hypo_events * 0.15))
final_score = mean(per_episode_score across 5 runs)
```
Expected scores: random agent ~0.05–0.15, PID ~0.45–0.60, good RL agent ~0.60+

**IMPORTANT for Task 3 grader:** The grader must use a fixed random seed
(seed=42) when sampling the 5 patients to ensure deterministic, reproducible
scoring. Always use the same 5 patients: pick the first 5 from
random.Random(42).sample(ALL_PATIENT_NAMES, 5).

---

## SECTION 7 — REWARD FUNCTION (server/reward_calculator.py)

The reward function MUST provide signal at every step, not just at episode end.
It must not be binary. Here is the exact reward design to implement:

```python
def calculate_step_reward(
    glucose: float,
    prev_glucose: float,
    bolus_given: float,
    glucose_2_steps_ago: float,
    bolus_2_steps_ago: float
) -> GlucoReward:

    # 1. Time-in-Range contribution
    if 70.0 <= glucose <= 180.0:
        tir_contribution = 1.0
    elif 54.0 <= glucose < 70.0:
        tir_contribution = 0.0   # just outside range, no positive reward
    elif 180.0 < glucose <= 250.0:
        tir_contribution = 0.0
    else:
        tir_contribution = 0.0

    # 2. Hypoglycemia penalty (asymmetric — hypo is more dangerous)
    if glucose < 54.0:
        hypo_penalty = -3.0      # severe hypoglycemia — life threatening
    elif glucose < 70.0:
        hypo_penalty = -1.0      # mild hypoglycemia
    else:
        hypo_penalty = 0.0

    # 3. Hyperglycemia penalty
    if glucose > 250.0:
        hyper_penalty = -1.5     # severe hyperglycemia
    elif glucose > 180.0:
        hyper_penalty = -0.5     # mild hyperglycemia
    else:
        hyper_penalty = 0.0

    # 4. Overdose penalty — punishes bolus that caused a crash
    overdose_penalty = 0.0
    if glucose < 54.0 and bolus_2_steps_ago > 5.0:
        overdose_penalty = -3.0  # dangerous overdose pattern

    step_total = tir_contribution + hypo_penalty + hyper_penalty + overdose_penalty

    return GlucoReward(
        tir_contribution=tir_contribution,
        hypo_penalty=hypo_penalty,
        hyper_penalty=hyper_penalty,
        overdose_penalty=overdose_penalty,
        step_total=step_total
    )
```

This produces clear variance: good steps score +1.0, hypo steps score -1.0
to -3.0, hyper steps score -0.5 to -1.5. This is the kind of shaped reward
that makes RL training tractable.

---

## SECTION 8 — CORE ENVIRONMENT (server/glucorl_environment.py)

This is the most important file. Model it on Kube SRE Gym's
`kube_sre_gym_environment.py` for the class structure and method signatures,
but the internals are completely different.

```python
class GlucoRLEnvironment:

    def __init__(self, task_id: int = 1):
        self.task_id = task_id
        self.patient = None
        self.step_count = 0
        self.done = False
        self.glucose_history = []
        self.reward_history = []
        self.action_history = []
        self.episode_reward = 0.0
        # ... other state

    def reset(self) -> GlucoObservation:
        """
        Initialise a new episode:
        - Select patient based on task_id (Task 1&2: adult#001, Task 3: random)
        - Reset simglucose patient to initial conditions
        - Reset all counters
        - Return initial observation
        The initial glucose for adult#001 starts around 140–160 mg/dL
        """

    def step(self, action: GlucoAction) -> tuple[GlucoObservation, GlucoReward, bool, dict]:
        """
        Process one 3-minute step:
        1. Convert GlucoAction to simglucose Action namedtuple
           (basal_rate/60 for umin, bolus/3 for umin spread over step)
        2. Determine if a meal is happening this step and create CHOInput if so
        3. Call patient.step(sim_action) to advance the simulator
        4. Read new glucose from patient.observation.CGM
        5. Compute reward via reward_calculator
        6. Check if episode is done (step >= 480 or severe hypo 3 times)
        7. Return (observation, reward, done, info)
        """

    def state(self) -> GlucoState:
        """
        Return full current state including all history.
        Used by the grader and the /state endpoint.
        """

    def _build_observation(self) -> GlucoObservation:
        """
        Build a GlucoObservation from current patient state.
        Compute glucose_trend from last 2 readings:
        - rate of change > 2 mg/dL/min -> rapidly_rising
        - rate > 1 -> rising
        - rate < -2 -> rapidly_falling
        - rate < -1 -> falling
        - else -> stable
        Announce meal if Task 2 and a meal is within 10 steps.
        """
```

**Critical implementation note for simglucose:**
The simglucose T1DPatient may raise exceptions for extreme glucose values
or invalid insulin inputs. Wrap patient.step() in try/except and handle:
- glucose < 10 mg/dL: terminate episode immediately (patient death), done=True
- Any exception: log it, return last known glucose, done=True

---

## SECTION 9 — FASTAPI SERVER (server/app.py)

Model this exactly on Kube SRE Gym's app.py. The OpenEnv validator sends HTTP
requests to these exact endpoints. Use the same FastAPI + uvicorn pattern.

Required endpoints:
```
POST /reset          — body: {"task_id": 1}  returns GlucoObservation
POST /step           — body: GlucoAction JSON, returns {observation, reward, done, info}
GET  /state          — returns GlucoState
GET  /tasks          — returns list of task descriptions
GET  /health         — returns {"status": "ok"}
```

The server must maintain one environment instance per session. For simplicity
(single-user hackathon deployment), use a global environment instance protected
by asyncio lock. Follow exactly how Kube SRE Gym handles the global env instance
in app.py — do not reinvent this pattern.

WebSocket endpoint is optional but include it if Kube SRE Gym uses it, as the
OpenEnv validator may check for it.

---

## SECTION 10 — openenv.yaml

Study Kube SRE Gym's openenv.yaml carefully and create:

```yaml
spec_version: 1
name: glucorl
display_name: GlucoRL — Insulin Dosing RL Environment
description: >
  An OpenEnv environment for training RL agents to make personalized insulin
  dosing decisions for Type 1 Diabetic patients. Agents observe continuous
  glucose monitor readings and decide basal and bolus insulin delivery to
  maintain blood glucose in the safe 70-180 mg/dL range across 3 tasks
  of increasing difficulty.
type: space
runtime: fastapi
app: server.app:app
port: 8000
tags:
  - openenv
  - healthcare
  - reinforcement-learning
  - glucose
  - insulin
tasks:
  - id: task_1
    name: Basal Rate Control
    difficulty: easy
    description: Single stable patient, no meals, optimize basal insulin rate
  - id: task_2
    name: Meal Bolus Timing
    difficulty: medium
    description: Manage glucose around 3 announced daily meals with bolus dosing
  - id: task_3
    name: Cross-Patient Generalisation
    difficulty: hard
    description: Adapt insulin policy to random unseen patients with unannounced meals
action_space:
  type: continuous
  fields:
    basal_rate: {type: float, min: 0.0, max: 5.0, unit: units/hr}
    bolus_dose: {type: float, min: 0.0, max: 20.0, unit: units}
observation_space:
  fields:
    glucose_mg_dl: {type: float, min: 20.0, max: 600.0}
    glucose_trend: {type: string, enum: [rapidly_falling, falling, stable, rising, rapidly_rising]}
    meal_announced: {type: bool}
    meal_grams_announced: {type: float}
    time_of_day_hours: {type: float, min: 0.0, max: 24.0}
    step: {type: int, min: 0, max: 479}
```

---

## SECTION 11 — DOCKERFILE

Follow Kube SRE Gym's Dockerfile exactly for the base image and startup command.
The only changes are the package installation and app path:

```dockerfile
FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

requirements.txt must include:
```
simglucose>=0.2.2
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
openai>=1.0.0
numpy>=1.24.0
scipy>=1.10.0
```

**Critical constraint:** The entire environment must run on vCPU=2, RAM=8GB.
simglucose is pure Python arithmetic and uses negligible memory. Do NOT add
any heavy ML libraries to requirements.txt — no torch, no tensorflow, no sklearn.
The inference.py uses a remote LLM via API, not a local model.

---

## SECTION 12 — CLIENT (client.py)

Model exactly on Kube SRE Gym's client.py. Create a GlucoEnv class that:
- Takes base_url as constructor parameter
- Implements reset(task_id=1) -> returns GlucoObservation
- Implements step(action: GlucoAction) -> returns StepResult with .observation, .reward, .done, .info
- Implements state() -> returns GlucoState
- Works as a context manager (with GlucoEnv(...) as env:)

This is what inference.py uses. Keep it simple and synchronous.

---

## SECTION 13 — INFERENCE SCRIPT (inference.py)

This file is MANDATORY and must be at the root of the project (not inside server/).
It will be run by the automated hackathon evaluator. Follow the exact pattern of
the sample inference script provided, adapted for GlucoRL.

**Critical requirements:**
- Must use OpenAI client (not anthropic, not requests — OpenAI client)
- Must read credentials from environment variables:
  `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` (used as api_key)
- Must run all 3 tasks and print final scores
- Must complete in under 20 minutes total
- Must not crash — handle all exceptions gracefully with fallback actions
- Fallback action if LLM fails or returns unparseable response:
  GlucoAction(basal_rate=1.0, bolus_dose=0.0) — safe default

```python
"""
GlucoRL Inference Script
========================
Runs a language model agent through all 3 GlucoRL tasks.

Required environment variables:
    API_BASE_URL   LLM API endpoint (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier
    HF_TOKEN       HuggingFace token used as API key
"""

import os
import json
import re
from openai import OpenAI
from client import GlucoEnv, GlucoAction

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_STEPS = 480   # full day — but cap at 96 for inference speed (every 5th step)
INFERENCE_STEP_INTERVAL = 5  # agent acts every 5 steps, PID fills the rest
TEMPERATURE = 0.1
MAX_TOKENS = 100
FALLBACK_ACTION = GlucoAction(basal_rate=1.0, bolus_dose=0.0)

SYSTEM_PROMPT = """You are an AI insulin dosing system for a Type 1 Diabetic patient.
At each step you observe:
- Current blood glucose in mg/dL (target range: 70-180)
- Glucose trend (rapidly_falling/falling/stable/rising/rapidly_rising)
- Whether a meal is coming soon and how many carbohydrates
- Current time of day

You must respond with ONLY a JSON object like this:
{"basal_rate": 1.2, "bolus_dose": 0.0}

Rules:
- basal_rate: 0.0 to 5.0 units/hr (continuous background insulin)
- bolus_dose: 0.0 to 20.0 units (meal/correction insulin, use 0 if no meal)
- If glucose is falling or low, reduce basal_rate and set bolus_dose to 0
- If glucose is rising high, increase basal_rate slightly
- If a meal is announced, give a bolus_dose proportional to meal_grams / 10
- Never give bolus when glucose is below 120 mg/dL
Do not include any explanation. Respond with only the JSON."""


def build_user_prompt(obs, step: int) -> str:
    return f"""Step: {step}
Glucose: {obs.glucose_mg_dl:.1f} mg/dL
Trend: {obs.glucose_trend}
Time: {obs.time_of_day_hours:.1f} hours
Meal announced: {obs.meal_announced}
Meal carbs: {obs.meal_grams_announced:.0f}g
Last basal: {obs.last_action_basal:.2f} u/hr
Last bolus: {obs.last_action_bolus:.2f} u

Respond with JSON only: {{"basal_rate": X, "bolus_dose": Y}}"""


def parse_action(response_text: str) -> GlucoAction:
    if not response_text:
        return FALLBACK_ACTION
    try:
        # Extract JSON from response
        match = re.search(r'\{[^}]+\}', response_text)
        if match:
            data = json.loads(match.group(0))
            return GlucoAction(
                basal_rate=float(data.get('basal_rate', 1.0)),
                bolus_dose=float(data.get('bolus_dose', 0.0))
            )
    except Exception:
        pass
    return FALLBACK_ACTION


def run_task(client_openai, env_url: str, task_id: int) -> dict:
    print(f"\n{'='*50}")
    print(f"Running Task {task_id}...")
    print(f"{'='*50}")

    with GlucoEnv(base_url=env_url) as env:
        result = env.reset(task_id=task_id)
        obs = result.observation
        total_reward = 0.0
        steps_completed = 0
        act_step = 0

        for step in range(480):
            if result.done:
                break

            # Agent decides every INFERENCE_STEP_INTERVAL steps
            if step % INFERENCE_STEP_INTERVAL == 0:
                user_prompt = build_user_prompt(obs, step)
                try:
                    completion = client_openai.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                    response_text = completion.choices[0].message.content or ""
                    action = parse_action(response_text)
                except Exception as exc:
                    print(f"  LLM call failed at step {step}: {exc}")
                    action = FALLBACK_ACTION
                act_step += 1
            else:
                # Between agent decisions, hold the last action
                action = action if 'action' in dir() else FALLBACK_ACTION

            result = env.step(action)
            obs = result.observation
            total_reward += result.reward.step_total
            steps_completed += 1

            if step % 48 == 0:  # Print every 4 simulated hours
                print(f"  Step {step:3d} | Glucose: {obs.glucose_mg_dl:6.1f} mg/dL "
                      f"| Trend: {obs.glucose_trend:15s} "
                      f"| Reward: {result.reward.step_total:+.2f}")

        state = env.state()
        tir = state.tir_current
        print(f"\nTask {task_id} complete:")
        print(f"  Steps: {steps_completed}")
        print(f"  TIR: {tir:.1%}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Hypo events: {state.hypo_events}")
        print(f"  Severe hypo: {state.severe_hypo_events}")

        return {
            "task_id": task_id,
            "tir": tir,
            "total_reward": total_reward,
            "hypo_events": state.hypo_events,
            "severe_hypo_events": state.severe_hypo_events,
            "steps": steps_completed
        }


def main():
    env_url = os.getenv("GLUCORL_ENV_URL") or "http://localhost:8000"
    print(f"GlucoRL Inference Script")
    print(f"Model: {MODEL_NAME}")
    print(f"Environment: {env_url}")

    client_openai = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    results = []

    for task_id in [1, 2, 3]:
        try:
            r = run_task(client_openai, env_url, task_id)
            results.append(r)
        except Exception as e:
            print(f"Task {task_id} failed: {e}")
            results.append({"task_id": task_id, "tir": 0.0, "total_reward": -999})

    print(f"\n{'='*50}")
    print("FINAL BASELINE SCORES")
    print(f"{'='*50}")
    for r in results:
        print(f"Task {r['task_id']}: TIR={r.get('tir',0):.1%} | "
              f"Reward={r.get('total_reward',0):.1f}")


if __name__ == "__main__":
    main()
```

---

## SECTION 14 — PID BASELINE (server/pid_controller.py)

Implement a simple PID controller for benchmark comparison. This is used in
eval.py to show the RL environment has a meaningful baseline to beat.

```python
class PIDController:
    """
    A simple PID controller for blood glucose management.
    Target: 120 mg/dL (centre of safe range)
    This mirrors what commercial insulin pumps use.
    """
    def __init__(self, target_glucose=120.0, kp=0.02, ki=0.0005, kd=0.01):
        self.target = target_glucose
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def act(self, glucose: float) -> GlucoAction:
        error = glucose - self.target
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        basal_adjustment = self.kp * error + self.ki * self.integral + self.kd * derivative
        basal_rate = max(0.0, min(5.0, 1.0 + basal_adjustment))

        # Simple meal bolus: give correction if glucose very high
        bolus_dose = max(0.0, (glucose - 200) / 50.0) if glucose > 200 else 0.0

        return GlucoAction(basal_rate=basal_rate, bolus_dose=bolus_dose)
```

---

## SECTION 15 — CONSTANTS (server/constants.py)

```python
# Glucose thresholds
GLUCOSE_TARGET_LOW = 70.0
GLUCOSE_TARGET_HIGH = 180.0
GLUCOSE_SEVERE_HYPO = 54.0
GLUCOSE_SEVERE_HYPER = 250.0
GLUCOSE_DEATH = 10.0

# Episode length
STEPS_PER_EPISODE = 480   # 480 steps × 3 min = 24 hours
STEP_DURATION_MIN = 3     # minutes per step

# Meal schedule (step numbers, fixed)
MEAL_SCHEDULE = {
    100: 50.0,   # Breakfast: step 100 = 5 hours in, 50g CHO
    200: 70.0,   # Lunch: step 200 = 10 hours in, 70g CHO
    320: 80.0,   # Dinner: step 320 = 16 hours in, 80g CHO
}
MEAL_ANNOUNCEMENT_STEPS = 10  # announce meal this many steps in advance

# All available patient names in simglucose
ALL_PATIENT_NAMES = (
    [f"adolescent#00{i}" for i in range(1, 10)] +
    ["adolescent#010"] +
    [f"adult#00{i}" for i in range(1, 10)] +
    ["adult#010"] +
    [f"child#00{i}" for i in range(1, 10)] +
    ["child#010"]
)

# Default patient for Task 1 and Task 2
DEFAULT_PATIENT = "adult#001"

# Task 3 deterministic patient sample (fixed seed for reproducibility)
import random
TASK3_EVAL_PATIENTS = random.Random(42).sample(ALL_PATIENT_NAMES, 5)
```

---

## SECTION 16 — ROUND 1 MANDATORY CHECKLIST

Every single item below must be satisfied before this project can be submitted.
Work through them in order. Do not move to the next item until the current one passes.

### Phase 1 — Automated Pass/Fail Gates (DISQUALIFICATION if any fail)

- [ ] **HF Space deploys** — `docker build` and `docker run` must succeed without errors
- [ ] **Ping returns 200** — GET /health returns `{"status": "ok"}`
- [ ] **reset() responds** — POST /reset with `{"task_id": 1}` returns a valid GlucoObservation JSON
- [ ] **OpenEnv spec compliance** — `openenv validate` passes on the openenv.yaml and all endpoints
- [ ] **Dockerfile builds** — no missing dependencies, no build errors
- [ ] **Baseline reproduces** — `python inference.py` runs to completion without uncaught exceptions
- [ ] **3+ tasks with graders** — /tasks endpoint enumerates 3 tasks, each grader returns score in [0.0, 1.0]
- [ ] **Graders are deterministic** — running the same episode twice produces the same score
- [ ] **inference.py at root** — file must be named exactly `inference.py` in the project root
- [ ] **Uses OpenAI client** — inference.py must use `from openai import OpenAI`, not requests or httpx

### Phase 2 — Agentic Evaluation (Scored)

- [ ] **Reward is non-binary** — check that step rewards vary meaningfully across an episode
- [ ] **Reward provides partial credit** — partial in-range time yields proportional score, not 0
- [ ] **Hard task genuinely hard** — Task 3 with a frontier LLM should score noticeably lower than Task 1
- [ ] **Score variance exists** — different agents must produce different scores on the same task
- [ ] **Environment runs on 2vCPU/8GB** — no heavy ML libraries, simglucose is lightweight
- [ ] **Inference completes in <20 min** — test timing locally before submitting

### Phase 3 — Human Review (Top submissions only)

- [ ] **Real-world utility is clear in README** — the T1D problem and clinical gap must be explained
- [ ] **PID baseline comparison included** — show your environment produces better results than PID
- [ ] **No grader exploits** — graders cannot be gamed by trivial actions (e.g., always bolus=0)
- [ ] **Action/observation spaces documented** — README has clear tables for both

---

## SECTION 17 — README.md CONTENT

The README drives 30% of the score (real-world utility). Write it to make the
clinical stakes immediately clear. Include these sections in order:

1. **One-line description** — "GlucoRL: An OpenEnv environment for training AI agents to manage insulin dosing in Type 1 Diabetes"
2. **The problem** — Why insulin dosing is hard, what happens when it goes wrong, why PID controllers fail to adapt
3. **Why RL** — What an RL agent can learn that a PID controller cannot (patient personalisation, meal adaptation)
4. **Environment description** — What the agent observes, what actions it takes, what reward it receives
5. **Tasks** — Table with Task 1/2/3, difficulty, objective, expected score range
6. **Action space table** — field name, type, range, description
7. **Observation space table** — field name, type, range, description
8. **Reward function** — explain all components and why hypo is penalised more than hyper
9. **Baseline scores** — table of LLM agent vs PID controller scores across all 3 tasks
10. **Setup instructions** — local docker run steps (see Section 18 below)
11. **Training with RL** — brief mention that the environment supports GRPO training via TRL

---

## SECTION 18 — BUILD ORDER FOR IMPLEMENTATION

Implement files in exactly this order. Test each one before moving to the next.
Do not write all files at once and then test — that makes debugging exponentially harder.

```
Step 1: server/constants.py          — no dependencies, no testing needed
Step 2: models.py                    — test: python -c "from models import GlucoAction; print(GlucoAction())"
Step 3: server/reward_calculator.py  — test: python -c "from server.reward_calculator import calculate_step_reward; print(calculate_step_reward(150, 140, 0, 140, 0))"
Step 4: server/patient_manager.py    — test: python -c "from server.patient_manager import PatientManager; p = PatientManager(); p.reset('adult#001'); print(p.get_glucose())"
Step 5: server/glucorl_environment.py — test: run a full 480-step episode, print glucose history
Step 6: server/graders.py            — test: run episode, call each grader, verify 0.0-1.0 output
Step 7: server/pid_controller.py     — test: run PID through Task 1, print TIR score
Step 8: server/app.py                — test: uvicorn server.app:app, curl /health, curl -X POST /reset
Step 9: client.py                    — test: with GlucoEnv("http://localhost:8000") as env: env.reset(1)
Step 10: inference.py                — test: set env vars, python inference.py, verify 3 scores print
Step 11: Dockerfile                  — test: docker build -t glucorl . && docker run -p 8000:8000 glucorl
Step 12: openenv.yaml                — test: openenv validate (install openenv CLI first)
Step 13: README.md                   — write last when all scores are known
Step 14: eval.py                     — run PID vs LLM comparison, record scores for README
```

---

## SECTION 19 — TESTING REQUIREMENTS

Write tests in tests/ directory:

### test_environment.py must verify:
- reset() returns a GlucoObservation with glucose in plausible range (40–400 mg/dL)
- step() with valid action advances step counter by 1
- step() with borderline actions (basal=0.0, bolus=20.0) does not crash
- done=True after 480 steps
- done=True if severe hypo occurs 3 times
- state() returns glucose_history of length equal to current step

### test_graders.py must verify:
- All graders return float in [0.0, 1.0]
- Same glucose_history produces same score every time (determinism)
- Episode with all glucose in 70–180 scores > 0.9 on Task 1
- Episode with 10 severe hypo events scores < 0.2 on Task 1

### test_reward.py must verify:
- Glucose 120 (in range) → step_total = +1.0
- Glucose 50 (severe hypo) → step_total = -3.0
- Glucose 300 (severe hyper) → step_total = -1.5
- Glucose 65 (mild hypo) → step_total = -1.0

---

## SECTION 20 — IMPORTANT IMPLEMENTATION NOTES

1. **simglucose import pattern:** simglucose has changed APIs across versions.
   The safest import is:
   ```python
   from simglucose.patient.t1dpatient import T1DPatient
   from simglucose.simulation.env import T1DSimEnv
   ```
   If T1DSimEnv doesn't work as expected, use T1DPatient directly with manual
   meal injection. Test the import before building patient_manager.py.

2. **Meal injection in simglucose:** Meals are injected as CHO (carbohydrates)
   via the Action namedtuple: `Action = namedtuple('Action', ['basal', 'bolus', 'CHO'])`
   or via a separate meal input depending on simglucose version.
   Check `simglucose.actuator.pump` and `simglucose.patient.t1dpatient` source
   to confirm the correct API for your installed version.

3. **Initial glucose randomisation:** To prevent the agent from overfitting to
   one starting glucose, add ±20 mg/dL noise to the initial glucose on each
   reset() call. This is essential for generalisation.
   ```python
   import numpy as np
   initial_noise = np.random.uniform(-20, 20)
   ```

4. **The inference.py env URL:** The inference.py connects to the environment
   server via HTTP. When running locally, the URL is http://localhost:8000.
   When running on HF Spaces, the URL is the Space URL. Use the environment
   variable GLUCORL_ENV_URL with localhost as default.

5. **Do not use async in client.py:** Keep it synchronous (requests library).
   The inference.py is a simple sequential script and async adds no value.

6. **Pydantic v2 compatibility:** Use `model_validate`, `model_dump` (not
   `parse_obj`, `dict()`). The Kube SRE Gym codebase may use Pydantic v1
   patterns — adapt to v2 if your environment uses Pydantic v2.

7. **Episode termination conditions:**
   - Normal: step >= 480
   - Emergency: glucose < 10 mg/dL (patient death simulation)
   - Safety: severe hypo (glucose < 54) for 5 consecutive steps

8. **OpenEnv validator:** Install with `pip install openenv` or check the
   hackathon repo for the correct install method. Run `openenv validate .`
   from the project root before submitting.

---

## SECTION 21 — USER INTERVENTION STEPS
## (Things the AI cannot do — requires manual human action)

The following steps require YOU (the human) to do manually.
The AI will build the code, but these actions need human accounts and credentials.

### Step A — Create HuggingFace Space (do this early)
1. Go to huggingface.co → New Space
2. Space name: `glucorl` (or `glucorl-env` if taken)
3. SDK: **Docker** (not Gradio, not Streamlit)
4. Visibility: **Public** (required for hackathon evaluation)
5. Hardware: **CPU Basic** (free tier — our env runs fine on 2vCPU/8GB)
6. Note the Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/glucorl`

### Step B — Create GitHub Repository
1. Create a new GitHub repo named `glucorl`
2. Clone it locally: `git clone https://github.com/YOUR_USERNAME/glucorl`
3. This is where you'll put all the code the AI generates

### Step C — Link GitHub to HuggingFace Space
1. In your HF Space settings → Repository → Link to GitHub repo
2. OR push directly to the HF Space git remote:
   `git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/glucorl`

### Step D — Set HuggingFace Space Secrets
In your HF Space → Settings → Repository Secrets, add:
- `ANTHROPIC_API_KEY` — if you want the inference.py to use Claude
- No other secrets needed for the environment server itself

### Step E — Install simglucose and test locally FIRST
Before building anything, verify simglucose installs cleanly:
```bash
pip install simglucose
python -c "from simglucose.patient.t1dpatient import T1DPatient; p = T1DPatient.withName('adult#001'); print('simglucose works:', p)"
```
If this fails, note the exact error and tell the AI — it will need to adjust
the import paths based on your installed version.

### Step F — Install OpenEnv CLI for validation
```bash
pip install openenv
openenv --version
```
Run `openenv validate .` from your project root before final submission.

### Step G — Docker test locally before pushing to HF
```bash
docker build -t glucorl .
docker run -p 8000:8000 glucorl
# In a new terminal:
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_id": 1}'
```
Both must return valid JSON with no errors before pushing.

### Step H — Test inference.py end-to-end
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="hf_your_token_here"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export GLUCORL_ENV_URL="http://localhost:8000"
python inference.py
```
Must print scores for all 3 tasks and exit cleanly.

### Step I — Push to HuggingFace Space (final deployment)
```bash
git add .
git commit -m "GlucoRL complete OpenEnv environment"
git push hf main   # pushes to HF Space
# OR if using GitHub + HF sync:
git push origin main
```
Wait 2–3 minutes for the HF Space to rebuild, then verify:
- Space URL loads
- `curl https://YOUR_USERNAME-glucorl.hf.space/health` returns 200

### Step J — Run final pre-submission checklist
```bash
# 1. Validate OpenEnv spec
openenv validate .

# 2. Confirm docker build works cleanly from scratch
docker build --no-cache -t glucorl .

# 3. Run inference against the live HF Space
export GLUCORL_ENV_URL="https://YOUR_USERNAME-glucorl.hf.space"
python inference.py

# 4. Confirm all 3 task scores are printed and in range 0.0–1.0
```

---

## SECTION 22 — FINAL SUBMISSION GUIDE

When all the above is done:

1. **GitHub repo must contain:** all code files, Dockerfile, openenv.yaml,
   requirements.txt, README.md, inference.py at root
2. **HF Space must be:** public, Docker SDK, running, returning 200 on /health
3. **Submit to hackathon:** provide the HF Space URL and GitHub repo URL
4. **HF Space tags:** ensure your Space has the `openenv` tag in its README
   frontmatter (already included in openenv.yaml but also add to Space README header):
   ```yaml
   ---
   title: GlucoRL
   emoji: 💉
   colorFrom: blue
   colorTo: green
   sdk: docker
   pinned: false
   app_port: 8000
   tags:
     - openenv
   ---
   ```

---

## FINAL INSTRUCTION TO AI

Build the entire GlucoRL project following all sections above, in the build
order specified in Section 18. After each file is generated, show the test
command and expected output. If simglucose behaves differently from what is
described here (different API, different import paths), adapt gracefully and
note what changed. The goal is a fully working, deployable, spec-compliant
OpenEnv environment that passes all Phase 1 automated validation gates and
scores well on Phase 2 agentic evaluation. Do not skip any section. Do not
add components from Kube SRE Gym that are not listed here (no curriculum
controller, no adversarial designer, no LLM judge, no GKE backend).
