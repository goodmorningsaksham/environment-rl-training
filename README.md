# GLUCORL ENHANCEMENT PROMPT
# Attach the current GlucoRL project zip alongside this prompt

---

## YOUR ROLE

You are an expert Python engineer enhancing an existing, fully working OpenEnv
reinforcement learning project called **GlucoRL**. The project is already
deployed on HuggingFace Spaces and passes all hackathon Round 1 validation
gates. Your job is to add new features and enhancements to make it more
clinically realistic, more useful for researchers, and more impressive to judges.

---

## STEP 1 — READ THE CODEBASE FIRST (MANDATORY)

Before doing anything else, read every file in the attached zip carefully.
You must understand the existing implementation before touching a single line.

Read these files in this order:

1. `server/constants.py` — glucose thresholds, meal schedule, patient names
2. `models.py` — GlucoAction, GlucoObservation, GlucoState, GlucoReward
3. `server/patient_manager.py` — how simglucose T1DPatient is used, how Gsub is read
4. `server/reward_calculator.py` — current reward function and all components
5. `server/glucorl_environment.py` — reset(), step(), state(), observation building
6. `server/graders.py` — score_task_1, score_task_2, score_task_3, grade()
7. `server/app.py` — FastAPI server, create_app factory, custom endpoints
8. `client.py` — GlucoEnv WebSocket client
9. `inference.py` — baseline inference script
10. `server/pid_controller.py` — PID baseline
11. `tests/test_environment.py`, `tests/test_graders.py`, `tests/test_reward.py`
12. `README.md` — existing documentation and baseline scores
13. `Dockerfile` and `requirements.txt` — deployment configuration

After reading all files, confirm with this exact response:

"I have read and understood the full GlucoRL codebase.
Key facts confirmed:
- simglucose uses observation.Gsub (NOT .CGM) for glucose readings
- GlucoObservation inherits from OpenEnv Observation base class
- GlucoState inherits from OpenEnv State base class
- GlucoAction inherits from OpenEnv Action base class
- Environment uses create_app factory from openenv.core
- Patient step uses SimAction(insulin=insulin_umin, CHO=cho) namedtuple
- One environment step = 3 simglucose mini-steps (STEP_DURATION_MIN=3)
- inference.py uses OpenAI client with API_BASE_URL, MODEL_NAME, HF_TOKEN
Ready to discuss enhancements."

Do not proceed until you have confirmed all of the above.

---

## STEP 2 — ASK THE USER WHICH ENHANCEMENTS TO IMPLEMENT

After confirming you read the codebase, present the following menu to the user
and ask them which items they want implemented. Wait for their answer before
writing any code.

Present the menu EXACTLY like this:

---

"Here are the available enhancements ranked from highest to lowest priority.
Which would you like me to implement? You can choose one, several, or all.
Tell me the numbers and I will implement them in priority order.

PRIORITY 1 — HIGH IMPACT, LOW EFFORT (30-120 minutes each):

[1] CGM Measurement Noise
    Add ±10 mg/dL Gaussian noise to glucose readings in patient_manager.py.
    Simulates real CGM sensor accuracy (ISO 15197 standard).
    The agent sees noisy readings; reward is computed on true glucose.
    Files changed: server/patient_manager.py, server/glucorl_environment.py,
    models.py (add true_glucose_mg_dl field), tests/test_reward.py

[2] Insulin-on-Board (IOB) Observation
    Track active insulin remaining from recent boluses using exponential decay.
    Add insulin_on_board_units field to GlucoObservation.
    IOB peaks at 60 min and clears over 4-5 hours — commercially standard.
    Files changed: models.py, server/glucorl_environment.py,
    server/constants.py (IOB decay constants), tests/test_environment.py

[3] Detailed Grader Score Breakdown
    Add grade_detailed(task_id, state) function returning decomposed score dict:
    {total, tir_score, post_meal_penalty, hypo_penalty, components}.
    Add /grade endpoint to app.py that accepts state and returns full breakdown.
    Makes environment genuinely useful for researchers diagnosing agent behaviour.
    Files changed: server/graders.py, server/app.py, tests/test_graders.py

[4] Live Glucose Visualisation Dashboard
    Add a /dashboard HTML endpoint to app.py showing a real-time glucose trace.
    Chart.js line plot with colour zones: green (70-180), red (<70), orange (>180).
    Displays current episode stats: TIR, step count, hypo events.
    Files changed: server/app.py (new /dashboard endpoint with inline HTML)

PRIORITY 2 — HIGH IMPACT, MEDIUM EFFORT (4-8 hours each):

[5] Exercise Events
    Add exercise_intensity field to GlucoObservation (0.0=rest, 1.0=intense).
    Exercise increases insulin sensitivity by 20-50% — the hardest T1D challenge.
    Random exercise events in Task 3 (unannounced, like meals).
    Announced in Task 2 (30 min ahead, like meals).
    Files changed: models.py, server/constants.py, server/patient_manager.py,
    server/glucorl_environment.py, server/graders.py, tests/test_environment.py

[6] Recovery Bonus in Reward Shaping
    Add trajectory-aware reward: +0.5 bonus when agent corrects a hypo/hyper
    event within 10 steps of it occurring.
    Teaches active correction rather than passive waiting.
    Files changed: server/reward_calculator.py, models.py (GlucoReward),
    server/glucorl_environment.py, tests/test_reward.py

[7] Glucose History Window in Observation
    Add glucose_history_window field to GlucoObservation: last 12 readings
    (36 minutes of CGM history) as a list of floats.
    Forces agents to use temporal context for better decisions.
    Files changed: models.py, server/glucorl_environment.py,
    tests/test_environment.py

PRIORITY 3 — MEDIUM IMPACT, HIGHER EFFORT (full day each):

[8] Sick Day Task 4 (Illness/Insulin Resistance)
    New task where patient has simulated illness: random 1.5-2x insulin
    resistance multiplier unknown to the agent. Agent must detect and adapt.
    Genuinely unsolvable by fixed-policy agents. True frontier difficulty.
    Files changed: server/constants.py, server/patient_manager.py,
    server/glucorl_environment.py, server/graders.py, server/app.py,
    openenv.yaml, README.md, tests/test_graders.py

[9] Sensor Failure / Partial Observability
    5% chance per episode of CGM dropout: glucose reads -1.0 for 3-6 steps.
    Agent must learn safe conservative policy during sensor blackout.
    Add sensor_active field to GlucoObservation.
    Files changed: models.py, server/glucorl_environment.py,
    server/patient_manager.py, tests/test_environment.py

[10] Multi-Day Episodes
    Add 3-day episode option (1440 steps). Patient starts day 2 with
    glucose state carried over from day 1 — tests persistent strategy.
    Add episode_duration_days parameter to reset().
    Files changed: server/constants.py, server/glucorl_environment.py,
    server/graders.py, client.py, tests/test_environment.py

Which enhancements would you like? Reply with numbers e.g. '1, 2, 3' or 'all'."

---

## STEP 3 — IMPLEMENTATION RULES

Once the user tells you which enhancements to implement, follow these rules
for every single one:

### Rule 1 — One enhancement at a time
Implement one enhancement fully before starting the next. Never mix code
from two different enhancements in the same response.

### Rule 2 — Always provide complete files
Never use placeholders like `# ... rest of implementation` or
`# same as before`. Every file you output must be the complete, full file
with all existing code preserved and new code added correctly.
If a file is long, use a CHECKPOINT and wait for 'continue'.

### Rule 3 — Preserve all existing functionality
Before adding new code, verify that:
- All existing Pydantic fields remain in models.py (never remove fields)
- All existing grader functions still work (never change scoring logic)
- The OpenEnv interface (reset/step/state) still works identically
- Dockerfile and requirements.txt do not need changes unless strictly necessary
- inference.py is NOT modified unless the enhancement explicitly requires it

### Rule 4 — Show test command after every file
After each file, write:
"File [name] updated. Test with: [exact command]"

### Rule 5 — Run tests after each enhancement
After completing all files for one enhancement, provide the exact test
commands to verify it works:
```
python -m pytest tests/ -v -k "[relevant test name]"
```
Then provide the expected output so the user knows what passing looks like.

### Rule 6 — If you are unsure about anything, ask
Especially about:
- simglucose API behaviour for new features
- Whether a new field should be Optional with a default
- Whether a change could break the OpenEnv create_app factory
- Whether a new dependency needs to be added to requirements.txt

### Rule 7 — Track what has been implemented
After completing each enhancement, output a short status list:
```
COMPLETED: [1] CGM Noise, [2] IOB
NEXT: [3] Detailed Grader Breakdown
REMAINING: [4], [5], [6]
```

---

## STEP 4 — IMPLEMENTATION SPECIFICATIONS

---

### ⚠️ CRITICAL NOTICE — READ BEFORE IMPLEMENTING ANYTHING IN THIS SECTION

The code snippets provided in every specification below are **reference
guidance only — they are NOT copy-paste implementations.**

They exist to communicate intent, logic direction, and key variable names.
They are deliberately incomplete, abbreviated, and context-free. If you
copy them directly without integrating them into the existing codebase they
WILL break the application.

**Your job as an expert engineer is to:**

1. Read the existing file completely before touching it
2. Understand exactly how the new feature integrates with what is already there
3. Write the complete, production-quality implementation yourself using the
   specification as a guide — not as a template
4. Ensure every existing feature, field, method, and test continues to work
   exactly as before
5. Use your full capabilities to design the cleanest, most robust integration
   possible — do not limit yourself to only what the snippet shows. If you see
   a better or safer way to implement something that achieves the same goal,
   use it and explain why
6. Handle edge cases the snippets do not show: what if a value is None, what
   if the episode ends mid-feature, what if the patient crashes during an
   exercise event, what if a sensor fails on step 0
7. When in doubt between a simpler safe implementation and a complex risky one,
   choose simple and safe — a working feature is always better than a broken
   impressive one

**The measure of success for each enhancement is:**
- All existing tests still pass with zero modifications to their assertions
- New tests pass for the new feature
- The FastAPI server starts cleanly with no import errors
- `openenv validate` still passes
- Docker builds and runs cleanly
- The enhancement makes the project genuinely better, not just bigger

**If any existing test would need its assertion changed to pass after your
implementation, STOP — you have broken something. Fix the root cause, not
the test.**

---

### [1] CGM Measurement Noise — Full Specification

**What changes in patient_manager.py:**
Add a `noise_enabled` parameter to PatientManager.__init__ (default True).
In the step() method, after reading `float(self._patient.observation.Gsub)`,
apply noise ONLY if noise_enabled is True:
```python
true_glucose = float(self._patient.observation.Gsub)
if self.noise_enabled:
    noise = np.random.normal(0, 10.0)
    cgm_glucose = true_glucose + noise
    cgm_glucose = max(20.0, min(600.0, cgm_glucose))
else:
    cgm_glucose = true_glucose
return cgm_glucose, true_glucose  # return both
```
Also update reset() to return both values.

**What changes in models.py:**
Add this field to GlucoObservation AFTER all existing fields:
```python
true_glucose_mg_dl: Optional[float] = Field(
    default=None,
    description="True blood glucose (Gsub) before CGM noise. "
                "None in production mode — exposed for research/debugging only."
)
```

**What changes in glucorl_environment.py:**
- PatientManager.step() now returns (cgm_glucose, true_glucose) tuple
- Store both: self._cgm_glucose_history and self._true_glucose_history
- Pass true_glucose to reward_calculator (reward on reality, not noise)
- Pass cgm_glucose to _build_observation (agent sees noisy reading)
- In _build_observation, set true_glucose_mg_dl=true_glucose only if
  self._debug_mode is True (add debug_mode param to __init__, default False)
- GlucoState should track true glucose history separately

**What changes in test files:**
Add to tests/test_environment.py:
- Test that CGM reading and true glucose can differ
- Test that reward is computed on true glucose not noisy reading
- Test that noise=0 case works (noise_enabled=False)

**README addition (add to Reward Function section):**
```
### CGM Simulation Fidelity

The environment simulates real CGM behaviour using the subcutaneous glucose
compartment (Gsub) from the UVa/Padova model, which naturally lags plasma
glucose by 5-15 minutes due to interstitial diffusion kinetics. Measurement
noise of σ=10 mg/dL is added to match real-world CGM accuracy specifications
per ISO 15197. The RL agent observes the noisy CGM reading; rewards are
computed on the true subcutaneous glucose value.
```

---

### [2] Insulin-on-Board (IOB) — Full Specification

**What changes in server/constants.py:**
Add after existing constants:
```python
# Insulin-on-board (IOB) pharmacokinetics
# Bilinear model: IOB peaks at 60 min, clears by ~240 min
IOB_PEAK_MIN = 60.0        # minutes to peak insulin activity
IOB_DURATION_MIN = 240.0   # minutes to full clearance
IOB_STEP_DECAY = 0.94      # per-step decay factor (3-min steps)
```

**What changes in server/glucorl_environment.py:**
Add IOB tracking to __init__:
```python
self._iob: float = 0.0  # current insulin on board in units
```

In reset(), add:
```python
self._iob = 0.0
```

In step(), after recording action, update IOB:
```python
# Add bolus to IOB, decay existing IOB
self._iob = (self._iob + bolus) * IOB_STEP_DECAY
self._iob = max(0.0, round(self._iob, 4))
```

In _build_observation(), add iob field to GlucoObservation.

**What changes in models.py:**
Add to GlucoObservation AFTER existing fields:
```python
insulin_on_board_units: float = Field(
    default=0.0,
    description="Active insulin remaining from recent boluses in units. "
                "Computed using bilinear pharmacokinetic decay model. "
                "Commercial pumps display this to prevent bolus stacking."
)
```

**What changes in client.py:**
Add `insulin_on_board_units` to _parse_result observation construction
with default 0.0 if key not present.

**What changes in tests/test_environment.py:**
Add:
- Test that IOB starts at 0.0 after reset
- Test that IOB increases after a bolus
- Test that IOB decays over multiple steps with no bolus
- Test that IOB resets between episodes

---

### [3] Detailed Grader Breakdown — Full Specification

**What changes in server/graders.py:**
Add after existing grade() function:

```python
def grade_detailed(task_id: int, state: GlucoState) -> dict:
    """
    Return full decomposed score breakdown for a completed episode.
    Same logic as grade() but exposes all components.

    Returns dict with:
        total: float           — final score 0.0-1.0
        tir_score: float       — raw Time-in-Range fraction
        tir_readings: int      — steps in range
        total_readings: int    — total steps scored
        hypo_penalty: float    — penalty applied for hypoglycemia
        severe_hypo_penalty: float
        post_meal_penalties: dict  — per-meal spike penalties (Task 2 only)
        components: dict       — all individual penalty components
        clinical_summary: dict — hypo_events, hyper_events, severe_hypo_events
    """
```

Implement it for each task using the same logic as the existing graders
but returning the full breakdown dict instead of just the total.

**What changes in server/app.py:**
Add new endpoint after /tasks:
```python
@app.post("/grade", tags=["Evaluation"])
async def grade_episode(task_id: int = 1):
    """
    Grade the current completed episode and return detailed score breakdown.
    Calls grade_detailed() on the current environment state.
    Returns 400 if episode is not done yet.
    """
```

**What changes in tests/test_graders.py:**
Add TestDetailedGrader class:
- Test that grade_detailed returns dict with all required keys
- Test that grade_detailed total matches grade() output
- Test that components sum to total
- Test that all values are finite floats

---

### [4] Live Glucose Visualisation Dashboard — Full Specification

**What changes in server/app.py:**
Add a /dashboard GET endpoint that returns an HTML page.
The page uses Chart.js (from cdnjs) to show:

1. A line chart of glucose over time with three background zones:
   - Green band: 70-180 mg/dL (target range)
   - Red band: 0-70 mg/dL (hypoglycemia zone)
   - Orange band: 180-400 mg/dL (hyperglycemia zone)

2. Current episode stats panel showing:
   - Current glucose with colour coding
   - Current TIR percentage
   - Step count / 480
   - Hypo events count
   - Severe hypo events count

3. Auto-refresh every 3 seconds via JavaScript fetch to /state endpoint

4. A "Reset Task" dropdown and button to start a new episode from the UI

The HTML should be returned as a Python string directly in the endpoint
using FastAPI's HTMLResponse. Do not create a separate templates directory.
Keep all CSS and JavaScript inline in the HTML string.
Use Chart.js from: https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js

The endpoint should be:
```python
from fastapi.responses import HTMLResponse

@app.get("/dashboard", response_class=HTMLResponse, tags=["UI"])
async def dashboard():
    """Live glucose monitoring dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)
```

Define DASHBOARD_HTML as a module-level string constant in app.py.

Add to README under API Endpoints table:
| GET | `/dashboard` | Live glucose monitoring web interface |

---

### [5] Exercise Events — Full Specification

**What changes in server/constants.py:**
```python
# Exercise event configuration
EXERCISE_INTENSITY_LEVELS = [0.3, 0.5, 0.7, 1.0]  # light to intense
EXERCISE_DURATION_STEPS = [10, 20, 30]   # 30 min to 90 min
EXERCISE_SENSITIVITY_MULTIPLIER = {      # insulin sensitivity increase
    0.3: 1.20,   # light exercise: 20% more sensitive
    0.5: 1.35,   # moderate: 35%
    0.7: 1.50,   # vigorous: 50%
    1.0: 1.70,   # intense: 70%
}
# Exercise schedule for Task 2 (announced): step 150 = 7.5 hours in
EXERCISE_SCHEDULE_TASK2 = {150: 0.5}     # step: intensity
EXERCISE_ANNOUNCEMENT_STEPS = 10
```

**What changes in models.py:**
Add to GlucoObservation:
```python
exercise_intensity: float = Field(
    default=0.0,
    ge=0.0, le=1.0,
    description="Current exercise intensity (0.0=rest, 1.0=maximum). "
                "Increases insulin sensitivity by 20-70%. "
                "Announced in Task 2, unannounced in Task 3."
)
exercise_announced: bool = Field(
    default=False,
    description="True if exercise is starting within 30 minutes (Task 2 only)."
)
```

**What changes in server/patient_manager.py:**
Add insulin_sensitivity_multiplier parameter to step():
```python
def step(self, basal_rate_uhr, bolus_dose_units,
         cho_grams=0.0, insulin_sensitivity_multiplier=1.0):
    # Apply exercise effect: more sensitive = less insulin needed
    # (multiply effective insulin by the sensitivity factor)
    effective_basal = basal_rate_uhr * insulin_sensitivity_multiplier
    effective_bolus = bolus_dose_units * insulin_sensitivity_multiplier
    # ... rest of existing conversion logic using effective values
```

**What changes in server/glucorl_environment.py:**
Add exercise state tracking to __init__:
```python
self._current_exercise_intensity: float = 0.0
self._exercise_steps_remaining: int = 0
self._exercise_schedule: dict = {}
```

In reset(), initialise exercise schedule based on task_id:
- Task 1: no exercise
- Task 2: fixed schedule from EXERCISE_SCHEDULE_TASK2
- Task 3: random exercise event with random intensity and duration

In step(), before calling patient_manager.step():
- Check if exercise starts this step, set intensity
- Decrement exercise_steps_remaining
- Clear intensity when steps exhausted
- Pass current intensity to patient_manager.step() as multiplier

In _build_observation(), set exercise_intensity and exercise_announced.

**What changes in server/graders.py:**
No changes to scoring logic. The existing TIR-based scoring automatically
penalises exercise-induced hypoglycemia through the existing hypo penalty.

**What changes in tests/test_environment.py:**
Add TestExerciseEvents class:
- Test Task 1 has no exercise events
- Test Task 2 exercise is announced before it starts
- Test Task 3 exercise events can occur (run 5 episodes, assert at least
  one has exercise_intensity > 0 at some point)
- Test that exercise_intensity returns to 0 after exercise ends

---

### [6] Recovery Bonus in Reward Shaping — Full Specification

**What changes in models.py:**
Add to GlucoReward:
```python
recovery_bonus: float = Field(
    default=0.0,
    description="Bonus of +0.5 when agent corrects a hypo/hyper event "
                "within 10 steps. Rewards active correction over passive waiting."
)
```

**What changes in server/glucorl_environment.py:**
Add recovery tracking to __init__:
```python
self._hypo_start_step: Optional[int] = None   # step when hypo began
self._hyper_start_step: Optional[int] = None  # step when hyper began
```

In step(), after computing glucose, pass recovery tracking info to
calculate_step_reward(). Track when glucose enters a bad zone and when
it recovers.

**What changes in server/reward_calculator.py:**
Update calculate_step_reward signature:
```python
def calculate_step_reward(
    glucose: float,
    prev_glucose: float,
    bolus_given: float,
    glucose_2_steps_ago: float,
    bolus_2_steps_ago: float,
    steps_since_hypo_start: Optional[int] = None,
    steps_since_hyper_start: Optional[int] = None,
) -> GlucoReward:
```

Recovery bonus logic:
```python
recovery_bonus = 0.0
RECOVERY_WINDOW = 10  # steps

# Hypo recovery: was below 70, now back in range within 10 steps
if (steps_since_hypo_start is not None and
    steps_since_hypo_start <= RECOVERY_WINDOW and
    glucose >= 70.0 and prev_glucose < 70.0):
    recovery_bonus = 0.5

# Hyper recovery: was above 180, now back in range within 10 steps
elif (steps_since_hyper_start is not None and
      steps_since_hyper_start <= RECOVERY_WINDOW and
      glucose <= 180.0 and prev_glucose > 180.0):
    recovery_bonus = 0.3  # smaller bonus — hyper less urgent

step_total = tir_contribution + hypo_penalty + hyper_penalty + \
             overdose_penalty + recovery_bonus
```

**What changes in tests/test_reward.py:**
Add TestRecoveryBonus class:
- Test hypo recovery within window gives +0.5
- Test hypo recovery outside window (>10 steps) gives 0
- Test hyper recovery within window gives +0.3
- Test no bonus when glucose stays in range throughout

---

### [7] Glucose History Window — Full Specification

**What changes in models.py:**
Add to GlucoObservation:
```python
glucose_history_window: list[float] = Field(
    default_factory=list,
    description="Last 12 CGM readings (36 minutes of history). "
                "Empty list before 12 steps have elapsed. "
                "Enables temporal reasoning without requiring RNN agents."
)
```

**What changes in server/glucorl_environment.py:**
In _build_observation(), compute the window:
```python
# Last 12 readings from CGM history (or fewer if episode just started)
window = self._cgm_glucose_history[-12:] if self._cgm_glucose_history else []
# Round to 1 decimal place to keep payload size reasonable
window = [round(g, 1) for g in window]
```

**What changes in client.py:**
Add `glucose_history_window` to _parse_result observation construction:
```python
glucose_history_window=obs_data.get("glucose_history_window", []),
```

**What changes in tests/test_environment.py:**
Add TestGlucoseHistoryWindow class:
- Test window is empty list at reset
- Test window grows to 12 after 12 steps
- Test window stays at max 12 after 20 steps (sliding window)
- Test window values match actual glucose history

---

### [8] Sick Day Task 4 — Full Specification

**What changes in server/constants.py:**
```python
# Task 4: Sick day insulin resistance
ILLNESS_RESISTANCE_MIN = 1.5   # minimum insulin resistance multiplier
ILLNESS_RESISTANCE_MAX = 2.5   # maximum insulin resistance multiplier
ILLNESS_ONSET_STEP_MIN = 20    # earliest illness can start
ILLNESS_ONSET_STEP_MAX = 100   # latest illness can start
```

**What changes in server/glucorl_environment.py:**
Add Task 4 support in reset():
- If task_id == 4, sample illness_resistance from uniform(1.5, 2.5)
- Sample illness_onset_step from uniform(20, 100)
- Store both as instance variables

In step(), if task_id == 4 and step >= illness_onset_step:
- Pass 1/illness_resistance as insulin_sensitivity_multiplier to patient
  (resistance = less effective insulin = less sensitive)

The agent is NOT told illness is happening or its severity —
it must infer from glucose behaviour.

**What changes in models.py:**
Add to GlucoObservation (optional debug field):
```python
illness_active: bool = Field(
    default=False,
    description="Whether illness/insulin resistance is active. "
                "Always False in Task 4 normal mode — exposed only for debugging."
)
```

**What changes in server/graders.py:**
Add score_task_4():
```python
def score_task_4(state: GlucoState) -> float:
    """
    Task 4 grader — Sick Day Management.
    Harder than Task 3: agent must detect and adapt to unknown insulin resistance.
    Expected scores: constant_basal ~0.05-0.15, PID ~0.10-0.20, good RL ~0.45+
    """
    glucose_history = state.glucose_history[1:]
    if not glucose_history:
        return 0.0

    total = len(glucose_history)
    in_range = sum(1 for g in glucose_history
                   if GLUCOSE_TARGET_LOW <= g <= GLUCOSE_TARGET_HIGH)
    tir = in_range / total

    # Illness makes hyper more likely — penalise severe hyper harder
    severe_hyper_steps = sum(1 for g in glucose_history if g > 300)
    severe_hyper_penalty = min(0.4, severe_hyper_steps / total * 2)

    score = tir - (state.severe_hypo_events * 0.15) - severe_hyper_penalty
    return max(0.0, min(1.0, score))
```

**What changes in server/app.py:**
Update /tasks endpoint to include Task 4.
Update grade_detailed() to handle task_id=4.

**What changes in openenv.yaml:**
Add Task 4 to tasks list:
```yaml
  - id: task_4
    name: Sick Day Management
    difficulty: expert
    description: Unknown insulin resistance from simulated illness. Agent must detect and adapt without being told.
```

**What changes in README.md:**
Add Task 4 row to Tasks table.
Add Task 4 to Baseline Scores table.

**What changes in tests/test_graders.py:**
Add TestTask4Scoring class with same pattern as other task tests.

---

### [9] Sensor Failure / Partial Observability — Full Specification

**What changes in server/constants.py:**
```python
# CGM sensor failure simulation
SENSOR_FAILURE_PROBABILITY = 0.05  # 5% chance per episode
SENSOR_FAILURE_DURATION_MIN = 3    # minimum blackout steps
SENSOR_FAILURE_DURATION_MAX = 6    # maximum blackout steps
SENSOR_FAILURE_VALUE = -1.0        # value returned during blackout
```

**What changes in models.py:**
Add to GlucoObservation:
```python
sensor_active: bool = Field(
    default=True,
    description="False when CGM sensor has failed. "
                "glucose_mg_dl will be -1.0 during sensor blackout. "
                "Agent must use safe conservative policy during blackout."
)
```

**What changes in server/glucorl_environment.py:**
Add sensor state to __init__:
```python
self._sensor_active: bool = True
self._sensor_failure_steps_remaining: int = 0
```

In reset(), determine if this episode has a sensor failure:
```python
if random.random() < SENSOR_FAILURE_PROBABILITY:
    self._sensor_failure_start = random.randint(50, 300)
    self._sensor_failure_duration = random.randint(
        SENSOR_FAILURE_DURATION_MIN, SENSOR_FAILURE_DURATION_MAX)
else:
    self._sensor_failure_start = None
    self._sensor_failure_duration = 0
```

In step(), before building observation:
```python
if (self._sensor_failure_start is not None and
    self._step_count == self._sensor_failure_start):
    self._sensor_active = False
    self._sensor_failure_steps_remaining = self._sensor_failure_duration

if self._sensor_failure_steps_remaining > 0:
    self._sensor_failure_steps_remaining -= 1
    if self._sensor_failure_steps_remaining == 0:
        self._sensor_active = True
```

In _build_observation():
```python
display_glucose = glucose if self._sensor_active else SENSOR_FAILURE_VALUE
```

Reward is always computed on true glucose regardless of sensor status.

**What changes in tests/test_environment.py:**
Add TestSensorFailure class:
- Test sensor_active is True by default
- Test that when sensor_active=False, glucose_mg_dl is SENSOR_FAILURE_VALUE
- Test reward is not affected by sensor failure (uses true glucose)

---

### [10] Multi-Day Episodes — Full Specification

**What changes in server/constants.py:**
```python
STEPS_PER_DAY = 480           # existing constant — make sure this exists
MAX_EPISODE_DAYS = 3
STEPS_PER_EPISODE_MULTI = {   # total steps by day count
    1: 480,
    2: 960,
    3: 1440,
}
```

**What changes in server/glucorl_environment.py:**
Add episode_days parameter to reset():
```python
def reset(self, seed=None, episode_id=None, task_id=1,
          episode_days=1, **kwargs):
    self._episode_days = min(max(1, int(episode_days)), MAX_EPISODE_DAYS)
    self._steps_per_episode = STEPS_PER_EPISODE_MULTI[self._episode_days]
```

Change termination check from hardcoded 480:
```python
if self._step_count >= self._steps_per_episode:
    self._done = True
```

Between days (step 480, 960), do NOT reset patient state — just log that
a new day has started. The patient physiology carries over.

Add day_number field to GlucoObservation:
```python
day_number: int = Field(
    default=1,
    description="Current day in multi-day episode (1-3)."
)
```

**What changes in server/graders.py:**
Graders work on full glucose_history regardless of length — no changes
needed to scoring logic. Multi-day TIR is naturally computed.

**What changes in client.py:**
Update reset() to accept episode_days parameter and pass it through.

**What changes in tests/test_environment.py:**
Add TestMultiDay class:
- Test 1-day episode terminates at step 480
- Test 2-day episode terminates at step 960
- Test patient glucose at step 481 reflects state from step 480
  (no artificial reset between days)

---

## STEP 5 — AFTER ALL CHOSEN ENHANCEMENTS ARE COMPLETE

Once you have implemented all the enhancements the user selected, do the
following in order:

### 5a — Run full test suite
Provide the command:
```bash
python -m pytest tests/ -v
```
Show expected output format. Tell the user what to look for.
If any test fails, fix it before proceeding.

### 5b — Update README.md
Update these sections:
- Environment Description — mention new observation fields
- Tasks table — add any new tasks
- Observation Space table — add all new fields with descriptions
- Reward Function table — add recovery_bonus if [6] was implemented
- Baseline Scores — note which enhancements affect scoring
- API Endpoints table — add /dashboard if [4] was implemented,
  add /grade if [3] was implemented

Do NOT change the baseline scores numbers themselves — those need to be
re-run with eval.py after enhancements are deployed.

### 5c — Verify Dockerfile still works
Confirm no new system dependencies were added that require apt-get changes.
Confirm no new heavy libraries (torch, tensorflow) were added to requirements.txt.
If any new pip package was added, confirm it's in requirements.txt.

### 5d — Final checklist
Output this checklist with PASS/FAIL for each item:

```
[ ] All existing tests still pass (no regressions)
[ ] New tests added for each enhancement
[ ] models.py has no removed fields (only additions)
[ ] inference.py unchanged (unless enhancement required it)
[ ] Dockerfile unchanged (or minimal change documented)
[ ] openenv.yaml updated if new tasks added
[ ] README.md updated with new fields and endpoints
[ ] No placeholder code remaining (no # TODO, no pass, no ...)
```

---

## CRITICAL REMINDERS

1. `patient.observation.Gsub` — NOT `.CGM`. This is the correct field name
   for simglucose 0.2.11. Using `.CGM` will cause AttributeError.

2. SimAction namedtuple is `SimAction(insulin=X, CHO=Y)` — confirm import from
   `simglucose.simulation.env import Action as SimAction` still works before
   modifying patient_manager.py.

3. GlucoObservation, GlucoAction, GlucoState all inherit from OpenEnv base classes.
   Adding new Optional fields with defaults is safe and backward compatible.
   Never remove or rename existing fields — the OpenEnv factory and client
   both depend on the exact field names.

4. The create_app factory in app.py handles /reset, /step, /state, /health,
   /schema, and /ws automatically. Do not re-implement these endpoints manually.
   Only add new custom endpoints beyond what create_app provides.

5. All new constants must go in server/constants.py, not inline in other files.

6. All new reward components must go through GlucoReward model in models.py
   and be computed in reward_calculator.py, not inline in glucorl_environment.py.
