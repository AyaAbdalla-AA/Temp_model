# Temp_model

# From Notebook to Production

## Source-Grounded Output Simulation

**Topic:** From Notebook to Production  
**Output types:** Full slide text + lab instructions  
**Audience assumption:** Junior-to-intermediate data science / ML learners who have used notebooks and need to understand production deployment workflows.  
**Source policy:** Directly grounded in the provided links. Some implementation details in the lab are educational assumptions, clearly marked.

## Sources Used

- Medium / Towards Data Engineering: "From Notebook to Production: How Machine Learning Models Are Actually Deployed"
- Towards Data Science: "Data Science Workflows: Notebook to Production"
- Atlassian: "Continuous integration"
- freeCodeCamp: "Learn to use GitHub Actions: a step-by-step guide"

## Learning Outcomes

By the end, learners should be able to:

- Explain why notebooks are useful for exploration but risky as production systems.
- Identify the main changes needed to convert notebook code into production-ready ML code.
- Describe how version control, testing, CI, packaging, and deployment fit into an ML workflow.
- Create a small GitHub Actions workflow that validates an ML project on every push.
- Distinguish source-backed implementation practices from product/team-specific deployment choices.

---

# Slide Deck

## Slide 1 - Title

**From Notebook to Production**

How machine learning work moves from experimental notebooks into reliable, deployable systems.

**Subtitle:** Turning exploration into repeatable software.

**Speaker note:**  
This lesson is not about abandoning notebooks. It is about understanding where notebooks fit and what must happen after exploration if a model is going to serve real users.

---

## Slide 2 - Why This Topic Matters

**Slide text:**

Many ML projects begin in notebooks because notebooks are fast, visual, and flexible.

But production asks for different qualities:

- Repeatability
- Testability
- Version control
- Automation
- Monitoring
- Collaboration
- Clear deployment paths

**Key idea:** A notebook proves an idea can work. Production proves it can keep working.

**Speaker note:**  
Use the contrast between discovery and reliability. A notebook can be the right tool at the start and the wrong container at the end.

---

## Slide 3 - The Notebook Is a Workshop, Not the Factory

**Slide text:**

In a notebook, the workflow is optimized for exploration:

- Try a dataset
- Inspect features
- Train a model
- Plot results
- Adjust quickly

In production, the workflow is optimized for operation:

- Run the same way every time
- Fail clearly
- Be reviewed by others
- Deploy safely
- Be monitored after release

**Takeaway:** The goal is not to delete notebooks. The goal is to extract reliable logic from them.

---

## Slide 4 - What Usually Changes First

**Slide text:**

Moving from notebook to production usually means transforming:

- Cells into modules
- Manual steps into scripts
- Hidden state into explicit inputs
- One-off experiments into tracked versions
- Local assumptions into environment configuration
- Informal checks into automated tests

**Production question:** If someone else runs this tomorrow, will it behave the same way?

**Speaker note:**  
This slide should make learners inspect their own notebooks. The enemy is not messy exploration; it is pretending messy exploration is deployment-ready.

---

## Slide 5 - A Production ML Project Shape

**Slide text:**

A production-oriented ML project often separates responsibilities:

```text
project/
  data/              # references or pipelines, not random local state
  notebooks/         # exploration and analysis
  src/               # reusable training/inference code
  tests/             # automated checks
  configs/           # parameters and environment settings
  models/            # saved artifacts or registry references
  workflows/         # CI/CD automation
```

**Key idea:** Structure makes responsibilities visible.

**Assumption note:** Exact folder names vary by team, but separation of concerns is the source-grounded principle.

---

## Slide 6 - Refactoring the Notebook

**Slide text:**

Notebook logic becomes production-ready when core steps are extracted into reusable functions:

- Load data
- Validate data
- Transform features
- Train model
- Evaluate model
- Save model artifact
- Run inference

**Before:** "Run these cells in order."  
**After:** "Run this command with explicit inputs and outputs."

**Speaker note:**  
Emphasize that production code should not depend on which cell was run three minutes ago.

---

## Slide 7 - Version Control Is the First Safety Net

**Slide text:**

Git helps teams track:

- Code changes
- Experiment logic
- Configuration updates
- Review history
- Collaboration decisions

For ML projects, version control also supports reproducibility:

- Which code trained this model?
- Which configuration was used?
- Which tests passed before deployment?

**Takeaway:** If the workflow cannot be reviewed, it cannot be trusted.

---

## Slide 8 - Testing Makes ML Work Less Fragile

**Slide text:**

Production ML needs more than accuracy checks.

Useful tests include:

- Unit tests for preprocessing functions
- Input schema checks
- Model loading checks
- Prediction shape/type checks
- Regression tests for known examples
- Pipeline smoke tests

**Key idea:** Tests do not prove the model is perfect. They catch breakage before users do.

---

## Slide 9 - Continuous Integration

**Slide text:**

Continuous Integration means developers frequently merge changes into a shared codebase, and automated checks run to catch integration problems early.

In an ML project, CI can run:

- Code formatting or linting
- Unit tests
- Data validation checks
- Small training smoke tests
- Model inference checks

**Why it matters:** CI turns "I think this still works" into a repeatable signal.

**Source anchor:** Atlassian frames CI as automatically building and testing changes so teams can detect problems early.

---

## Slide 10 - GitHub Actions as a CI Tool

**Slide text:**

GitHub Actions can run workflows when events happen in a repository.

Example triggers:

- Push to a branch
- Pull request opened
- Manual workflow dispatch

Example workflow jobs:

- Set up Python
- Install dependencies
- Run tests
- Validate a training script
- Save logs or artifacts

**Key idea:** CI becomes part of the repository, not a separate ritual.

**Source anchor:** freeCodeCamp introduces GitHub Actions through workflow files that define triggers, jobs, and steps.

---

## Slide 11 - Deployment Is Not One Thing

**Slide text:**

ML deployment can mean different production patterns:

- Batch predictions on a schedule
- Real-time API endpoint
- Embedded model inside an application
- Streaming prediction system
- Human-in-the-loop decision support

The right pattern depends on:

- Latency needs
- Data freshness
- Cost
- Monitoring requirements
- User experience

**Takeaway:** Do not ask "How do we deploy ML?" Ask "How will this model be used?"

---

## Slide 12 - The Production Checklist

**Slide text:**

Before calling an ML model production-ready, ask:

- Is the code outside the notebook?
- Are inputs and outputs explicit?
- Can the project run from a clean environment?
- Are important steps tested?
- Does CI run on changes?
- Is the model artifact versioned or traceable?
- Is deployment pattern defined?
- Is monitoring planned?

**Closing line:** Production is not a destination. It is a discipline for keeping useful models useful.

---

# Lab Instructions

## Lab Title

**Convert a Notebook-Style ML Workflow into a CI-Checked Production Skeleton**

## Lab Goal

Learners will convert a simple notebook-style ML workflow into a small production-oriented repository structure and add a GitHub Actions workflow that runs tests automatically.

## Scenario

You are part of a data science team that trained a model in a notebook. The model works locally, but the team wants to move toward production. Your task is to extract the core logic into reusable Python modules, add basic tests, and configure CI so every push validates the project.

## Prerequisites

- Basic Python
- Basic Git/GitHub
- Familiarity with notebooks
- A GitHub repository
- Python 3.10 or newer

## Starter Repository Shape

Create this structure:

```text
notebook-to-production-lab/
  src/
    ml_project/
      __init__.py
      data.py
      features.py
      train.py
      predict.py
  tests/
    test_features.py
    test_predict.py
  .github/
    workflows/
      ci.yml
  requirements.txt
  README.md
```

## Task 1 - Identify Notebook Responsibilities

**Instructions:**

Imagine your notebook has these cells:

- Load a CSV dataset
- Clean missing values
- Create numeric features
- Train a classifier
- Evaluate accuracy
- Save the model
- Run prediction on a sample input

Write the responsibility of each cell in a short table.

**Expected output:**

| Notebook cell | Production responsibility | Target file |
|---------------|---------------------------|-------------|
| Load CSV | Data loading | `data.py` |
| Clean values | Feature preparation | `features.py` |
| Train model | Training entry point | `train.py` |
| Predict sample | Inference function | `predict.py` |

**Checkpoint:**  
Learners should see that production work starts by naming responsibilities clearly.

## Task 2 - Extract Feature Logic

**Instructions:**

Create `src/ml_project/features.py`:

```python
def normalize_text(value: str) -> str:
    return value.strip().lower()


def build_feature_count(items: list[str]) -> int:
    return len([item for item in items if item.strip()])
```

**Expected outcome:**  
Feature logic exists in a reusable module instead of being trapped in notebook cells.

## Task 3 - Add Unit Tests

**Instructions:**

Create `tests/test_features.py`:

```python
from ml_project.features import build_feature_count, normalize_text


def test_normalize_text_strips_and_lowers():
    assert normalize_text("  Production ML  ") == "production ml"


def test_build_feature_count_ignores_empty_items():
    assert build_feature_count(["data", "", "model", "   "]) == 2
```

**Expected outcome:**  
Learners can run:

```bash
pytest
```

and see tests pass.

## Task 4 - Add a Prediction Function

**Instructions:**

Create `src/ml_project/predict.py`:

```python
def predict_label(score: float) -> str:
    if score >= 0.5:
        return "production-ready"
    return "needs-work"
```

Create `tests/test_predict.py`:

```python
from ml_project.predict import predict_label


def test_predict_label_ready():
    assert predict_label(0.8) == "production-ready"


def test_predict_label_needs_work():
    assert predict_label(0.2) == "needs-work"
```

**Assumption note:**  
This is a toy model stand-in. The learning goal is production structure and CI, not model quality.

## Task 5 - Add Requirements

**Instructions:**

Create `requirements.txt`:

```text
pytest
```

If using a `src/` layout, install the project path during CI using:

```bash
pip install -r requirements.txt
pip install -e .
```

For a minimal lab, learners may instead set `PYTHONPATH=src` in the CI command.

## Task 6 - Create GitHub Actions CI

**Instructions:**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run tests
        run: |
          PYTHONPATH=src pytest
```

**Expected outcome:**  
Every push and pull request runs the test suite automatically.

## Task 7 - Reflect on Production Readiness

**Instructions:**

Answer these questions:

1. What logic moved out of the notebook?
2. What can now be tested automatically?
3. What is still missing before this could serve real users?
4. What deployment pattern would fit this project: batch, API, embedded, or another pattern?

**Expected learner insight:**  
CI is a foundation, not the whole production system. Real deployment still needs model artifact management, environment configuration, data validation, deployment infrastructure, and monitoring.

## Lab Success Criteria

The lab is complete when:

- Project code is separated from notebook exploration.
- At least two tests pass locally.
- GitHub Actions workflow is committed.
- CI runs on push or pull request.
- Learner can explain what remains before real production deployment.

## Extension Challenge

Add a `train.py` script that writes a model artifact or placeholder artifact to `models/`, then update CI to run a training smoke test.

**Stretch reflection:**  
Should training run on every push, only on demand, or only when data/model code changes? Explain the tradeoff.
