# CLPsych 2026 Task 3: Sequence Summaries & Dynamic Signatures

Mistral 7B + MIND/ABCD framework for modeling mental health trajectory dynamics from social media timelines.

---

## Overview

This repository implements our system for CLPsych 2026 Task 3, focused on:

* Task 3.1: Generating structured psychological summaries of timeline sequences
* Task 3.2: Extracting recurrent cross-user mental health patterns

Our approach uses zero-shot prompting with Mistral 7B and is grounded in the MIND/ABCD clinical framework.

---

## Method

### MIND/ABCD Framework

We model psychological states using six components:

* Affect (A)
* Behavior toward Self (B-S)
* Behavior toward Others (B-O)
* Cognition toward Self (C-S)
* Cognition toward Others (C-O)
* Desire (D)

Each post is represented as a combination of adaptive and maladaptive self-states with presence scores.

---

### Pipeline

1. Load sequence-level and post-level data
2. Construct structured prompts using the MIND framework
3. Generate summaries via Mistral 7B (Ollama backend)
4. Aggregate summaries to extract recurrent dynamic signatures

---

## Architecture

```
data → data_loader → prompt_builder → llm_generator → task runners → outputs
```

### Modules

* `data_loader.py` – Handles sequence and post formats
* `prompt_builder.py` – Builds MIND-grounded prompts
* `llm_generator.py` – Interface to Mistral via Ollama
* `task31_runner.py` – Summary generation
* `task32_runner.py` – Signature extraction
* `run_pipeline.py` – Main entry point

---

## Setup

```bash
bash setup.sh
pip install -r requirements.txt
ollama pull mistral
ollama serve
```

---

## Run

```bash
python run_pipeline.py \
  --data data/sample_train.json \
  --backend ollama \
  --model mistral \
  --few-shot \
  --evaluate
```

### Mock Run (no LLM)

```bash
python run_pipeline.py --mock --evaluate
```

---

## Results

| Metric              | Value |
| ------------------- | ----- |
| ABCD Tag Coverage   | 0.705 |
| Section Coverage    | 0.851 |
| Sequences Processed | 74    |

The system generates structured summaries with strong ABCD coverage and extracts six dynamic signatures (three deterioration and three improvement).

---

## Limitations

* No direct access to raw post text in the training pipeline
* Context window limitations for Task 3.2
* Zero-shot approach without fine-tuning
* Summary compression reduces long-range detail

---

## Project Structure

```
clpsych_task3/
├── data/              # Dataset (not included / partial)
├── outputs/           # Generated results (ignored)
├── src/               # Core modules
├── run_pipeline.py    # Entry point
├── setup.sh
├── requirements.txt
└── README.md
```

---

## System Report

Full technical details are available in:

```
CLPsych2026_Task3_System_Report.pdf
```

---

## Submission

* Task 3.1: Upload `outputs/task31_pred.json` to Codabench
* Task 3.2: Email `outputs/task32_submission.txt`

---

## Citation

```
@misc{clpsych2026_task3,
  title={CLPsych 2026 Task 3 System},
  author={Pawan Kumar},
  year={2026}
}
```

---

## Author

Pawan Kumar

---

## Notes

* Uses only open-source models (Mistral 7B)
* Fully compliant with data access constraints
* All inference runs locally via Ollama
