"""
task31_runner.py — Task 3.1: Generate structured narrative summaries.

For each sequence (Switch or Escalation event), generate a 3-part summary:
  1. CENTRAL THEME
  2. WITHIN-STATE DYNAMICS
  3. BETWEEN-STATE DYNAMICS
"""

import os
import sys
import json
from typing import List, Dict, Optional

# ── Path setup (works when run directly or imported) ──────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from prompt_builder import (
    TASK31_SYSTEM,
    build_task31_prompt,
    build_task31_prompt_with_example,
)
from llm_generator import generate_batch


# ── Main runner ───────────────────────────────────────────────

def run_task31(
    sequences: List[Dict],
    backend: str = "mock",
    model: str = "llama3:8b",
    output_dir: str = "outputs",
    few_shot: bool = False,
    max_new_tokens: int = 700,
    temperature: float = 0.3,
) -> List[Dict]:
    """
    Generate Task 3.1 summaries for all sequences.
    Returns list of result dicts with generated summary + metadata.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Building prompts for {len(sequences)} sequences...")

    prompts = []
    for i, seq in enumerate(sequences):
        # Use few-shot only if we have a prior gold example to draw from
        if few_shot and i > 0 and sequences[i - 1].get("gold_summary"):
            user_prompt = build_task31_prompt_with_example(seq, sequences[i - 1])
        else:
            user_prompt = build_task31_prompt(seq, include_posts=bool(seq.get("posts")))
        prompts.append({"system": TASK31_SYSTEM, "user": user_prompt})

    print(f"  Running generation (backend={backend}, model={model})...")
    generated_texts = generate_batch(
        prompts=prompts,
        backend=backend,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        verbose=True,
    )

    # Package results
    results = []
    for seq, gen_text in zip(sequences, generated_texts):
        result = {
            "timeline_id":      seq["timeline_id"],
            "sequence_id":      seq["sequence_id"],
            "change_type":      seq["change_type"],
            "direction":        seq.get("direction", "unknown"),
            "postids":          seq["postids"],
            "gold_summary":     seq.get("gold_summary", ""),
            "generated_summary": gen_text,
            "parsed_sections":  _parse_sections(gen_text),
        }
        results.append(result)

    # Save
    out_path = os.path.join(output_dir, "task31_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {len(results)} summaries → {out_path}")

    # Also save Codabench submission format
    _save_submission_format(results, output_dir)

    return results


def _parse_sections(text: str) -> Dict[str, str]:
    """
    Extract the three named sections from a generated summary.
    Gracefully handles missing or malformed sections.
    """
    sections = {"central_theme": "", "within_state": "", "between_state": ""}

    # Try to split on numbered headers
    import re
    patterns = [
        (r"1\.\s*CENTRAL THEME[:\n]+(.*?)(?=2\.\s*WITHIN|$)", "central_theme"),
        (r"2\.\s*WITHIN.STATE DYNAMICS[:\n]+(.*?)(?=3\.\s*BETWEEN|$)", "within_state"),
        (r"3\.\s*BETWEEN.STATE DYNAMICS[:\n]+(.*?)$", "between_state"),
    ]
    for pattern, key in patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            sections[key] = m.group(1).strip()

    # Fallback: if no headers found, dump everything into central_theme
    if not any(sections.values()):
        sections["central_theme"] = text.strip()

    return sections


def _save_submission_format(results: List[Dict], output_dir: str):
    """
    Save in Codabench submission format: one JSON file with timeline_id → summary.
    """
    submission = {}
    for r in results:
        key = f"{r['timeline_id']}_{r['sequence_id']}"
        submission[key] = {
            "timeline_id":   r["timeline_id"],
            "sequence_id":   r["sequence_id"],
            "change_type":   r["change_type"],
            "summary":       r["generated_summary"],
        }
    path = os.path.join(output_dir, "task31_submission.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)
    print(f"  Saved Codabench submission → {path}")


# ── Evaluation ────────────────────────────────────────────────

def evaluate_summaries(results: List[Dict], output_dir: str = "outputs") -> Dict:
    """
    Evaluate generated summaries against gold using ROUGE and BERTScore (if available).
    Skips silently if gold summaries are not present.
    """
    gold_texts = [r["gold_summary"] for r in results if r.get("gold_summary")]
    gen_texts  = [r["generated_summary"] for r in results if r.get("gold_summary")]

    if not gold_texts:
        print("  No gold summaries found — skipping evaluation.")
        return {}

    metrics = {}

    # ROUGE
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1_scores, r2_scores, rL_scores = [], [], []
        for gold, gen in zip(gold_texts, gen_texts):
            s = scorer.score(gold, gen)
            r1_scores.append(s["rouge1"].fmeasure)
            r2_scores.append(s["rouge2"].fmeasure)
            rL_scores.append(s["rougeL"].fmeasure)
        metrics["rouge1"] = round(sum(r1_scores) / len(r1_scores), 4)
        metrics["rouge2"] = round(sum(r2_scores) / len(r2_scores), 4)
        metrics["rougeL"] = round(sum(rL_scores) / len(rL_scores), 4)
        print(f"\n  ROUGE-1: {metrics['rouge1']}  ROUGE-2: {metrics['rouge2']}  ROUGE-L: {metrics['rougeL']}")
    except ImportError:
        print("  rouge_score not installed — skipping ROUGE. Run: pip install rouge_score")

    # BERTScore (optional, slow)
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(gen_texts, gold_texts, lang="en", verbose=False)
        metrics["bertscore_f1"] = round(F1.mean().item(), 4)
        print(f"  BERTScore F1: {metrics['bertscore_f1']}")
    except ImportError:
        pass  # BERTScore is optional

    # Save
    eval_path = os.path.join(output_dir, "task31_eval.json")
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Evaluation saved → {eval_path}")

    return metrics
