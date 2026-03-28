import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from lite_llm_pretraining.common import (
    estimate_loss,
    load_checkpoint,
    load_json,
    load_loss_mask,
    load_memmap,
    perplexity,
    set_seed,
    token_dtype_from_meta,
)
from lite_llm_pretraining.model import CheckpointLanguageModel
from lite_llm_pretraining.story_inference import build_prompt_from_profile, resolve_inference_profile
from lite_llm_pretraining.story_inference import extract_qa_answer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint against a structured Q/A suite."
    )
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--suite_path", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--repetition_window", type=int, default=None)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def normalize_answer(text: str):
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = " ".join(text.split())
    return text


def answer_tokens(text: str):
    normalized = normalize_answer(text)
    if not normalized:
        return []
    return normalized.split()


def answer_exact_match(prediction: str, references: list[str]):
    normalized_prediction = normalize_answer(prediction)
    return any(normalized_prediction == normalize_answer(reference) for reference in references)


def answer_f1(prediction: str, reference: str):
    pred_tokens = answer_tokens(prediction)
    ref_tokens = answer_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = 0
    remaining = list(ref_tokens)
    for token in pred_tokens:
        if token in remaining:
            common += 1
            remaining.remove(token)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_metrics(prediction: str, references: list[str]):
    exact_match = answer_exact_match(prediction, references)
    token_f1 = max((answer_f1(prediction, reference) for reference in references), default=0.0)
    normalized_prediction = normalize_answer(prediction)
    contains_reference = any(
        normalize_answer(reference) in normalized_prediction
        for reference in references
        if normalize_answer(reference)
    )
    return {
        "exact_match": exact_match,
        "token_f1": round(token_f1, 4),
        "contains_reference": contains_reference,
    }


def aggregate_tag_stats(samples):
    grouped = defaultdict(lambda: {"total": 0, "passed": 0, "exact_match": 0.0, "token_f1": 0.0})
    for sample in samples:
        for tag in sample["tags"]:
            grouped[tag]["total"] += 1
            grouped[tag]["passed"] += int(sample["strict_pass"])
            grouped[tag]["exact_match"] += float(sample["answer_metrics"]["exact_match"])
            grouped[tag]["token_f1"] += sample["answer_metrics"]["token_f1"]

    summary = {}
    for tag, stats in grouped.items():
        total = max(1, stats["total"])
        summary[tag] = {
            "total": stats["total"],
            "strict_passed": stats["passed"],
            "strict_pass_rate": round(stats["passed"] / total, 4),
            "exact_match": round(stats["exact_match"] / total, 4),
            "token_f1": round(stats["token_f1"] / total, 4),
        }
    return dict(sorted(summary.items()))


def evaluate_qa_suite(
    checkpoint_dir: Path,
    suite_path: Path,
    data_dir: Path | None = None,
    max_new_tokens: int = 80,
    temperature: float = 0.3,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
    repetition_window: int | None = None,
    eval_batches: int = 10,
    seed: int = 123,
):
    suite = load_json(suite_path)
    set_seed(seed)
    inference_profile = resolve_inference_profile(checkpoint_dir)
    pass_f1_threshold = float(suite.get("pass_f1_threshold", 0.8))

    report = {
        "checkpoint_dir": str(checkpoint_dir),
        "suite_path": str(suite_path),
        "suite_name": suite.get("name"),
        "seed": seed,
        "inference_profile": inference_profile,
        "samples": [],
    }

    lm = CheckpointLanguageModel(checkpoint_dir)
    for sample in suite["samples"]:
        question = sample["question"]
        context = sample.get("context", "")
        references = list(sample.get("answers", []))
        model_prompt = build_prompt_from_profile(question, inference_profile, context=context)
        output = lm.generate(
            model_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )
        predicted_answer = extract_qa_answer(
            output,
            answer_word_limit=inference_profile.get("answer_word_limit"),
        )
        metrics = answer_metrics(predicted_answer, references)
        strict_pass = metrics["exact_match"] or metrics["token_f1"] >= pass_f1_threshold
        report["samples"].append(
            {
                "id": sample["id"],
                "tags": sample.get("tags", []),
                "question": question,
                "context": context,
                "answers": references,
                "model_prompt": model_prompt,
                "output": output,
                "predicted_answer": predicted_answer,
                "answer_metrics": metrics,
                "strict_pass": strict_pass,
            }
        )

    if data_dir is not None:
        meta = load_json(data_dir / "meta.json")
        model, model_config, state = load_checkpoint(checkpoint_dir)
        token_dtype = token_dtype_from_meta(meta)
        val_data = load_memmap(data_dir, "val", token_dtype=token_dtype)
        val_loss_mask = load_loss_mask(data_dir, "val", meta)
        val_loss = estimate_loss(
            model,
            val_data,
            batch_size=1,
            context_size=model_config["context_size"],
            steps=eval_batches,
            loss_mask_data=val_loss_mask,
        )
        report["validation"] = {
            "eval_batches": eval_batches,
            "val_loss": round(val_loss, 6),
            "val_ppl": round(perplexity(val_loss), 6),
            "checkpoint_step": state.get("step"),
            "best_val_loss": state.get("best_val_loss"),
        }

    strict_passed = sum(int(sample["strict_pass"]) for sample in report["samples"])
    exact_match_sum = sum(float(sample["answer_metrics"]["exact_match"]) for sample in report["samples"])
    token_f1_sum = sum(sample["answer_metrics"]["token_f1"] for sample in report["samples"])
    contains_reference_sum = sum(
        int(sample["answer_metrics"]["contains_reference"]) for sample in report["samples"]
    )
    total = max(1, len(report["samples"]))
    report["summary"] = {
        "strict_passed": strict_passed,
        "total_samples": len(report["samples"]),
        "strict_pass_rate": round(strict_passed / total, 4),
        "exact_match": round(exact_match_sum / total, 4),
        "token_f1": round(token_f1_sum / total, 4),
        "answer_presence_rate": round(contains_reference_sum / total, 4),
        "tag_summary": aggregate_tag_stats(report["samples"]),
    }
    return report


def main():
    args = parse_args()
    report = evaluate_qa_suite(
        checkpoint_dir=Path(args.checkpoint_dir),
        suite_path=Path(args.suite_path),
        data_dir=Path(args.data_dir) if args.data_dir else None,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        eval_batches=args.eval_batches,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
