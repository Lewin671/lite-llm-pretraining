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
from lite_llm_pretraining.story_inference import (
    PLAIN_STORY_TEMPLATE,
    build_story_prompt,
    resolve_inference_profile,
)
from lite_llm_pretraining.validate_checkpoint import sample_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint against a structured TinyStories prompt suite."
    )
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--suite_path", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--repetition_window", type=int, default=None)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def normalize_terms(terms):
    return [term.lower() for term in terms]


def anchor_metrics(anchors, text: str, early_anchor_word_budget: int):
    output_words = re.findall(r"[A-Za-z']+", text.lower())
    output_word_set = set(output_words)
    leading_word_set = set(output_words[:early_anchor_word_budget])

    groups = {}
    total_groups = 0
    hit_groups = 0
    all_anchor_terms = []
    for group_name, group_terms in anchors.items():
        normalized_terms = normalize_terms(group_terms)
        all_anchor_terms.extend(normalized_terms)
        matched_terms = sorted(term for term in normalized_terms if term in output_word_set)
        hit = len(matched_terms) > 0
        total_groups += 1
        hit_groups += int(hit)
        groups[group_name] = {
            "expected_terms": normalized_terms,
            "matched_terms": matched_terms,
            "hit": hit,
        }

    early_anchor_hit = any(term in leading_word_set for term in all_anchor_terms)
    return {
        "groups": groups,
        "anchor_group_count": total_groups,
        "anchor_group_hit_count": hit_groups,
        "anchor_group_hit_ratio": round(hit_groups / max(1, total_groups), 4),
        "early_anchor_hit": early_anchor_hit,
    }


def quality_checks_only(checks):
    return {
        key: value
        for key, value in checks.items()
        if not key.startswith("prompt_")
    }


def aggregate_tag_stats(samples):
    grouped = defaultdict(lambda: {"total": 0, "strict_passed": 0, "anchor_groups_hit": 0.0})
    for sample in samples:
        for tag in sample["tags"]:
            grouped[tag]["total"] += 1
            grouped[tag]["strict_passed"] += int(sample["strict_pass"])
            grouped[tag]["anchor_groups_hit"] += sample["anchor_metrics"][
                "anchor_group_hit_ratio"
            ]

    tag_summary = {}
    for tag, stats in grouped.items():
        tag_summary[tag] = {
            "total": stats["total"],
            "strict_passed": stats["strict_passed"],
            "strict_pass_rate": round(stats["strict_passed"] / max(1, stats["total"]), 4),
            "avg_anchor_group_hit_ratio": round(
                stats["anchor_groups_hit"] / max(1, stats["total"]),
                4,
            ),
        }
    return dict(sorted(tag_summary.items()))


def evaluate_prompt_suite(
    checkpoint_dir: Path,
    suite_path: Path,
    data_dir: Path | None = None,
    max_new_tokens: int = 160,
    temperature: float = 0.5,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
    repetition_window: int | None = None,
    eval_batches: int = 10,
    seed: int = 123,
):
    suite = load_json(suite_path)
    set_seed(seed)
    inference_profile = resolve_inference_profile(checkpoint_dir)
    early_anchor_word_budget = suite.get("early_anchor_word_budget", 40)

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
        prompt = sample["prompt"]
        model_prompt = prompt
        if inference_profile.get("mode") == "story":
            model_prompt = build_story_prompt(
                prompt,
                inference_profile.get("prompt_template", PLAIN_STORY_TEMPLATE),
            )
        output = lm.generate(
            model_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )
        metrics, checks = sample_metrics(prompt, output)
        anchor_eval = anchor_metrics(
            sample["anchors"],
            output,
            early_anchor_word_budget=early_anchor_word_budget,
        )
        quality_checks = quality_checks_only(checks)
        strict_pass = (
            all(quality_checks.values())
            and anchor_eval["anchor_group_hit_count"] == anchor_eval["anchor_group_count"]
            and anchor_eval["early_anchor_hit"]
        )
        report["samples"].append(
            {
                "id": sample["id"],
                "tags": sample.get("tags", []),
                "prompt": prompt,
                "model_prompt": model_prompt,
                "anchors": sample["anchors"],
                "output": output,
                "metrics": metrics,
                "quality_checks": quality_checks,
                "anchor_metrics": anchor_eval,
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
    anchor_group_hits = sum(
        sample["anchor_metrics"]["anchor_group_hit_count"] for sample in report["samples"]
    )
    anchor_group_total = sum(
        sample["anchor_metrics"]["anchor_group_count"] for sample in report["samples"]
    )
    early_anchor_hits = sum(
        int(sample["anchor_metrics"]["early_anchor_hit"]) for sample in report["samples"]
    )
    report["summary"] = {
        "strict_passed": strict_passed,
        "total_samples": len(report["samples"]),
        "strict_pass_rate": round(strict_passed / max(1, len(report["samples"])), 4),
        "anchor_group_hit_ratio": round(anchor_group_hits / max(1, anchor_group_total), 4),
        "early_anchor_hit_rate": round(
            early_anchor_hits / max(1, len(report["samples"])),
            4,
        ),
        "tag_summary": aggregate_tag_stats(report["samples"]),
    }
    return report


def main():
    args = parse_args()
    report = evaluate_prompt_suite(
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
