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


def normalize_anchor_groups(anchor_groups):
    normalized = {}
    for group_name, group_terms in anchor_groups.items():
        normalized[group_name] = normalize_terms(group_terms)
    return normalized


def resolve_anchor_spec(sample: dict, suite: dict):
    anchors = sample.get("anchors", {})
    if "required" in anchors or "optional" in anchors:
        required_groups = normalize_anchor_groups(anchors.get("required", {}))
        optional_groups = normalize_anchor_groups(anchors.get("optional", {}))
    else:
        required_groups = normalize_anchor_groups(anchors)
        optional_groups = {}

    early_anchor_groups = sample.get(
        "early_anchor_groups",
        suite.get("default_early_anchor_groups", ["name", "object"]),
    )
    return {
        "required": required_groups,
        "optional": optional_groups,
        "early_anchor_groups": list(early_anchor_groups),
    }


def term_in_text(term: str, normalized_text: str, output_word_set: set[str]):
    if " " in term:
        return bool(
            re.search(rf"(?<![A-Za-z']){re.escape(term)}(?![A-Za-z'])", normalized_text)
        )
    return term in output_word_set


def group_match_metrics(groups, normalized_text: str, output_word_set):
    summary = {}
    total_groups = 0
    hit_groups = 0
    for group_name, normalized_terms in groups.items():
        matched_terms = sorted(
            term
            for term in normalized_terms
            if term_in_text(term, normalized_text, output_word_set)
        )
        hit = len(matched_terms) > 0
        total_groups += 1
        hit_groups += int(hit)
        summary[group_name] = {
            "expected_terms": normalized_terms,
            "matched_terms": matched_terms,
            "hit": hit,
        }
    return summary, total_groups, hit_groups


def anchor_metrics(anchor_spec, text: str, early_anchor_word_budget: int):
    output_words = re.findall(r"[A-Za-z']+", text.lower())
    normalized_text = " ".join(output_words)
    output_word_set = set(output_words)
    leading_words = output_words[:early_anchor_word_budget]
    leading_text = " ".join(leading_words)
    leading_word_set = set(leading_words)

    required_groups, required_count, required_hits = group_match_metrics(
        anchor_spec["required"],
        normalized_text,
        output_word_set,
    )
    optional_groups, optional_count, optional_hits = group_match_metrics(
        anchor_spec["optional"],
        normalized_text,
        output_word_set,
    )
    early_terms = []
    for group_name in anchor_spec["early_anchor_groups"]:
        early_terms.extend(anchor_spec["required"].get(group_name, []))
        early_terms.extend(anchor_spec["optional"].get(group_name, []))
    early_anchor_hit = any(
        term_in_text(term, leading_text, leading_word_set) for term in early_terms
    )
    return {
        "required_groups": required_groups,
        "required_group_count": required_count,
        "required_group_hit_count": required_hits,
        "required_group_hit_ratio": round(required_hits / max(1, required_count), 4),
        "optional_groups": optional_groups,
        "optional_group_count": optional_count,
        "optional_group_hit_count": optional_hits,
        "optional_group_hit_ratio": round(optional_hits / max(1, optional_count), 4),
        "early_anchor_groups": list(anchor_spec["early_anchor_groups"]),
        "early_anchor_hit": early_anchor_hit,
    }


def quality_checks_only(checks):
    return {
        key: value
        for key, value in checks.items()
        if not key.startswith("prompt_")
    }


def aggregate_tag_stats(samples):
    grouped = defaultdict(
        lambda: {"total": 0, "strict_passed": 0, "required_groups_hit": 0.0}
    )
    for sample in samples:
        for tag in sample["tags"]:
            grouped[tag]["total"] += 1
            grouped[tag]["strict_passed"] += int(sample["strict_pass"])
            grouped[tag]["required_groups_hit"] += sample["anchor_metrics"][
                "required_group_hit_ratio"
            ]

    tag_summary = {}
    for tag, stats in grouped.items():
        tag_summary[tag] = {
            "total": stats["total"],
            "strict_passed": stats["strict_passed"],
            "strict_pass_rate": round(stats["strict_passed"] / max(1, stats["total"]), 4),
            "avg_required_group_hit_ratio": round(
                stats["required_groups_hit"] / max(1, stats["total"]),
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
        anchor_spec = resolve_anchor_spec(sample, suite)
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
            anchor_spec,
            output,
            early_anchor_word_budget=early_anchor_word_budget,
        )
        quality_checks = quality_checks_only(checks)
        strict_pass = (
            all(quality_checks.values())
            and anchor_eval["required_group_hit_count"] == anchor_eval["required_group_count"]
            and anchor_eval["early_anchor_hit"]
        )
        report["samples"].append(
            {
                "id": sample["id"],
                "tags": sample.get("tags", []),
                "prompt": prompt,
                "model_prompt": model_prompt,
                "anchors": anchor_spec,
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
    required_group_hits = sum(
        sample["anchor_metrics"]["required_group_hit_count"] for sample in report["samples"]
    )
    required_group_total = sum(
        sample["anchor_metrics"]["required_group_count"] for sample in report["samples"]
    )
    early_anchor_hits = sum(
        int(sample["anchor_metrics"]["early_anchor_hit"]) for sample in report["samples"]
    )
    report["summary"] = {
        "strict_passed": strict_passed,
        "total_samples": len(report["samples"]),
        "strict_pass_rate": round(strict_passed / max(1, len(report["samples"])), 4),
        "required_group_hit_ratio": round(
            required_group_hits / max(1, required_group_total),
            4,
        ),
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
