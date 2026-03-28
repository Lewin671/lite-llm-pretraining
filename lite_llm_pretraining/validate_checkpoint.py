import argparse
import json
import re
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
    build_prompt_from_profile,
    resolve_inference_profile,
)


DEFAULT_PROMPTS = [
    "There was a little boy named Timmy who loved red kites.",
    "Lily found a shiny key under the old tree.",
    "Maya and her puppy got lost in the big park.",
]

COMMON_WORDS = {
    "a",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "day",
    "for",
    "had",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "little",
    "not",
    "of",
    "on",
    "once",
    "one",
    "said",
    "saw",
    "she",
    "that",
    "the",
    "there",
    "they",
    "time",
    "to",
    "was",
    "we",
    "went",
    "with",
    "you",
}

PROMPT_STOPWORDS = COMMON_WORDS | {
    "big",
    "found",
    "little",
    "loved",
    "named",
    "old",
    "park",
    "puppy",
    "shiny",
    "tree",
}


def prompt_feature_metrics(prompt: str, text: str):
    prompt_names = sorted(
        {
            word.lower()
            for word in re.findall(r"\b[A-Z][a-z']+\b", prompt)
            if word.lower() not in PROMPT_STOPWORDS
        }
    )
    prompt_words = [
        word
        for word in re.findall(r"[A-Za-z']+", prompt.lower())
        if len(word) >= 3 and word not in PROMPT_STOPWORDS
    ]
    prompt_content_words = sorted(
        {word for word in prompt_words if word not in set(prompt_names)}
    )
    prompt_keywords = sorted(set(prompt_names) | set(prompt_content_words))

    output_words = re.findall(r"[A-Za-z']+", text.lower())
    output_word_set = set(output_words)
    leading_output_words = set(output_words[:40])

    name_hit_count = sum(word in output_word_set for word in prompt_names)
    content_hit_count = sum(word in output_word_set for word in prompt_content_words)
    keyword_hit_count = sum(word in output_word_set for word in prompt_keywords)
    leading_keyword_hit_count = sum(word in leading_output_words for word in prompt_keywords)

    return {
        "prompt_name_count": len(prompt_names),
        "prompt_name_hit_count": name_hit_count,
        "prompt_name_hit_ratio": round(name_hit_count / max(1, len(prompt_names)), 4),
        "prompt_content_count": len(prompt_content_words),
        "prompt_content_hit_count": content_hit_count,
        "prompt_content_hit_ratio": round(
            content_hit_count / max(1, len(prompt_content_words)), 4
        ),
        "prompt_keyword_count": len(prompt_keywords),
        "prompt_keyword_hit_count": keyword_hit_count,
        "prompt_keyword_hit_ratio": round(
            keyword_hit_count / max(1, len(prompt_keywords)), 4
        ),
        "prompt_leading_keyword_hit_count": leading_keyword_hit_count,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sample-based quality checks for a saved checkpoint."
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Directory containing weights.npz and model_config.json.",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Optional dataset directory for validation loss estimation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=160,
        help="Number of new tokens to generate per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for validation generations.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Optional top-k sampling cutoff for validation generations.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Optional repetition penalty greater than 1.0 for validation generations.",
    )
    parser.add_argument(
        "--repetition_window",
        type=int,
        default=None,
        help="Optional lookback window used by repetition penalty during validation.",
    )
    parser.add_argument(
        "--eval_batches",
        type=int,
        default=10,
        help="Number of validation batches used when data_dir is provided.",
    )
    parser.add_argument(
        "--prompts_json",
        default=None,
        help="Optional JSON array of prompts. Falls back to built-in prompts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed used for deterministic validation sampling.",
    )
    return parser.parse_args()


def repeated_ngram_ratio(words, n: int = 3):
    if len(words) < n:
        return 0.0
    ngrams = [" ".join(words[idx : idx + n]) for idx in range(len(words) - n + 1)]
    unique_ngrams = len(set(ngrams))
    return 1.0 - (unique_ngrams / len(ngrams))


def max_run_length(text: str):
    if not text:
        return 0
    longest = 1
    current = 1
    previous = text[0]
    for char in text[1:]:
        if char == previous:
            current += 1
            longest = max(longest, current)
        else:
            previous = char
            current = 1
    return longest


def prompt_keyword_metrics(prompt: str, text: str):
    return prompt_feature_metrics(prompt, text)


def sample_metrics(prompt: str, text: str):
    stripped = text.strip()
    words = re.findall(r"[A-Za-z']+", stripped.lower())
    printable_chars = sum(char.isprintable() or char in "\n\r\t" for char in text)
    sentence_end_count = sum(text.count(mark) for mark in ".!?")
    unknown_marker_count = (
        text.count("⁇") + text.count("<unk>") + text.count("<|endoftext|>")
    )
    metrics = {
        "char_count": len(text),
        "word_count": len(words),
        "unique_word_ratio": round(len(set(words)) / max(1, len(words)), 4),
        "common_word_ratio": round(
            sum(word in COMMON_WORDS for word in words) / max(1, len(words)), 4
        ),
        "short_word_ratio": round(
            sum(len(word) == 1 for word in words) / max(1, len(words)), 4
        ),
        "repeated_trigram_ratio": round(repeated_ngram_ratio(words, n=3), 4),
        "printable_ratio": round(printable_chars / max(1, len(text)), 4),
        "sentence_end_count": sentence_end_count,
        "max_char_run": max_run_length(text),
        "unknown_marker_count": unknown_marker_count,
        **prompt_keyword_metrics(prompt, text),
    }
    checks = {
        "enough_words": metrics["word_count"] >= 25,
        "has_sentence_end": sentence_end_count >= 2,
        "mostly_printable": metrics["printable_ratio"] >= 0.98,
        "enough_common_words": metrics["common_word_ratio"] >= 0.22,
        "limited_short_words": metrics["short_word_ratio"] <= 0.2,
        "limited_repetition": metrics["repeated_trigram_ratio"] <= 0.35,
        "no_long_char_runs": metrics["max_char_run"] <= 8,
        "no_unknown_markers": metrics["unknown_marker_count"] == 0,
        "prompt_name_relevance": (
            metrics["prompt_name_count"] == 0 or metrics["prompt_name_hit_count"] >= 1
        ),
        "prompt_content_relevance": (
            metrics["prompt_content_count"] == 0
            or metrics["prompt_content_hit_count"] >= 1
        ),
        "prompt_early_relevance": metrics["prompt_leading_keyword_hit_count"] >= 1,
        "prompt_relevance": metrics["prompt_keyword_hit_ratio"] >= 0.5,
    }
    return metrics, checks


def load_prompts(prompts_json: str | None):
    if prompts_json is None:
        return list(DEFAULT_PROMPTS)
    payload = json.loads(prompts_json)
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError("--prompts_json must be a JSON array of strings")
    return payload


def validate_checkpoint(
    checkpoint_dir: Path,
    data_dir: Path | None = None,
    prompts=None,
    max_new_tokens: int = 160,
    temperature: float = 0.8,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
    repetition_window: int | None = None,
    eval_batches: int = 10,
    seed: int = 123,
):
    prompts = list(prompts or DEFAULT_PROMPTS)
    set_seed(seed)
    inference_profile = resolve_inference_profile(checkpoint_dir)
    report = {
        "checkpoint_dir": str(checkpoint_dir),
        "prompts": prompts,
        "seed": seed,
        "inference_profile": inference_profile,
        "samples": [],
    }

    lm = CheckpointLanguageModel(checkpoint_dir)
    for prompt in prompts:
        model_prompt = prompt
        if inference_profile.get("mode") in {"story", "qa"}:
            model_prompt = build_prompt_from_profile(prompt, inference_profile)
        output = lm.generate(
            model_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )
        metrics, checks = sample_metrics(prompt, output)
        report["samples"].append(
            {
                "prompt": prompt,
                "model_prompt": model_prompt,
                "output": output,
                "metrics": metrics,
                "checks": checks,
                "passed": all(checks.values()),
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

    report["summary"] = {
        "passed_samples": sum(1 for sample in report["samples"] if sample["passed"]),
        "total_samples": len(report["samples"]),
    }
    return report


def main():
    args = parse_args()
    report = validate_checkpoint(
        Path(args.checkpoint_dir),
        data_dir=Path(args.data_dir) if args.data_dir else None,
        prompts=load_prompts(args.prompts_json),
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
