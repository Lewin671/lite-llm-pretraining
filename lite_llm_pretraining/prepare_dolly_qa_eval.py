import argparse
import json
from pathlib import Path

import numpy as np

from lite_llm_pretraining.common import save_json
from lite_llm_pretraining.prepare_dolly_qa import (
    filter_examples,
    load_examples,
    transform_examples,
    trim_context,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a compact Dolly-based Q/A evaluation suite."
    )
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--dev_out", default="prompts/dolly_qa_simple_dev_v1.json")
    parser.add_argument("--holdout_out", default="prompts/dolly_qa_simple_holdout_v1.json")
    parser.add_argument("--dev_count", type=int, default=32)
    parser.add_argument("--holdout_count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context_word_limit", type=int, default=96)
    parser.add_argument("--suite_name_prefix", default="dolly_qa_simple")
    parser.add_argument("--allowed_categories_json", default=None)
    parser.add_argument("--min_answer_words", type=int, default=1)
    parser.add_argument("--max_answer_words", type=int, default=12)
    parser.add_argument("--max_question_words", type=int, default=24)
    parser.add_argument("--require_single_line_answer", action="store_true")
    parser.add_argument("--factoid_only", action="store_true")
    parser.add_argument("--normalize_factoid_answers", action="store_true")
    parser.add_argument("--max_normalized_answer_words", type=int, default=None)
    parser.add_argument("--pass_f1_threshold", type=float, default=0.9)
    return parser.parse_args()


def suite_payload(name: str, examples, context_word_limit: int, pass_f1_threshold: float):
    samples = []
    for index, example in enumerate(examples, start=1):
        samples.append(
            {
                "id": f"{name}-{index:03d}",
                "question": example["instruction"].strip(),
                "context": trim_context(example.get("context", ""), context_word_limit),
                "answers": [example["response"].strip()],
                "tags": [example.get("category", "unknown") or "unknown"],
            }
        )
    return {
        "name": name,
        "task": "qa",
        "pass_f1_threshold": pass_f1_threshold,
        "samples": samples,
    }


def main():
    args = parse_args()
    allowed_categories = None
    if args.allowed_categories_json:
        payload = json.loads(args.allowed_categories_json)
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise ValueError("--allowed_categories_json must be a JSON array of strings")
        allowed_categories = payload

    examples = load_examples(Path(args.source_path), allowed_categories=allowed_categories)
    examples = filter_examples(
        examples,
        min_answer_words=args.min_answer_words,
        max_answer_words=args.max_answer_words,
        max_question_words=args.max_question_words,
        require_single_line_answer=args.require_single_line_answer,
    )
    examples = transform_examples(
        examples,
        factoid_only=args.factoid_only,
        normalize_factoid_answers=args.normalize_factoid_answers,
        max_normalized_answer_words=args.max_normalized_answer_words,
    )
    required = args.dev_count + args.holdout_count
    if len(examples) < required:
        raise ValueError(f"not enough filtered Dolly examples: need {required}, got {len(examples)}")

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(examples))
    rng.shuffle(indices)
    shuffled = [examples[index] for index in indices]
    dev_examples = shuffled[: args.dev_count]
    holdout_examples = shuffled[args.dev_count : args.dev_count + args.holdout_count]

    dev_payload = suite_payload(
        f"{args.suite_name_prefix}_dev_v1",
        dev_examples,
        args.context_word_limit,
        args.pass_f1_threshold,
    )
    holdout_payload = suite_payload(
        f"{args.suite_name_prefix}_holdout_v1",
        holdout_examples,
        args.context_word_limit,
        args.pass_f1_threshold,
    )

    save_json(Path(args.dev_out), dev_payload)
    save_json(Path(args.holdout_out), holdout_payload)
    print(f"saved dev suite to {args.dev_out}")
    print(f"saved holdout suite to {args.holdout_out}")
    print(f"filtered examples: {len(examples)}")


if __name__ == "__main__":
    main()
