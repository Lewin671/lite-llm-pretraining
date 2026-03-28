import argparse
import json
from pathlib import Path

from lite_llm_pretraining.common import load_json
from lite_llm_pretraining.evaluate_prompt_suite import evaluate_prompt_suite
from lite_llm_pretraining.evaluate_qa_suite import evaluate_qa_suite


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint against a supported suite.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--suite_path", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--repetition_window", type=int, default=None)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def suite_task(suite_path: Path):
    suite = load_json(suite_path)
    return suite.get("task", "prompt")


def evaluate_suite(
    checkpoint_dir: Path,
    suite_path: Path,
    data_dir: Path | None = None,
    max_new_tokens: int = 120,
    temperature: float = 0.5,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
    repetition_window: int | None = None,
    eval_batches: int = 10,
    seed: int = 123,
):
    task = suite_task(suite_path)
    if task == "qa":
        return evaluate_qa_suite(
            checkpoint_dir=checkpoint_dir,
            suite_path=suite_path,
            data_dir=data_dir,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            eval_batches=eval_batches,
            seed=seed,
        )
    return evaluate_prompt_suite(
        checkpoint_dir=checkpoint_dir,
        suite_path=suite_path,
        data_dir=data_dir,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
        eval_batches=eval_batches,
        seed=seed,
    )


def main():
    args = parse_args()
    report = evaluate_suite(
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
