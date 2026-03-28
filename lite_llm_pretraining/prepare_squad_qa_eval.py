import argparse
import json
import random
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

from lite_llm_pretraining.common import save_json


DEFAULT_SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare compact SQuAD 2.0 Q/A eval suites for local tiny-model experiments."
    )
    parser.add_argument("--source_url", default=DEFAULT_SQUAD_DEV_URL)
    parser.add_argument("--source_path", default="data/squad/dev-v2.0.json")
    parser.add_argument("--dev_out", default="prompts/squad_qa_dev_v1.json")
    parser.add_argument("--holdout_out", default="prompts/squad_qa_holdout_v1.json")
    parser.add_argument("--dev_count", type=int, default=32)
    parser.add_argument("--holdout_count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context_char_limit", type=int, default=420)
    return parser.parse_args()


def download_if_missing(url: str, out_path: Path):
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "lite-llm-pretraining/1.0"})
    with urlopen(request) as response, out_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def trim_context_window(context: str, answer_start: int, answer_text: str, context_char_limit: int):
    answer_end = answer_start + len(answer_text)
    half_window = max(40, context_char_limit // 2)
    start = max(0, answer_start - half_window)
    end = min(len(context), answer_end + half_window)
    snippet = context[start:end]
    if start > 0:
        left_boundary = snippet.find(" ")
        if left_boundary > 0:
            snippet = snippet[left_boundary + 1 :]
    if end < len(context):
        right_boundary = snippet.rfind(" ")
        if right_boundary > 0:
            snippet = snippet[:right_boundary]
    return snippet.strip()


def unique_answers(answers):
    seen = set()
    unique = []
    for answer in answers:
        text = answer.get("text", "").strip()
        if text and text not in seen:
            unique.append(text)
            seen.add(text)
    return unique


def flatten_squad_samples(source_path: Path, context_char_limit: int):
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    samples = []
    for article in payload.get("data", []):
        title = article.get("title", "").strip()
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                if qa.get("is_impossible"):
                    continue
                answers = unique_answers(qa.get("answers", []))
                if not answers:
                    continue
                primary_answer = qa["answers"][0]
                trimmed_context = trim_context_window(
                    context,
                    int(primary_answer.get("answer_start", 0)),
                    primary_answer.get("text", ""),
                    context_char_limit=context_char_limit,
                )
                samples.append(
                    {
                        "id": qa["id"],
                        "question": qa["question"].strip(),
                        "context": trimmed_context,
                        "answers": answers,
                        "tags": [title.lower()] if title else [],
                    }
                )
    return samples


def write_suite(path: Path, name: str, samples, source_url: str, source_path: Path, context_char_limit: int):
    payload = {
        "name": name,
        "task": "qa",
        "source_url": source_url,
        "source_path": str(source_path),
        "pass_f1_threshold": 0.8,
        "context_char_limit": context_char_limit,
        "samples": samples,
    }
    save_json(path, payload)


def main():
    args = parse_args()
    source_path = Path(args.source_path)
    download_if_missing(args.source_url, source_path)
    samples = flatten_squad_samples(source_path, context_char_limit=args.context_char_limit)
    random.seed(args.seed)
    random.shuffle(samples)

    dev_samples = samples[: args.dev_count]
    holdout_samples = samples[args.dev_count : args.dev_count + args.holdout_count]
    if len(holdout_samples) < args.holdout_count:
        raise ValueError("not enough SQuAD samples to build requested holdout suite")

    write_suite(
        Path(args.dev_out),
        "squad_qa_dev_v1",
        dev_samples,
        source_url=args.source_url,
        source_path=source_path,
        context_char_limit=args.context_char_limit,
    )
    write_suite(
        Path(args.holdout_out),
        "squad_qa_holdout_v1",
        holdout_samples,
        source_url=args.source_url,
        source_path=source_path,
        context_char_limit=args.context_char_limit,
    )
    print(f"saved {len(dev_samples)} samples to {args.dev_out}")
    print(f"saved {len(holdout_samples)} samples to {args.holdout_out}")


if __name__ == "__main__":
    main()
