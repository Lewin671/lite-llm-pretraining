import argparse
import re
import tempfile
from pathlib import Path

import numpy as np
import sentencepiece as spm

from lite_llm_pretraining.common import load_json, save_json
from lite_llm_pretraining.prepare_tinystories import prepare_dataset as prepare_byte_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare TinyStories with a SentencePiece tokenizer."
    )
    parser.add_argument(
        "--out_dir",
        default="data/tinystories-sp",
        help="Directory for the prepared tokenized dataset.",
    )
    parser.add_argument(
        "--byte_data_dir",
        default="data/tinystories-byte-clean",
        help="Directory containing cleaned TinyStories UTF-8 byte data.",
    )
    parser.add_argument(
        "--source_data_dir",
        dest="byte_data_dir",
        help="Legacy alias for --byte_data_dir.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=2048,
        help="SentencePiece vocabulary size.",
    )
    parser.add_argument(
        "--model_type",
        default="bpe",
        choices=["bpe", "unigram"],
        help="SentencePiece model type.",
    )
    parser.add_argument(
        "--byte_fallback",
        action="store_true",
        help="Enable SentencePiece byte fallback to reduce unknown characters.",
    )
    parser.add_argument(
        "--input_sentence_size",
        type=int,
        default=200000,
        help="Optional number of sampled sentences used for tokenizer training.",
    )
    parser.add_argument(
        "--max_sentence_length",
        type=int,
        default=16384,
        help="Maximum sentence length accepted during tokenizer training.",
    )
    parser.add_argument(
        "--disable_shuffle_input_sentence",
        action="store_true",
        help="Disable sentence shuffling when input_sentence_size is set.",
    )
    parser.add_argument(
        "--story_format",
        default="plain",
        choices=["plain", "prompt_continuation"],
        help="How each TinyStories sample is formatted before tokenizer training and encoding.",
    )
    parser.add_argument(
        "--prompt_sentence_count",
        type=int,
        default=1,
        help="Number of opening sentences kept in the prompt when using prompt_continuation format.",
    )
    return parser.parse_args()


def ensure_clean_byte_data(byte_data_dir: Path):
    meta_path = byte_data_dir / "meta.json"
    if meta_path.exists():
        return load_json(meta_path)
    return prepare_byte_dataset(byte_data_dir, preserve_eot_marker=False)


def train_sentencepiece(
    text_path: Path,
    out_dir: Path,
    vocab_size: int,
    model_type: str,
    byte_fallback: bool = False,
    input_sentence_size: int = 200000,
    max_sentence_length: int = 16384,
    shuffle_input_sentence: bool = True,
):
    model_prefix = out_dir / "tokenizer"
    spm.SentencePieceTrainer.train(
        input=str(text_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        normalization_rule_name="identity",
        bos_id=-1,
        pad_id=-1,
        unk_id=0,
        eos_id=1,
        byte_fallback=byte_fallback,
        input_sentence_size=input_sentence_size,
        max_sentence_length=max_sentence_length,
        shuffle_input_sentence=shuffle_input_sentence and input_sentence_size > 0,
    )
    return model_prefix.with_suffix(".model"), model_prefix.with_suffix(".vocab")


def iter_stories(text_path: Path):
    story_lines = []
    with text_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                story_lines.append(line.rstrip("\n"))
                continue
            if story_lines:
                yield "\n".join(story_lines)
                story_lines = []
        if story_lines:
            yield "\n".join(story_lines)


def split_story_sentences(story: str):
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", story.strip()) if part.strip()]


def split_prompt_continuation(story: str, prompt_sentence_count: int):
    sentences = split_story_sentences(story)
    if len(sentences) > prompt_sentence_count:
        prompt = " ".join(sentences[:prompt_sentence_count]).strip()
        continuation = " ".join(sentences[prompt_sentence_count:]).strip()
        return prompt, continuation

    words = story.split()
    if len(words) < 12:
        midpoint = max(1, len(words) // 2)
    else:
        midpoint = max(8, len(words) // 3)
    prompt = " ".join(words[:midpoint]).strip()
    continuation = " ".join(words[midpoint:]).strip()
    return prompt, continuation


def format_story(story: str, story_format: str, prompt_sentence_count: int):
    if story_format == "plain":
        return story
    prompt, continuation = split_prompt_continuation(story, prompt_sentence_count)
    return f"Prompt: {prompt}\nContinuation: {continuation}"


def write_formatted_training_corpus(
    source_path: Path,
    out_path: Path,
    story_format: str,
    prompt_sentence_count: int,
):
    with out_path.open("w", encoding="utf-8") as handle:
        for story in iter_stories(source_path):
            formatted = format_story(story, story_format, prompt_sentence_count)
            handle.write(formatted)
            handle.write("\n\n")


def encode_split(
    text_path: Path,
    out_path: Path,
    processor: spm.SentencePieceProcessor,
    story_format: str,
    prompt_sentence_count: int,
):
    eos_id = processor.eos_id()
    total_tokens = 0
    with out_path.open("wb") as handle:
        for story in iter_stories(text_path):
            formatted = format_story(story, story_format, prompt_sentence_count)
            token_ids = processor.encode(formatted, out_type=int)
            if eos_id >= 0:
                token_ids.append(eos_id)
            np.asarray(token_ids, dtype=np.uint16).tofile(handle)
            total_tokens += len(token_ids)
    return total_tokens


def prepare_dataset(
    out_dir: Path,
    byte_data_dir: Path,
    vocab_size: int = 2048,
    model_type: str = "bpe",
    byte_fallback: bool = False,
    input_sentence_size: int = 200000,
    max_sentence_length: int = 16384,
    shuffle_input_sentence: bool = True,
    story_format: str = "plain",
    prompt_sentence_count: int = 1,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_clean_byte_data(byte_data_dir)

    train_text_path = byte_data_dir / "train.bin"
    val_text_path = byte_data_dir / "val.bin"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix="-tinystories-spm-train.txt",
        prefix="lite-llm-pretraining-",
        delete=False,
    ) as handle:
        formatted_train_path = Path(handle.name)
    write_formatted_training_corpus(
        train_text_path,
        formatted_train_path,
        story_format=story_format,
        prompt_sentence_count=prompt_sentence_count,
    )
    try:
        model_path, vocab_path = train_sentencepiece(
            formatted_train_path,
            out_dir,
            vocab_size=vocab_size,
            model_type=model_type,
            byte_fallback=byte_fallback,
            input_sentence_size=input_sentence_size,
            max_sentence_length=max_sentence_length,
            shuffle_input_sentence=shuffle_input_sentence,
        )
    finally:
        formatted_train_path.unlink(missing_ok=True)
    processor = spm.SentencePieceProcessor(model_file=str(model_path))

    train_tokens = encode_split(
        train_text_path,
        out_dir / "train.bin",
        processor,
        story_format=story_format,
        prompt_sentence_count=prompt_sentence_count,
    )
    val_tokens = encode_split(
        val_text_path,
        out_dir / "val.bin",
        processor,
        story_format=story_format,
        prompt_sentence_count=prompt_sentence_count,
    )

    meta = {
        "dataset": "tinystories",
        "tokenizer": {
            "name": "sentencepiece",
            "model_file": model_path.name,
            "vocab_size": processor.vocab_size(),
            "eos_token_id": processor.eos_id(),
        },
        "vocab_size": processor.vocab_size(),
        "token_dtype": "uint16",
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "source_data_dir": str(byte_data_dir),
        "model_type": model_type,
        "byte_fallback": byte_fallback,
        "input_sentence_size": input_sentence_size,
        "max_sentence_length": max_sentence_length,
        "shuffle_input_sentence": shuffle_input_sentence,
        "story_format": story_format,
        "prompt_sentence_count": prompt_sentence_count,
    }
    save_json(out_dir / "meta.json", meta)
    return meta


def main():
    args = parse_args()
    meta = prepare_dataset(
        Path(args.out_dir),
        Path(args.byte_data_dir),
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        byte_fallback=args.byte_fallback,
        input_sentence_size=args.input_sentence_size,
        max_sentence_length=args.max_sentence_length,
        shuffle_input_sentence=not args.disable_shuffle_input_sentence,
        story_format=args.story_format,
        prompt_sentence_count=args.prompt_sentence_count,
    )
    print(f"saved dataset to {args.out_dir}")
    print(f"vocab size: {meta['vocab_size']}")
    print(f"train tokens: {meta['train_tokens']}")
    print(f"val tokens: {meta['val_tokens']}")


if __name__ == "__main__":
    main()
