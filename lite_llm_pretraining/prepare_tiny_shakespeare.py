import argparse
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from lite_llm_pretraining.common import save_json


DATASET_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Tiny Shakespeare and prepare a byte-level dataset."
    )
    parser.add_argument(
        "--out_dir",
        default="data/tinyshakespeare-byte",
        help="Directory for the prepared dataset.",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Fraction of tokens used for training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "input.txt"
    urlretrieve(DATASET_URL, raw_path)

    raw_text = raw_path.read_text(encoding="utf-8")
    token_ids = np.frombuffer(raw_text.encode("utf-8"), dtype=np.uint8).astype(np.uint16)

    split_idx = int(len(token_ids) * args.train_split)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")

    save_json(
        out_dir / "meta.json",
        {
            "dataset": "tinyshakespeare",
            "tokenizer": "utf8-bytes",
            "vocab_size": 256,
            "train_tokens": int(train_ids.size),
            "val_tokens": int(val_ids.size),
            "source_url": DATASET_URL,
        },
    )

    print(f"saved dataset to {out_dir}")
    print(f"train tokens: {train_ids.size}")
    print(f"val tokens: {val_ids.size}")


if __name__ == "__main__":
    main()

