import argparse
from pathlib import Path
from urllib.request import Request, urlopen

from lite_llm_pretraining.common import save_json


TRAIN_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStories-train.txt"
)
VAL_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStories-valid.txt"
)
USER_AGENT = "lite-llm-pretraining/0.1"
CHUNK_SIZE = 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download TinyStories and prepare a byte-level dataset."
    )
    parser.add_argument(
        "--out_dir",
        default="data/tinystories-byte",
        help="Directory for the prepared dataset.",
    )
    parser.add_argument(
        "--train_url",
        default=TRAIN_URL,
        help="Source URL for the TinyStories training split.",
    )
    parser.add_argument(
        "--val_url",
        default=VAL_URL,
        help="Source URL for the TinyStories validation split.",
    )
    return parser.parse_args()


def download_bytes(url: str, out_path: Path):
    existing_size = out_path.stat().st_size if out_path.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"

    request = Request(url, headers=headers)
    mode = "ab" if existing_size else "wb"
    token_count = existing_size
    with urlopen(request) as response, out_path.open(mode) as handle:
        if existing_size and response.status == 200:
            handle.close()
            out_path.unlink(missing_ok=True)
            return download_bytes(url, out_path)
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            token_count += len(chunk)
    return token_count


def prepare_dataset(
    out_dir: Path,
    train_url: str | None = TRAIN_URL,
    val_url: str | None = VAL_URL,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_url = train_url or TRAIN_URL
    val_url = val_url or VAL_URL

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    train_tokens = download_bytes(train_url, train_path)
    val_tokens = download_bytes(val_url, val_path)

    meta = {
        "dataset": "tinystories",
        "tokenizer": "utf8-bytes",
        "vocab_size": 256,
        "token_dtype": "uint8",
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_url": train_url,
        "val_url": val_url,
    }
    save_json(out_dir / "meta.json", meta)
    return meta


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    meta = prepare_dataset(out_dir, train_url=args.train_url, val_url=args.val_url)
    print(f"saved dataset to {out_dir}")
    print(f"train tokens: {meta['train_tokens']}")
    print(f"val tokens: {meta['val_tokens']}")


if __name__ == "__main__":
    main()
