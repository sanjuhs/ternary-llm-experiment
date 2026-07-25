from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

DATASET_ID = "roneneldan/TinyStories"
PARQUET_API = f"https://datasets-server.huggingface.co/parquet?dataset={DATASET_ID}"
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def parquet_manifest() -> list[dict[str, Any]]:
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for resumable dataset downloads")
    response = subprocess.run(
        [curl, "-4", "-fsSL", "--retry", "3", PARQUET_API],
        check=True,
        capture_output=True,
        text=True,
    )
    files = json.loads(response.stdout)["parquet_files"]
    return [
        {
            "split": item["split"],
            "filename": item["filename"],
            "url": item["url"],
            "size": int(item["size"]),
        }
        for item in files
    ]


def _download_one(item: dict[str, Any], raw_dir: Path) -> Path:
    destination = raw_dir / item["split"] / item["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = item["size"]
    if destination.exists() and destination.stat().st_size == expected_size:
        print(f"cached: {destination}")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for resumable dataset downloads")
    subprocess.run(
        [
            curl,
            "-4",
            "-fL",
            "--retry",
            "5",
            "--retry-all-errors",
            "--connect-timeout",
            "15",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            item["url"],
        ],
        check=True,
    )
    if partial.stat().st_size != expected_size:
        raise OSError(
            f"size mismatch for {partial}: got {partial.stat().st_size}, expected {expected_size}"
        )
    os.replace(partial, destination)
    return destination


def download_parquet_files(raw_dir: Path) -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = {"train": [], "validation": []}
    manifest = parquet_manifest()
    with ThreadPoolExecutor(max_workers=min(4, len(manifest))) as executor:
        paths = executor.map(lambda item: _download_one(item, raw_dir), manifest)
        for item, path in zip(manifest, paths, strict=True):
            files[item["split"]].append(path)
    return files


def load_tiny_stories(cache_dir: Path, raw_dir: Path) -> DatasetDict:
    files = download_parquet_files(raw_dir)
    dataset = load_dataset(
        "parquet",
        data_files={split: [str(path) for path in paths] for split, paths in files.items()},
        cache_dir=str(cache_dir),
    )
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"expected DatasetDict, received {type(dataset).__name__}")
    return dataset


def select_rows(dataset: Dataset, maximum: int | None) -> Dataset:
    if maximum is None:
        return dataset
    if maximum <= 0:
        raise ValueError("story limits must be positive")
    return dataset.select(range(min(maximum, len(dataset))))


def story_iterator(dataset: Dataset) -> Iterator[str]:
    for row in dataset:
        text = row["text"]
        if isinstance(text, str) and text.strip():
            yield text


def train_tokenizer(
    stories: Iterable[str],
    *,
    output_path: Path,
    vocab_size: int,
) -> Tokenizer:
    if vocab_size <= len(SPECIAL_TOKENS) + 256:
        raise ValueError("vocab_size must leave room for the byte alphabet and special tokens")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(stories, trainer=trainer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer


def write_token_stream(
    dataset: Dataset,
    tokenizer: Tokenizer,
    output_path: Path,
    *,
    chunk_tokens: int = 1_000_000,
) -> int:
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    if bos_id is None or eos_id is None:
        raise ValueError("tokenizer is missing <bos> or <eos>")
    if tokenizer.get_vocab_size() > np.iinfo(np.uint16).max:
        raise ValueError("vocabulary is too large for uint16 token shards")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer: list[int] = []
    total = 0
    with output_path.open("wb") as output:
        for text in tqdm(story_iterator(dataset), total=len(dataset), desc=output_path.stem):
            ids = [bos_id, *tokenizer.encode(text, add_special_tokens=False).ids, eos_id]
            buffer.extend(ids)
            total += len(ids)
            if len(buffer) >= chunk_tokens:
                np.asarray(buffer, dtype=np.uint16).tofile(output)
                buffer.clear()
        if buffer:
            np.asarray(buffer, dtype=np.uint16).tofile(output)
    return total


def prepare(
    *,
    cache_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    max_train_stories: int | None,
    max_validation_stories: int | None,
    vocab_size: int,
    tokenizer_from: Path | None,
) -> dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    dataset = load_tiny_stories(cache_dir, raw_dir)
    train = select_rows(dataset["train"], max_train_stories)
    validation = select_rows(dataset["validation"], max_validation_stories)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    if tokenizer_from is None:
        tokenizer = train_tokenizer(
            story_iterator(train),
            output_path=tokenizer_path,
            vocab_size=vocab_size,
        )
        tokenizer_source = "trained_on_selected_train_split"
    else:
        if not tokenizer_from.exists():
            raise FileNotFoundError(f"tokenizer not found: {tokenizer_from}")
        tokenizer = Tokenizer.from_file(str(tokenizer_from))
        tokenizer.save(str(tokenizer_path))
        tokenizer_source = str(tokenizer_from)
    train_tokens = write_token_stream(train, tokenizer, output_dir / "train.bin")
    validation_tokens = write_token_stream(validation, tokenizer, output_dir / "validation.bin")
    metadata = {
        "dataset": DATASET_ID,
        "selection": {
            "train_stories": len(train),
            "validation_stories": len(validation),
            "strategy": "first_n",
        },
        "tokenizer": {
            "type": "byte_level_bpe",
            "path": tokenizer_path.name,
            "vocab_size": tokenizer.get_vocab_size(),
            "source": tokenizer_source,
            "special_tokens": {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS},
        },
        "tokens": {
            "dtype": "uint16",
            "train": train_tokens,
            "validation": validation_tokens,
        },
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download and cache TinyStories")
    download.add_argument("--raw-dir", type=Path, default=Path("data/raw"))

    prepare_parser = subparsers.add_parser(
        "prepare", help="train a tokenizer and write token streams"
    )
    prepare_parser.add_argument("--cache-dir", type=Path, default=Path("data/huggingface"))
    prepare_parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    prepare_parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    prepare_parser.add_argument("--max-train-stories", type=int, default=50_000)
    prepare_parser.add_argument("--max-validation-stories", type=int, default=2_000)
    prepare_parser.add_argument("--vocab-size", type=int, default=4_096)
    prepare_parser.add_argument(
        "--tokenizer-from",
        type=Path,
        help="reuse an existing tokenizer instead of training a new one",
    )
    prepare_parser.add_argument(
        "--full",
        action="store_true",
        help="prepare all train and validation stories (download is always complete)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "download":
        files = download_parquet_files(args.raw_dir)
        summary = {
            split: {
                "files": len(paths),
                "bytes": sum(path.stat().st_size for path in paths),
            }
            for split, paths in files.items()
        }
        print(json.dumps({"dataset": DATASET_ID, "download": summary}, indent=2))
        return

    max_train = None if args.full else args.max_train_stories
    max_validation = None if args.full else args.max_validation_stories
    metadata = prepare(
        cache_dir=args.cache_dir,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        max_train_stories=max_train,
        max_validation_stories=max_validation,
        vocab_size=args.vocab_size,
        tokenizer_from=args.tokenizer_from,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
