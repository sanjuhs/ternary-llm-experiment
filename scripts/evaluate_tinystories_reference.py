#!/usr/bin/env python3
"""Evaluate a released Hugging Face TinyStories model on the official split."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def stories(paths: list[Path]) -> Iterator[str]:
    for path in paths:
        table = pq.read_table(path, columns=["text"])
        yield from table.column("text").to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation-parquet", type=Path, nargs="+", required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
    ).to(args.device)
    model.eval()
    boundary_id = tokenizer.eos_token_id
    if boundary_id is None:
        raise ValueError("reference tokenizer does not define an EOS token")

    token_stream: list[int] = []
    utf8_bytes = 0
    story_count = 0
    for text in stories(args.validation_parquet):
        utf8_bytes += len(text.encode("utf-8"))
        story_count += 1
        token_stream.append(boundary_id)
        token_stream.extend(tokenizer.encode(text, add_special_tokens=False))
        token_stream.append(boundary_id)

    window = args.context_length
    starts = list(range(0, len(token_stream) - window - 1, window))
    if args.max_batches is not None:
        starts = starts[: args.max_batches * args.batch_size]
    total_nll = 0.0
    total_targets = 0
    with torch.no_grad():
        for batch_start in range(0, len(starts), args.batch_size):
            offsets = starts[batch_start : batch_start + args.batch_size]
            inputs = torch.tensor(
                [token_stream[offset : offset + window] for offset in offsets],
                dtype=torch.long,
                device=args.device,
            )
            targets = torch.tensor(
                [token_stream[offset + 1 : offset + window + 1] for offset in offsets],
                dtype=torch.long,
                device=args.device,
            )
            logits = model(inputs).logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            )
            total_nll += float(loss.item())
            total_targets += targets.numel()

    loss = total_nll / total_targets
    estimated_full_nll = loss * (len(token_stream) - 1)
    result = {
        "model": str(args.model),
        "stories": story_count,
        "utf8_bytes": utf8_bytes,
        "context_length": window,
        "stream_tokens": len(token_stream),
        "evaluated_tokens": total_targets,
        "coverage": total_targets / (len(token_stream) - 1),
        "loss_nats_per_token": loss,
        "perplexity": math.exp(loss),
        "estimated_bits_per_utf8_byte": estimated_full_nll
        / (utf8_bytes * math.log(2.0)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
