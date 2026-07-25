from pathlib import Path

import numpy as np
from datasets import Dataset

from ternary_llm.data import story_iterator, train_tokenizer, write_token_stream


def test_tokenizer_and_binary_stream_round_trip(tmp_path: Path) -> None:
    dataset = Dataset.from_dict(
        {
            "text": [
                "Once upon a time there was a little cat.",
                "The cat found a red ball and played.",
                "They went home and felt happy.",
            ]
        }
    )
    tokenizer = train_tokenizer(
        story_iterator(dataset),
        output_path=tmp_path / "tokenizer.json",
        vocab_size=300,
    )
    count = write_token_stream(dataset, tokenizer, tmp_path / "train.bin")
    tokens = np.fromfile(tmp_path / "train.bin", dtype=np.uint16)

    assert count == len(tokens)
    assert count > len(dataset)
    assert int(tokens[0]) == tokenizer.token_to_id("<bos>")
    assert (tokens == tokenizer.token_to_id("<eos>")).sum() == len(dataset)
