import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Config, Tokenizer

filename_sets = {
    "english_partition": "en/*.jsonl",
    "italian_partition": "it/*.jsonl",
    # "code_partition": "..." # TODO: consider even language as code language
}

def prepare_full(
    source_path: Path, tokenizer_path: Path, destination_path: Path, chunk_size: int, match: str = ""
) -> None:
    """Prepare the "Red Pajama" dataset using the original tokenizer."""

    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)

    for set_name, pattern in filename_sets.items():
        if match and match not in set_name:
            continue

        filenames = glob.glob(os.path.join(source_path, pattern), recursive=True)

        if not filenames:
            raise RuntimeError(
                f"No files matching {pattern} found at {source_path}. \nMake sure you download the data, e.g. wget -i"
                " https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt or through"
                " \nhttps://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T"
                " \nhttps://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample \n"
            )

        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=set_name,
            chunk_size=chunk_size,
            sep_token=tokenizer.eos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
        )

        for name in filenames:
            filepath = source_path / name

            print(f"Processing {name}")

            with open(filepath, encoding="utf-8") as f:
                for row in tqdm(f):
                    text = json.loads(row)["text"]
                    text_ids = tokenizer.encode(text)
                    builder.add_array(np.array(text_ids, dtype=builder.dtype))

        builder.write_reminder()


def prepare(
    source_path: Path = Path("data/CulturaX"),
    checkpoint_dir: Path = Path("checkpoints/CulturaX_tokenizer"),
    destination_path: Path = Path("data_tokenized/lit_CulturaX"),
    match: str = "",    
) -> None:
    """Prepare the "Red Pajama" dataset. We assume tokenizer has been trained."""
    config = Config.from_checkpoint(checkpoint_dir)

    prepare_full(
        source_path=source_path,
        checkpoint_dir=checkpoint_dir,
        destination_path=destination_path,
        chunk_size=(config.block_size + 1) * 1024,  # block size + 1 for causal, 1024 blocks
        match=match,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
