import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from tqdm import tqdm

MAGIC = 20240520
VERSION = 1
HEADER_INTS = 256

def _write_shard(out_path: Path, tokens_u16: np.ndarray):
    assert tokens_u16.dtype == np.uint16
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = int(tokens_u16.size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header.tobytes(order="C"))
        f.write(tokens_u16.tobytes(order="C"))

def _flush(buf: np.ndarray,
           out_dir: Path,
           prefix: str,
           split: str,
           shard_idx: int,
           align_multiple_of: Optional[int],
           eos_id: int) -> int:
    if buf.size == 0:
        return shard_idx
    if align_multiple_of and align_multiple_of > 0:
        r = buf.size % align_multiple_of
        if r:
            pad = align_multiple_of - r
            buf = np.concatenate([buf, np.full(pad, eos_id, dtype=np.uint16)], axis=0)
    _write_shard(out_dir / f"{prefix}_{split}_{shard_idx:05d}.bin", buf)
    return shard_idx + 1

def build_bins_from_dataset(
    dataset: Union[str, Dataset, DatasetDict],
    splits: Optional[List[str]],
    tokenizer: PreTrainedTokenizerBase,
    out_dir: Union[str, Path],
    prefix: str = "shard",
    shard_size_tokens: int = 50_000_000,
    eos_id: Optional[int] = None,
    align_multiple_of: int = 0,
    text_key: str = "text",
    batch_size_texts: int = 2048,
    streaming: bool = False,
):
    """
    Build FineWeb-style .bin shards from a HuggingFace dataset using a provided tokenizer.
    """
    out_dir = Path(out_dir)

    # Resolve tokenizer + EOS
    if eos_id is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            raise ValueError("Please provide eos_id or use a tokenizer with eos_token_id.")

    # Ensure token ids will fit uint16 (matches your loader)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None and vocab_size >= 65536:
        raise ValueError(
            f"Tokenizer vocab_size={vocab_size} >= 65536; your dataloader expects uint16 tokens."
        )

    # Resolve dataset & splits
    if isinstance(dataset, str):
        # Load by name/path
        if splits is None:
            # default common splits if present
            splits = ["train"]
        ds_dict = {}
        for sp in splits:
            ds_dict[sp] = load_dataset(dataset, split=sp, streaming=streaming)
    elif isinstance(dataset, DatasetDict):
        ds_dict = {sp: dataset[sp] for sp in (splits or list(dataset.keys()))}
    elif isinstance(dataset, Dataset):
        ds_dict = {splits[0] if splits else "train": dataset}
    else:
        raise TypeError("dataset must be str | datasets.Dataset | datasets.DatasetDict")

    pad_to = align_multiple_of if align_multiple_of > 0 else None

    for split, ds in ds_dict.items():
        # Use iterable to keep memory low / enable streaming-like flow
        if not streaming and hasattr(ds, "to_iterable_dataset"):
            ds = ds.to_iterable_dataset(num_shards=1)

        shard_idx = 0
        buf = np.empty(0, dtype=np.uint16)
        batch: List[str] = []

        pbar = tqdm(desc=f"Tokenizing {split}", unit="docs")

        def process_batch(texts: List[str]):
            nonlocal buf, shard_idx
            if not texts:
                return
            enc = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)
            flat_ids: List[int] = []
            max_id = 0
            for ids in enc["input_ids"]:
                if ids:
                    max_id = max(max_id, max(ids))
                    flat_ids.extend(ids)
                flat_ids.append(eos_id)
            if max_id >= 65536:
                raise ValueError(
                    "Found token id >= 65536; cannot fit uint16. Use another tokenizer or change loader."
                )
            arr16 = np.asarray(flat_ids, dtype=np.uint16)

            # Append and emit full shards
            if buf.size == 0:
                buf = arr16
            else:
                buf = np.concatenate([buf, arr16], axis=0)

            while buf.size >= shard_size_tokens:
                to_write = buf[:shard_size_tokens].copy()
                _write_shard(out_dir / f"{prefix}_{split}_{shard_idx:05d}.bin", to_write)
                shard_idx += 1
                buf = buf[shard_size_tokens:]

        for ex in ds:
            txt = ex.get(text_key, None) if isinstance(ex, dict) else None
            if isinstance(txt, str) and txt:
                batch.append(txt)
                pbar.update(1)
                if len(batch) >= batch_size_texts:
                    process_batch(batch)
                    batch = []

        pbar.close()
        if batch:
            process_batch(batch)

        shard_idx = _flush(buf, out_dir, prefix, split, shard_idx, pad_to, eos_id)
        print(f"[{split}] wrote {shard_idx} shards to {out_dir}")

def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="roneneldan/TinyStories",
                    help="HF dataset name/path (or local path).")
    ap.add_argument("--splits", default="train,validation",
                    help="Comma-separated splits to export, e.g. 'train' or 'train,validation'.")
    ap.add_argument("--tokenizer", default="gpt2",
                    help="HF tokenizer name/path.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", default="tiny")
    ap.add_argument("--shard_size_tokens", type=int, default=50_000_000)
    ap.add_argument("--eos_id", type=int, default=None)
    ap.add_argument("--align_multiple_of", type=int, default=0,
                    help="If >0, pad each shard's token count to this multiple (e.g., world_size*train_bs*seq_len*grad_accum).")
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--batch_size_texts", type=int, default=2048)
    ap.add_argument("--streaming", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    build_bins_from_dataset(
        dataset=args.dataset,
        splits=splits,
        tokenizer=tok,
        out_dir=args.out_dir,
        prefix=args.prefix,
        shard_size_tokens=args.shard_size_tokens,
        eos_id=(None if args.eos_id is None else int(args.eos_id)),
        align_multiple_of=args.align_multiple_of,
        text_key=args.text_key,
        batch_size_texts=args.batch_size_texts,
        streaming=bool(args.streaming),
    )

if __name__ == "__main__":
    _cli()
