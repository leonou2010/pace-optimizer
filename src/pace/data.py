"""Dataset loading and preparation for PACE training.

Training consumes a tokenized dataset saved with HuggingFace ``datasets``
(``save_to_disk``) under ``<path>/train`` and ``<path>/val``, where each example
has ``input_ids``, ``attention_mask``, and ``labels`` (with non-assistant tokens
masked to ``-100``), padded to a fixed ``max_seq_len``.

``prepare_sft`` builds such a dataset from a HuggingFace chat dataset (e.g.
``HuggingFaceTB/smol-smoltalk``). ``load_prepared`` loads an existing one.

The tokenization here is byte-for-byte the recipe used for the paper's SFT runs:
each turn is rendered as hand-rolled ChatML ``<|im_start|>{role}\n{content}<|im_end|>\n``
with the header and body tokenized separately and ``add_special_tokens=False`` (so no
BOS is prepended and no system turn is injected); labels are ``-100`` everywhere
except the assistant turn's body (the assistant *header* is masked too). The paper
SmolLM2/smol-smoltalk data uses ``max_seq_len=1280``, ``subset_ratio=0.5``,
``seed=42``, and the dataset's native ``train`` / ``test`` splits.
"""

import os
import random
from typing import Optional, Tuple

import torch
from datasets import load_from_disk


def collate_fn(batch):
    input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    attention_mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_prepared(path: str, max_samples: Optional[int] = None) -> Tuple:
    """Load a prepared dataset saved under ``<path>/train`` and ``<path>/val``."""
    train_ds = load_from_disk(os.path.join(path, "train"))
    val_ds = load_from_disk(os.path.join(path, "val"))
    if max_samples:
        train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
        val_ds = val_ds.select(range(min(max_samples, len(val_ds))))
    return train_ds, val_ds


def _set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)


def tokenize_conversation(messages, tokenizer, max_seq_len):
    """Tokenize a conversation with response-only masking (hand-rolled ChatML).

    Each turn renders to ``<|im_start|>{role}\\n`` (header) + ``{content}<|im_end|>\\n``
    (body), tokenized separately with ``add_special_tokens=False``. The header is
    always masked; the body is supervised only for assistant turns.
    """
    input_ids, labels = [], []
    for msg in messages:
        role = msg.get("role") or msg.get("from")
        content = msg.get("content") or msg.get("value")
        # ShareGPT-style role names map onto the ChatML roles.
        is_assistant = role in ("assistant", "gpt")

        header_ids = tokenizer.encode(f"<|im_start|>{role}\n", add_special_tokens=False)
        body_ids = tokenizer.encode(f"{content}<|im_end|>\n", add_special_tokens=False)

        input_ids.extend(header_ids)
        labels.extend([-100] * len(header_ids))

        input_ids.extend(body_ids)
        labels.extend(body_ids if is_assistant else [-100] * len(body_ids))

    input_ids = input_ids[:max_seq_len]
    labels = labels[:max_seq_len]
    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask, labels


def _build_split(rows, tokenizer, max_seq_len, pad_id, name):
    """Tokenize, filter, and pad a list of conversation rows into a Dataset."""
    from datasets import Dataset

    out_ids, out_attn, out_labels, skipped = [], [], [], 0
    for ex in rows:
        messages = ex.get("messages") or ex.get("conversations")
        # Skip rows with no assistant content to supervise (matches the paper recipe).
        if not messages or not any(
            (m.get("role") or m.get("from")) in ("assistant", "gpt") for m in messages
        ):
            skipped += 1
            continue
        ids, attn, labels = tokenize_conversation(messages, tokenizer, max_seq_len)
        if len(ids) < 10 or all(l == -100 for l in labels):
            skipped += 1
            continue
        out_ids.append(ids)
        out_attn.append(attn)
        out_labels.append(labels)

    # Pad to a fixed length for efficient batching.
    for i in range(len(out_ids)):
        pad = max_seq_len - len(out_ids[i])
        out_ids[i] = out_ids[i] + [pad_id] * pad
        out_attn[i] = out_attn[i] + [0] * pad
        out_labels[i] = out_labels[i] + [-100] * pad

    print(f"  {name}: kept {len(out_ids)}, skipped {skipped}")
    return Dataset.from_dict(
        {"input_ids": out_ids, "attention_mask": out_attn, "labels": out_labels}
    )


def prepare_sft(
    out_dir: str,
    tokenizer_name: str,
    dataset_name: str = "HuggingFaceTB/smol-smoltalk",
    max_seq_len: int = 1280,
    subset_ratio: float = 0.5,
    seed: int = 42,
    max_train: int | None = None,
    max_val: int | None = None,
) -> str:
    """Build a tokenized, response-masked SFT dataset and save it to ``out_dir``.

    Reproduces the paper's SFT data exactly (see module docstring). With the
    defaults (``max_seq_len=1280``, ``subset_ratio=0.5``, ``seed=42``,
    ``dataset_name='HuggingFaceTB/smol-smoltalk'``) and a SmolLM2 tokenizer this
    yields the same 113832-train / 11986-val tokenized split used for the figures.

    ``subset_ratio`` takes a fixed, seeded ``random.sample`` fraction of the dataset's
    native ``train`` / ``test`` splits. ``max_train`` / ``max_val`` instead stream a
    deterministic first-N slice (avoiding a full download) for a self-contained test;
    the per-conversation tokenization is identical in both cases.
    """
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if max_train:
        # Streamed slice: take the first-N rows of each native split (deterministic,
        # avoids a full download). Tokenization is identical to the full path.
        n_val = max_val or max(1, max_train // 4)
        train_stream = load_dataset(dataset_name, split="train", streaming=True)
        train_rows = [r for _, r in zip(range(max_train), train_stream)]
        try:
            val_stream = load_dataset(dataset_name, split="test", streaming=True)
        except Exception:
            val_stream = load_dataset(dataset_name, split="train", streaming=True)
        val_rows = [r for _, r in zip(range(n_val), val_stream)]
    else:
        ds = load_dataset(dataset_name)
        train_split = ds["train"]
        if "test" in ds:
            val_split = ds["test"]
        elif "validation" in ds:
            val_split = ds["validation"]
        else:
            val_split = None
        # Fixed seeded subsets of the native splits (exact paper recipe: sample
        # train first, then val, after seeding the `random` module).
        _set_seed(seed)
        n_train = int(len(train_split) * subset_ratio)
        train_idx = sorted(random.sample(range(len(train_split)), n_train))
        if val_split is not None:
            n_val = int(len(val_split) * subset_ratio)
            val_idx = sorted(random.sample(range(len(val_split)), n_val))
            train_rows = train_split.select(train_idx)
            val_rows = val_split.select(val_idx)
        else:
            # No native eval split: hold out a fraction of train.
            n_val = max(1, int(len(train_split) * subset_ratio * 0.05))
            remaining = sorted(set(range(len(train_split))) - set(train_idx))
            val_idx = sorted(random.sample(remaining, min(n_val, len(remaining))))
            train_rows = train_split.select(train_idx)
            val_rows = train_split.select(val_idx)

    sample = train_rows[0] if len(train_rows) else {}
    if not (sample.get("messages") or sample.get("conversations")):
        raise ValueError(
            "prepare_sft expects a chat/SFT dataset with a 'messages' or "
            "'conversations' field; raw-text / pretraining corpora (e.g. FineWeb) "
            "are not supported by the shipped SFT trainer."
        )

    pad_id = tok.pad_token_id
    train_ds = _build_split(train_rows, tok, max_seq_len, pad_id, "train")
    val_ds = _build_split(val_rows, tok, max_seq_len, pad_id, "val")
    train_ds.save_to_disk(os.path.join(out_dir, "train"))
    val_ds.save_to_disk(os.path.join(out_dir, "val"))
    return out_dir
