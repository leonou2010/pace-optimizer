"""Dataset loading and preparation for PACE training.

Training consumes a tokenized dataset saved with HuggingFace ``datasets``
(``save_to_disk``) under ``<path>/train`` and ``<path>/val``, where each example
has ``input_ids``, ``attention_mask``, and ``labels`` (with prompt tokens masked
to ``-100``), padded to a fixed ``max_seq_len``.

``prepare_sft`` builds such a dataset from a HuggingFace chat dataset (e.g.
``HuggingFaceTB/smol-smoltalk``). ``load_prepared`` loads an existing one.
"""

import os
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


def prepare_sft(
    out_dir: str,
    tokenizer_name: str,
    dataset_name: str = "HuggingFaceTB/smol-smoltalk",
    max_seq_len: int = 1280,
    val_size: int = 6000,
    seed: int = 42,
    max_train: int | None = None,
    max_val: int | None = None,
) -> str:
    """Build a tokenized, prompt-masked SFT dataset and save it to ``out_dir``.

    Each conversation is rendered with the tokenizer's chat template; tokens that
    belong to the user turns are masked to ``-100`` so the loss is computed only
    on the assistant responses. Sequences are padded/truncated to ``max_seq_len``.

    ``max_train`` / ``max_val`` cap the number of examples (streamed, so only a
    small slice is downloaded) — handy for a quick self-contained smoke test.
    """
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        # Base-model tokenizers often have no chat template; use a ChatML default
        # so SFT conversations can be rendered.
        tok.chat_template = (
            "{% for message in messages %}"
            "{% if loop.first and messages[0]['role'] != 'system' %}"
            "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )

    if max_train:
        # Stream a tiny slice instead of downloading the whole corpus.
        n_val = max_val or max(1, max_train // 4)
        stream = load_dataset(dataset_name, split="train", streaming=True)
        rows = [r for _, r in zip(range(max_train + n_val), stream)]
        train_raw = Dataset.from_list(rows[:max_train])
        val_raw = Dataset.from_list(rows[max_train:max_train + n_val])
    else:
        ds = load_dataset(dataset_name)
        if "validation" in ds:
            train_raw, val_raw = ds["train"], ds["validation"]
        else:
            split = ds["train"].train_test_split(test_size=val_size, seed=seed)
            train_raw, val_raw = split["train"], split["test"]

    def _render(example):
        msgs = example.get("messages") or example.get("conversations")
        # Render incrementally: tokenize the conversation prefix after each turn and
        # take the newly added tokens as that turn's segment. This keeps any template
        # preamble (e.g. a system header on the first turn) attached correctly and
        # lets us supervise only the assistant turns.
        input_ids, labels, prev_len, convo = [], [], 0, []
        for msg in msgs:
            role = msg.get("role") or msg.get("from")
            content = msg.get("content") or msg.get("value")
            convo.append({"role": role, "content": content})
            # render to text, then tokenize -> plain list[int] (avoids tokenizers.Encoding)
            text = tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            seg = ids[prev_len:]
            prev_len = len(ids)
            input_ids.extend(seg)
            labels.extend(seg if role in ("assistant", "gpt") else [-100] * len(seg))
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
        attn = [1] * len(input_ids)
        pad = max_seq_len - len(input_ids)
        if pad > 0:
            input_ids += [tok.pad_token_id] * pad
            labels += [-100] * pad
            attn += [0] * pad
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    cols = train_raw.column_names
    train_tok = train_raw.map(_render, remove_columns=cols, desc="tokenizing train")
    val_tok = val_raw.map(_render, remove_columns=cols, desc="tokenizing val")
    train_tok.save_to_disk(os.path.join(out_dir, "train"))
    val_tok.save_to_disk(os.path.join(out_dir, "val"))
    return out_dir
