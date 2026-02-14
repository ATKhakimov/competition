from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import math
import random

import numpy as np
import polars as pl
import torch
from torch.utils.data import IterableDataset


PAD_IDX = 0
UNK_IDX = 1
MASK_IDX = 2


@dataclass
class SequenceWindow:
    customer_id: int
    event_ids: np.ndarray
    event_token: np.ndarray
    channel_token: np.ndarray
    hour_token: np.ndarray
    delta_log: np.ndarray
    valid_len: int


def _parse_event_time_expr(col: str) -> pl.Expr:
    dt = pl.col(col).str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
    return dt.dt.epoch("s")


def _tokenize_int(values: np.ndarray) -> np.ndarray:
    v = values.astype(np.int64, copy=False)
    out = np.where(v >= 0, v + 3, UNK_IDX)
    return out.astype(np.int64, copy=False)


def _hour_token_from_ts(ts: np.ndarray) -> np.ndarray:
    hours = ((ts // 3600) % 24).astype(np.int64)
    return (hours + 1).astype(np.int64, copy=False)


def _delta_log(ts: np.ndarray) -> np.ndarray:
    d = np.zeros_like(ts, dtype=np.float32)
    if len(ts) > 1:
        raw = ts[1:] - ts[:-1]
        raw = np.clip(raw, 0, 86400 * 30)
        d[1:] = np.log1p(raw.astype(np.float32))
    return d


def _iter_file_windows(
    file_path: Path,
    seq_len: int,
    min_len: int,
    stride: int,
) -> Iterable[SequenceWindow]:
    df = (
        pl.read_parquet(str(file_path))
        .select(
            [
                "customer_id",
                "event_id",
                "event_type_nm",
                "channel_indicator_sub_type",
                _parse_event_time_expr("event_dttm").alias("event_ts"),
            ]
        )
        .drop_nulls(["customer_id", "event_id", "event_ts"])
        .sort(["customer_id", "event_ts", "event_id"])
    )
    if df.height == 0:
        return

    c = df["customer_id"].to_numpy()
    eid = df["event_id"].to_numpy()
    ev = _tokenize_int(df["event_type_nm"].fill_null(-1).to_numpy())
    ch = _tokenize_int(df["channel_indicator_sub_type"].fill_null(-1).to_numpy())
    ts = df["event_ts"].to_numpy().astype(np.int64, copy=False)
    hr = _hour_token_from_ts(ts)
    dl = _delta_log(ts)

    boundaries = np.r_[0, np.flatnonzero(c[1:] != c[:-1]) + 1, len(c)]
    for bi in range(len(boundaries) - 1):
        s = int(boundaries[bi])
        e = int(boundaries[bi + 1])
        n = e - s
        if n < min_len:
            continue
        st = 0
        while st < n:
            en = min(st + seq_len, n)
            ln = en - st
            if ln >= min_len:
                ws = SequenceWindow(
                    customer_id=int(c[s]),
                    event_ids=eid[s + st : s + en].astype(np.int64, copy=False),
                    event_token=ev[s + st : s + en].astype(np.int64, copy=False),
                    channel_token=ch[s + st : s + en].astype(np.int64, copy=False),
                    hour_token=hr[s + st : s + en].astype(np.int64, copy=False),
                    delta_log=dl[s + st : s + en].astype(np.float32, copy=False),
                    valid_len=ln,
                )
                yield ws
            if en == n:
                break
            st += stride


class TransactionSequenceDataset(IterableDataset):
    def __init__(
        self,
        parquet_files: list[Path],
        seq_len: int = 128,
        min_len: int = 8,
        stride: int = 32,
        shuffle_files: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.files = list(parquet_files)
        self.seq_len = int(seq_len)
        self.min_len = int(min_len)
        self.stride = int(stride)
        self.shuffle_files = bool(shuffle_files)
        self.seed = int(seed)

    def __iter__(self):
        files = list(self.files)
        if self.shuffle_files:
            rnd = random.Random(self.seed + int(torch.randint(0, 100000, ()).item()))
            rnd.shuffle(files)
        for fp in files:
            yield from _iter_file_windows(
                file_path=fp,
                seq_len=self.seq_len,
                min_len=self.min_len,
                stride=self.stride,
            )


class FoundationCollator:
    def __init__(
        self,
        seq_len: int,
        event_vocab_size: int,
        channel_vocab_size: int,
        mask_prob: float = 0.15,
    ) -> None:
        self.seq_len = int(seq_len)
        self.event_vocab_size = int(event_vocab_size)
        self.channel_vocab_size = int(channel_vocab_size)
        self.mask_prob = float(mask_prob)

    def __call__(self, batch: list[SequenceWindow]) -> dict[str, torch.Tensor]:
        b = len(batch)
        l = self.seq_len
        event = np.full((b, l), PAD_IDX, dtype=np.int64)
        channel = np.full((b, l), PAD_IDX, dtype=np.int64)
        hour = np.full((b, l), PAD_IDX, dtype=np.int64)
        delta = np.zeros((b, l), dtype=np.float32)
        event_id = np.full((b, l), -1, dtype=np.int64)
        customer = np.zeros((b,), dtype=np.int64)
        valid_len = np.zeros((b,), dtype=np.int64)
        for i, w in enumerate(batch):
            n = min(w.valid_len, l)
            event[i, :n] = w.event_token[:n]
            channel[i, :n] = w.channel_token[:n]
            hour[i, :n] = w.hour_token[:n]
            delta[i, :n] = w.delta_log[:n]
            event_id[i, :n] = w.event_ids[:n]
            customer[i] = w.customer_id
            valid_len[i] = n

        attn = np.arange(l)[None, :] < valid_len[:, None]

        next_event = np.full((b, l), -100, dtype=np.int64)
        next_channel = np.full((b, l), -100, dtype=np.int64)
        next_delta = np.full((b, l), -1.0, dtype=np.float32)
        for i in range(b):
            n = int(valid_len[i])
            if n > 1:
                next_event[i, : n - 1] = event[i, 1:n]
                next_channel[i, : n - 1] = channel[i, 1:n]
                next_delta[i, : n - 1] = delta[i, 1:n]

        event_in = event.copy()
        channel_in = channel.copy()
        mlm_event = np.full((b, l), -100, dtype=np.int64)
        mlm_channel = np.full((b, l), -100, dtype=np.int64)
        rng = np.random.default_rng()
        for i in range(b):
            n = int(valid_len[i])
            if n <= 0:
                continue
            mask = rng.random(n) < self.mask_prob
            if not mask.any():
                mask[rng.integers(0, n)] = True
            pos = np.flatnonzero(mask)
            mlm_event[i, pos] = event[i, pos]
            mlm_channel[i, pos] = channel[i, pos]
            replace_roll = rng.random(len(pos))
            for j, p in enumerate(pos):
                r = replace_roll[j]
                if r < 0.8:
                    event_in[i, p] = MASK_IDX
                    channel_in[i, p] = MASK_IDX
                elif r < 0.9:
                    event_in[i, p] = int(rng.integers(3, max(self.event_vocab_size, 4)))
                    channel_in[i, p] = int(rng.integers(3, max(self.channel_vocab_size, 4)))

        return {
            "event_id": torch.from_numpy(event_id),
            "customer_id": torch.from_numpy(customer),
            "valid_len": torch.from_numpy(valid_len),
            "attention_mask": torch.from_numpy(attn),
            "event_token_in": torch.from_numpy(event_in),
            "channel_token_in": torch.from_numpy(channel_in),
            "hour_token": torch.from_numpy(hour),
            "delta_log": torch.from_numpy(delta),
            "next_event_target": torch.from_numpy(next_event),
            "next_channel_target": torch.from_numpy(next_channel),
            "next_delta_target": torch.from_numpy(next_delta),
            "mlm_event_target": torch.from_numpy(mlm_event),
            "mlm_channel_target": torch.from_numpy(mlm_channel),
        }


def discover_parquet_files(data_dir: Path, patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(data_dir.glob(pat)))
    return [f for f in files if f.exists()]


def estimate_vocab_sizes(parquet_files: list[Path]) -> tuple[int, int]:
    max_event = -1
    max_channel = -1
    for fp in parquet_files:
        df = pl.read_parquet(
            str(fp),
            columns=["event_type_nm", "channel_indicator_sub_type"],
        )
        if df.height == 0:
            continue
        emax = int(df["event_type_nm"].fill_null(-1).max())
        cmax = int(df["channel_indicator_sub_type"].fill_null(-1).max())
        max_event = max(max_event, emax)
        max_channel = max(max_channel, cmax)
    # +3: PAD, UNK, MASK offsets
    return max(4, max_event + 4), max(4, max_channel + 4)

