#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from competition.foundation_data import (  # noqa: E402
    FoundationCollator,
    TransactionSequenceDataset,
    discover_parquet_files,
)
from competition.foundation_model import TemporalFoundationEncoder  # noqa: E402
from competition.pipeline_config import load_config  # noqa: E402


def _pick_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _to_rows(
    event_ids: np.ndarray,
    customer_ids: np.ndarray,
    valid_len: np.ndarray,
    hidden: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat_event = []
    flat_customer = []
    flat_emb = []
    for i in range(len(valid_len)):
        n = int(valid_len[i])
        if n <= 0:
            continue
        flat_event.append(event_ids[i, :n])
        flat_customer.append(np.full((n,), customer_ids[i], dtype=np.int64))
        flat_emb.append(hidden[i, :n])
    if not flat_event:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, hidden.shape[-1]), dtype=np.float32),
        )
    return (
        np.concatenate(flat_event, axis=0),
        np.concatenate(flat_customer, axis=0),
        np.concatenate(flat_emb, axis=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export event/customer embeddings from foundation model.")
    parser.add_argument("--config", default="conf/foundation_pretrain.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", default="foundation_export")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-event-rows", type=int, default=2_000_000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    data_dir = Path(cfg["paths"]["data_dir"])
    exp_cfg = cfg["export"]
    files = discover_parquet_files(data_dir, list(exp_cfg["source_globs"]))
    if not files:
        raise RuntimeError("No export source files found")

    seq_len = int(ckpt.get("seq_len", cfg["pretrain"]["seq_len"]))
    stride = int(exp_cfg.get("stride", seq_len))
    min_len = int(exp_cfg.get("min_len", 2))
    dataset = TransactionSequenceDataset(
        parquet_files=files,
        seq_len=seq_len,
        min_len=min_len,
        stride=stride,
        shuffle_files=False,
        seed=42,
    )
    collator = FoundationCollator(
        seq_len=seq_len,
        event_vocab_size=int(ckpt["event_vocab_size"]),
        channel_vocab_size=int(ckpt["channel_vocab_size"]),
        mask_prob=0.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(exp_cfg.get("batch_size", 64)),
        num_workers=int(exp_cfg.get("num_workers", 2)),
        pin_memory=True,
        collate_fn=collator,
    )

    model = TemporalFoundationEncoder(
        event_vocab_size=int(ckpt["event_vocab_size"]),
        channel_vocab_size=int(ckpt["channel_vocab_size"]),
        hour_vocab_size=25,
        d_model=int(cfg["pretrain"]["d_model"]),
        n_heads=int(cfg["pretrain"]["n_heads"]),
        n_layers=int(cfg["pretrain"]["n_layers"]),
        ff_mult=int(cfg["pretrain"].get("ff_mult", 4)),
        dropout=float(cfg["pretrain"].get("dropout", 0.1)),
        max_seq_len=seq_len,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = _pick_device(args.device)
    model = model.to(device)

    out_dir = Path(cfg["paths"]["artifacts_dir"]) / "foundation" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    customer_sum: dict[int, np.ndarray] = {}
    customer_cnt: dict[int, int] = {}
    event_chunks = []
    total_event_rows = 0
    emb_dim = int(cfg["pretrain"]["d_model"])

    with torch.no_grad():
        for batch in loader:
            for k in batch.keys():
                batch[k] = batch[k].to(device, non_blocking=True)
            out = model(
                event_token=batch["event_token_in"],
                channel_token=batch["channel_token_in"],
                hour_token=batch["hour_token"],
                delta_log=batch["delta_log"],
                attention_mask=batch["attention_mask"],
            )
            hidden = out["hidden"].detach().cpu().numpy().astype(np.float32, copy=False)
            event_ids = batch["event_id"].cpu().numpy()
            customer_ids = batch["customer_id"].cpu().numpy()
            valid_len = batch["valid_len"].cpu().numpy()
            eids, cids, eemb = _to_rows(event_ids, customer_ids, valid_len, hidden)
            if len(eids) == 0:
                continue

            if total_event_rows < args.max_event_rows:
                remain = max(0, args.max_event_rows - total_event_rows)
                take = min(remain, len(eids))
                if take > 0:
                    chunk = {
                        "event_id": eids[:take],
                        "customer_id": cids[:take],
                    }
                    for j in range(emb_dim):
                        chunk[f"emb_{j}"] = eemb[:take, j]
                    event_chunks.append(pl.DataFrame(chunk))
                    total_event_rows += take

            for i in range(len(eids)):
                cid = int(cids[i])
                if cid not in customer_sum:
                    customer_sum[cid] = np.zeros((emb_dim,), dtype=np.float64)
                    customer_cnt[cid] = 0
                customer_sum[cid] += eemb[i].astype(np.float64, copy=False)
                customer_cnt[cid] += 1

    if event_chunks:
        pl.concat(event_chunks, how="vertical").write_parquet(out_dir / "event_embeddings.parquet")

    rows = {"customer_id": [], **{f"cust_emb_{j}": [] for j in range(emb_dim)}}
    for cid in sorted(customer_sum.keys()):
        rows["customer_id"].append(cid)
        vec = customer_sum[cid] / max(customer_cnt[cid], 1)
        for j in range(emb_dim):
            rows[f"cust_emb_{j}"].append(float(vec[j]))
    pl.DataFrame(rows).write_parquet(out_dir / "customer_embeddings.parquet")

    with (out_dir / "export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "device": str(device),
                "event_rows_written": int(total_event_rows),
                "customers_written": int(len(rows["customer_id"])),
                "source_files": [str(f) for f in files],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[done] export_dir={out_dir}")


if __name__ == "__main__":
    main()

