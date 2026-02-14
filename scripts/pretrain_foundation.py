#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from competition.foundation_data import (
    FoundationCollator,
    TransactionSequenceDataset,
    discover_parquet_files,
    estimate_vocab_sizes,
)
from competition.foundation_model import TemporalFoundationEncoder
from competition.mlflow_tracking import MlflowTracker
from competition.pipeline_config import load_config


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _pick_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _loss_terms(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    w_next_event: float,
    w_next_channel: float,
    w_mlm_event: float,
    w_mlm_channel: float,
    w_next_delta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_next_event = F.cross_entropy(
        out["next_event_logits"].reshape(-1, out["next_event_logits"].shape[-1]),
        batch["next_event_target"].reshape(-1),
        ignore_index=-100,
    )
    loss_next_channel = F.cross_entropy(
        out["next_channel_logits"].reshape(-1, out["next_channel_logits"].shape[-1]),
        batch["next_channel_target"].reshape(-1),
        ignore_index=-100,
    )
    loss_mlm_event = F.cross_entropy(
        out["next_event_logits"].reshape(-1, out["next_event_logits"].shape[-1]),
        batch["mlm_event_target"].reshape(-1),
        ignore_index=-100,
    )
    loss_mlm_channel = F.cross_entropy(
        out["next_channel_logits"].reshape(-1, out["next_channel_logits"].shape[-1]),
        batch["mlm_channel_target"].reshape(-1),
        ignore_index=-100,
    )
    delta_mask = batch["next_delta_target"] >= 0
    if bool(delta_mask.any()):
        loss_delta = F.mse_loss(
            out["next_delta_pred"][delta_mask],
            batch["next_delta_target"][delta_mask],
        )
    else:
        loss_delta = out["next_delta_pred"].new_tensor(0.0)

    total = (
        w_next_event * loss_next_event
        + w_next_channel * loss_next_channel
        + w_mlm_event * loss_mlm_event
        + w_mlm_channel * loss_mlm_channel
        + w_next_delta * loss_delta
    )
    return total, {
        "loss_next_event": float(loss_next_event.item()),
        "loss_next_channel": float(loss_next_channel.item()),
        "loss_mlm_event": float(loss_mlm_event.item()),
        "loss_mlm_channel": float(loss_mlm_channel.item()),
        "loss_next_delta": float(loss_delta.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain transaction foundation encoder.")
    parser.add_argument("--config", default="conf/foundation_pretrain.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None, help="auto|cpu|cuda|cuda:0|cuda:1")
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--mlflow-uri", default=None, type=str)
    parser.add_argument("--mlflow-experiment", default="competition-foundation", type=str)
    parser.add_argument("--mlflow-run-name", default=None, type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = args.run_name or _run_id()
    out_dir = Path(cfg["paths"]["artifacts_dir"]) / "foundation" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = MlflowTracker(
        enabled=not args.disable_mlflow,
        tracking_uri=args.mlflow_uri,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name or run_name,
    )

    pre_cfg = cfg["pretrain"]
    data_dir = Path(cfg["paths"]["data_dir"])
    patterns = list(pre_cfg["source_globs"])
    files = discover_parquet_files(data_dir, patterns)
    if not files:
        raise RuntimeError(f"No files found for patterns={patterns} in {data_dir}")

    event_vocab_size, channel_vocab_size = estimate_vocab_sizes(files)
    seq_len = int(pre_cfg["seq_len"])
    min_len = int(pre_cfg["min_len"])
    stride = int(pre_cfg["stride"])
    dataset = TransactionSequenceDataset(
        parquet_files=files,
        seq_len=seq_len,
        min_len=min_len,
        stride=stride,
        shuffle_files=bool(pre_cfg.get("shuffle_files", True)),
        seed=int(pre_cfg.get("seed", 42)),
    )
    collator = FoundationCollator(
        seq_len=seq_len,
        event_vocab_size=event_vocab_size,
        channel_vocab_size=channel_vocab_size,
        mask_prob=float(pre_cfg.get("mask_prob", 0.15)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(pre_cfg["batch_size"]),
        num_workers=int(pre_cfg.get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collator,
    )

    requested_device = args.device or str(pre_cfg.get("device", "auto"))
    device = _pick_device(requested_device)
    use_bf16 = bool(pre_cfg.get("bf16", True)) and device.type == "cuda"
    model = TemporalFoundationEncoder(
        event_vocab_size=event_vocab_size,
        channel_vocab_size=channel_vocab_size,
        hour_vocab_size=25,
        d_model=int(pre_cfg["d_model"]),
        n_heads=int(pre_cfg["n_heads"]),
        n_layers=int(pre_cfg["n_layers"]),
        ff_mult=int(pre_cfg.get("ff_mult", 4)),
        dropout=float(pre_cfg.get("dropout", 0.1)),
        max_seq_len=seq_len,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(pre_cfg["lr"]),
        weight_decay=float(pre_cfg.get("weight_decay", 0.01)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    grad_accum = int(pre_cfg.get("grad_accum_steps", 1))
    max_steps = int(pre_cfg["max_steps"])
    log_every = int(pre_cfg.get("log_every", 20))
    save_every = int(pre_cfg.get("save_every", 500))
    clip_grad = float(pre_cfg.get("clip_grad_norm", 1.0))

    w_next_event = float(pre_cfg.get("w_next_event", 1.0))
    w_next_channel = float(pre_cfg.get("w_next_channel", 1.0))
    w_mlm_event = float(pre_cfg.get("w_mlm_event", 1.0))
    w_mlm_channel = float(pre_cfg.get("w_mlm_channel", 1.0))
    w_next_delta = float(pre_cfg.get("w_next_delta", 0.2))

    step = 0
    accum = 0
    running_total = 0.0
    running_n = 0
    model.train()
    print(
        f"[setup] device={device} bf16={use_bf16} files={len(files)} "
        f"seq_len={seq_len} batch={pre_cfg['batch_size']} max_steps={max_steps}"
    )
    tracker.log_params(
        {
            "run_name": run_name,
            "config_path": args.config,
            "device": str(device),
            "bf16": use_bf16,
            "seq_len": seq_len,
            "batch_size": int(pre_cfg["batch_size"]),
            "grad_accum_steps": grad_accum,
            "max_steps": max_steps,
            "d_model": int(pre_cfg["d_model"]),
            "n_heads": int(pre_cfg["n_heads"]),
            "n_layers": int(pre_cfg["n_layers"]),
            "lr": float(pre_cfg["lr"]),
            "source_files_count": len(files),
        }
    )
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            for k in batch.keys():
                batch[k] = batch[k].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                out = model(
                    event_token=batch["event_token_in"],
                    channel_token=batch["channel_token_in"],
                    hour_token=batch["hour_token"],
                    delta_log=batch["delta_log"],
                    attention_mask=batch["attention_mask"],
                )
                loss, terms = _loss_terms(
                    out=out,
                    batch=batch,
                    w_next_event=w_next_event,
                    w_next_channel=w_next_channel,
                    w_mlm_event=w_mlm_event,
                    w_mlm_channel=w_mlm_channel,
                    w_next_delta=w_next_delta,
                )
                loss = loss / grad_accum

            loss.backward()
            accum += 1
            if accum >= grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                step += 1
                running_total += float(loss.item()) * grad_accum
                running_n += 1
                if step % log_every == 0:
                    avg = running_total / max(running_n, 1)
                    print(
                        f"step={step}/{max_steps} "
                        f"loss={avg:.5f} "
                        f"next_e={terms['loss_next_event']:.4f} "
                        f"next_c={terms['loss_next_channel']:.4f} "
                        f"mlm_e={terms['loss_mlm_event']:.4f} "
                        f"mlm_c={terms['loss_mlm_channel']:.4f} "
                        f"next_dt={terms['loss_next_delta']:.4f}"
                    )
                    tracker.log_metrics(
                        {
                            "train_loss": float(avg),
                            "loss_next_event": float(terms["loss_next_event"]),
                            "loss_next_channel": float(terms["loss_next_channel"]),
                            "loss_mlm_event": float(terms["loss_mlm_event"]),
                            "loss_mlm_channel": float(terms["loss_mlm_channel"]),
                            "loss_next_delta": float(terms["loss_next_delta"]),
                        },
                        step=step,
                    )
                    running_total = 0.0
                    running_n = 0
                if step % save_every == 0:
                    ckpt = {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "event_vocab_size": event_vocab_size,
                        "channel_vocab_size": channel_vocab_size,
                        "seq_len": seq_len,
                        "config_path": args.config,
                        "run_name": run_name,
                    }
                    torch.save(ckpt, out_dir / f"checkpoint_step_{step}.pt")
                    tracker.log_artifact(out_dir / f"checkpoint_step_{step}.pt")

    final_path = out_dir / "checkpoint_final.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "event_vocab_size": event_vocab_size,
            "channel_vocab_size": channel_vocab_size,
            "seq_len": seq_len,
            "config_path": args.config,
            "run_name": run_name,
        },
        final_path,
    )
    meta = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_final": str(final_path),
        "event_vocab_size": event_vocab_size,
        "channel_vocab_size": channel_vocab_size,
        "device": str(device),
        "bf16": use_bf16,
        "max_steps": max_steps,
        "source_files": [str(f) for f in files],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    tracker.log_metrics({"final_step": float(step)})
    tracker.log_artifact(final_path)
    tracker.log_artifact(out_dir / "summary.json")
    tracker.end(status="FINISHED")
    print(f"[done] run_dir={out_dir}")


if __name__ == "__main__":
    main()
