from __future__ import annotations

import polars as pl


FEATURE_COLUMNS = [
    "customer_id",
    "event_id",
    "event_dttm",
    "event_type_nm",
    "event_desc",
    "channel_indicator_type",
    "channel_indicator_sub_type",
    "operaton_amt",
    "currency_iso_cd",
    "mcc_code",
    "pos_cd",
    "accept_language",
    "browser_language",
    "timezone",
    "session_id",
    "operating_system_type",
    "battery",
    "device_system_version",
    "screen_size",
    "developer_tools",
    "phone_voip_call_state",
    "web_rdp_connection",
    "compromised",
]


def with_time_and_amount_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    dt = pl.col("event_dttm").str.strptime(
        pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False
    )
    return lf.with_columns(
        dt.alias("event_dt"),
        dt.dt.truncate("1w").alias("week_start"),
    ).with_columns(
        pl.col("event_dt").dt.hour().alias("event_hour"),
        pl.col("event_dt").dt.weekday().alias("event_weekday"),
        pl.col("event_dt").dt.day().alias("event_day"),
        pl.col("event_dt").dt.week().alias("event_week"),
        pl.col("event_dt").dt.month().alias("event_month"),
        pl.when(pl.col("event_dt").dt.weekday().is_in([5, 6]))
        .then(1)
        .otherwise(0)
        .alias("is_weekend"),
        pl.col("event_dt").dt.epoch("s").alias("event_ts"),
        pl.col("operaton_amt").fill_null(0.0).alias("operaton_amt"),
        pl.col("operaton_amt").fill_null(0.0).log1p().alias("log_operaton_amt"),
        pl.when(pl.col("operaton_amt").fill_null(0.0) == 0.0)
        .then(1)
        .otherwise(0)
        .alias("is_zero_amt"),
    )

