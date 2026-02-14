from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd


def _as_str(s: pd.Series) -> np.ndarray:
    return s.astype("string").fillna("<NULL>").to_numpy()


def _bool_from_mixed(s: pd.Series) -> np.ndarray:
    t = s.astype("string").fillna("<NULL>").str.lower()
    return (~t.isin(["0", "false", "none", "null", "<null>", "<na>", "nan"])).to_numpy(
        dtype=np.int8
    )


def add_sequence_first_features(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df

    out = df.copy()
    out = out.sort_values(["customer_id", "event_ts", "event_id"], kind="mergesort")
    n = len(out)

    # Base device/session context (target-free)
    os_s = _as_str(out["operating_system_type"])
    ver_s = _as_str(out["device_system_version"])
    scr_s = _as_str(out["screen_size"])
    tz_s = _as_str(out["timezone"])
    a_lang = _as_str(out["accept_language"])
    b_lang = _as_str(out["browser_language"])
    mcc_s = _as_str(out["mcc_code"])
    ch_s = _as_str(out["channel_indicator_sub_type"])
    ev_s = _as_str(out["event_type_nm"])
    sess_s = _as_str(out["session_id"])

    out["device_fingerprint"] = (
        pd.Series(os_s)
        + "|"
        + pd.Series(ver_s)
        + "|"
        + pd.Series(scr_s)
        + "|"
        + pd.Series(tz_s)
        + "|"
        + pd.Series(a_lang)
        + "|"
        + pd.Series(b_lang)
    )
    fp_s = out["device_fingerprint"].to_numpy()

    out["lang_mismatch_flag"] = (
        (pd.Series(a_lang) != pd.Series(b_lang))
        & (pd.Series(a_lang) != "<NULL>")
        & (pd.Series(b_lang) != "<NULL>")
    ).astype(np.int8)

    dev_tools = _bool_from_mixed(out["developer_tools"])
    compromised = _bool_from_mixed(out["compromised"])
    rdp = _bool_from_mixed(out["web_rdp_connection"])
    out["rdp_or_root_flag"] = (dev_tools | compromised | rdp).astype(np.int8)

    battery_num = pd.to_numeric(out["battery"], errors="coerce")
    out["battery_num"] = battery_num.fillna(-1.0).astype(np.float32)
    out["battery_bucket"] = pd.cut(
        battery_num,
        bins=[-np.inf, 5, 20, 80, 100, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype("float").fillna(-1).astype(np.int8)
    out["battery_is_missing"] = battery_num.isna().astype(np.int8)

    # Arrays for prefix features
    cust = out["customer_id"].to_numpy()
    ts = out["event_ts"].to_numpy(dtype=np.int64)
    amt = out["operaton_amt"].fillna(0.0).to_numpy(dtype=np.float64)
    hour = out["event_hour"].to_numpy(dtype=np.int16)
    tz = tz_s
    lang_mismatch = out["lang_mismatch_flag"].to_numpy(dtype=np.int8)
    rdp_or_root = out["rdp_or_root_flag"].to_numpy(dtype=np.int8)
    battery = out["battery_num"].to_numpy(dtype=np.float32)

    seq_idx = np.zeros(n, dtype=np.int32)
    dt_prev = np.zeros(n, dtype=np.float32)
    dt_prev_same_channel = np.full(n, -1.0, dtype=np.float32)
    dt_prev_same_mcc = np.full(n, -1.0, dtype=np.float32)
    t_since_last_mcc = np.full(n, -1.0, dtype=np.float32)

    cnt_10m = np.zeros(n, dtype=np.int32)
    cnt_1h = np.zeros(n, dtype=np.int32)
    cnt_1d = np.zeros(n, dtype=np.int32)
    cnt_7d = np.zeros(n, dtype=np.int32)

    sum_amt_1d = np.zeros(n, dtype=np.float32)
    sum_amt_7d = np.zeros(n, dtype=np.float32)
    mean_amt_7d = np.zeros(n, dtype=np.float32)
    max_amt_7d = np.zeros(n, dtype=np.float32)
    burst_10m_over_1h = np.zeros(n, dtype=np.float32)
    burst_1h_over_1d = np.zeros(n, dtype=np.float32)

    is_first_seen_event_type = np.zeros(n, dtype=np.int8)
    is_first_seen_mcc = np.zeros(n, dtype=np.int8)
    is_first_seen_channel_subtype = np.zeros(n, dtype=np.int8)

    is_new_device = np.zeros(n, dtype=np.int8)
    device_change_count_7d = np.zeros(n, dtype=np.int32)
    timezone_change_flag = np.zeros(n, dtype=np.int8)
    battery_jump_flag = np.zeros(n, dtype=np.int8)

    session_event_index = np.zeros(n, dtype=np.int32)
    session_duration_so_far = np.zeros(n, dtype=np.float32)
    session_cnt_so_far = np.zeros(n, dtype=np.int32)
    session_amt_sum_so_far = np.zeros(n, dtype=np.float32)
    session_has_device_risk = np.zeros(n, dtype=np.int8)

    boundaries = np.r_[0, np.flatnonzero(cust[1:] != cust[:-1]) + 1, n]
    for bi in range(len(boundaries) - 1):
        s, e = int(boundaries[bi]), int(boundaries[bi + 1])
        t = ts[s:e]
        a = amt[s:e]
        idx = np.arange(e - s, dtype=np.int32)
        seq_idx[s:e] = idx

        if e - s > 1:
            dt_prev[s + 1 : e] = (t[1:] - t[:-1]).astype(np.float32)

        left_10m = np.searchsorted(t, t - 600, side="left")
        left_1h = np.searchsorted(t, t - 3600, side="left")
        left_1d = np.searchsorted(t, t - 86400, side="left")
        left_7d = np.searchsorted(t, t - 86400 * 7, side="left")
        cnt_10m[s:e] = idx - left_10m + 1
        cnt_1h[s:e] = idx - left_1h + 1
        cnt_1d[s:e] = idx - left_1d + 1
        cnt_7d[s:e] = idx - left_7d + 1

        csum = np.cumsum(a, dtype=np.float64)
        sum_amt_1d[s:e] = (csum - np.where(left_1d > 0, csum[left_1d - 1], 0)).astype(np.float32)
        sum_amt_7d[s:e] = (csum - np.where(left_7d > 0, csum[left_7d - 1], 0)).astype(np.float32)
        mean_amt_7d[s:e] = (sum_amt_7d[s:e] / np.maximum(cnt_7d[s:e], 1)).astype(np.float32)

        q: deque[tuple[int, float]] = deque()
        for j in range(e - s):
            cur_t = t[j]
            while q and q[0][0] < left_7d[j]:
                q.popleft()
            while q and q[-1][1] <= a[j]:
                q.pop()
            q.append((j, float(a[j])))
            max_amt_7d[s + j] = q[0][1]

        burst_10m_over_1h[s:e] = cnt_10m[s:e] / np.maximum(cnt_1h[s:e], 1)
        burst_1h_over_1d[s:e] = cnt_1h[s:e] / np.maximum(cnt_1d[s:e], 1)

        seen_ev: set[str] = set()
        seen_mcc: set[str] = set()
        seen_ch: set[str] = set()
        seen_fp: set[str] = set()
        last_ts_by_ch: dict[str, int] = {}
        last_ts_by_mcc: dict[str, int] = {}
        change_ts: deque[int] = deque()
        prev_fp = None
        prev_tz = None
        prev_battery = None

        sess_start: dict[str, int] = {}
        sess_cnt: dict[str, int] = {}
        sess_amt: dict[str, float] = {}
        sess_risk: dict[str, int] = {}

        for j in range(s, e):
            ch = ch_s[j]
            m = mcc_s[j]
            ev = ev_s[j]
            fp = fp_s[j]
            cur_ts = ts[j]
            cur_sess = sess_s[j]

            if ev not in seen_ev:
                is_first_seen_event_type[j] = 1
                seen_ev.add(ev)
            if m not in seen_mcc:
                is_first_seen_mcc[j] = 1
                seen_mcc.add(m)
            if ch not in seen_ch:
                is_first_seen_channel_subtype[j] = 1
                seen_ch.add(ch)

            if ch in last_ts_by_ch:
                dt_prev_same_channel[j] = float(cur_ts - last_ts_by_ch[ch])
            if m in last_ts_by_mcc:
                dt_prev_same_mcc[j] = float(cur_ts - last_ts_by_mcc[m])
                t_since_last_mcc[j] = dt_prev_same_mcc[j]
            last_ts_by_ch[ch] = cur_ts
            last_ts_by_mcc[m] = cur_ts

            if fp not in seen_fp:
                is_new_device[j] = 1
                seen_fp.add(fp)

            if prev_fp is not None and fp != prev_fp:
                change_ts.append(cur_ts)
            while change_ts and change_ts[0] < cur_ts - 86400 * 7:
                change_ts.popleft()
            device_change_count_7d[j] = len(change_ts)
            prev_fp = fp

            if prev_tz is not None and tz[j] != prev_tz:
                timezone_change_flag[j] = 1
            prev_tz = tz[j]

            if prev_battery is not None and battery[j] >= 0 and prev_battery >= 0:
                if abs(float(battery[j] - prev_battery)) >= 40:
                    battery_jump_flag[j] = 1
            prev_battery = battery[j]

            if cur_sess not in sess_start:
                sess_start[cur_sess] = cur_ts
                sess_cnt[cur_sess] = 0
                sess_amt[cur_sess] = 0.0
                sess_risk[cur_sess] = 0
            sess_cnt[cur_sess] += 1
            sess_amt[cur_sess] += float(amt[j])
            sess_risk[cur_sess] = int(sess_risk[cur_sess] or rdp_or_root[j] or lang_mismatch[j])

            session_event_index[j] = sess_cnt[cur_sess]
            session_cnt_so_far[j] = sess_cnt[cur_sess]
            session_duration_so_far[j] = float(cur_ts - sess_start[cur_sess])
            session_amt_sum_so_far[j] = float(sess_amt[cur_sess])
            session_has_device_risk[j] = sess_risk[cur_sess]

    out["seq_customer_idx"] = seq_idx
    out["dt_since_prev_sec"] = dt_prev
    out["dt_since_prev_same_channel_sec"] = dt_prev_same_channel
    out["dt_since_prev_same_mcc_sec"] = dt_prev_same_mcc
    out["time_since_last_seen_mcc_sec"] = t_since_last_mcc

    out["rolling_count_10m"] = cnt_10m
    out["rolling_count_1h"] = cnt_1h
    out["rolling_count_1d"] = cnt_1d
    out["rolling_count_7d"] = cnt_7d
    out["sum_amt_1d"] = sum_amt_1d
    out["sum_amt_7d"] = sum_amt_7d
    out["mean_amt_7d"] = mean_amt_7d
    out["max_amt_7d"] = max_amt_7d
    out["burst_10m_over_1h"] = burst_10m_over_1h
    out["burst_1h_over_1d"] = burst_1h_over_1d

    out["is_first_seen_event_type_for_customer"] = is_first_seen_event_type
    out["is_first_seen_mcc_for_customer"] = is_first_seen_mcc
    out["is_first_seen_channel_subtype_for_customer"] = is_first_seen_channel_subtype

    out["is_new_device_for_customer"] = is_new_device
    out["device_change_count_7d"] = device_change_count_7d
    out["timezone_change_flag"] = timezone_change_flag
    out["battery_jump_flag"] = battery_jump_flag

    out["session_event_index"] = session_event_index
    out["session_duration_so_far"] = session_duration_so_far
    out["session_cnt_so_far"] = session_cnt_so_far
    out["session_amt_sum_so_far"] = session_amt_sum_so_far
    out["session_has_device_risk_flag"] = session_has_device_risk

    out["is_night"] = ((hour <= 5) | (hour >= 23)).astype(np.int8)
    out["amount_bucket"] = pd.cut(
        out["log_operaton_amt"].astype(float),
        bins=[-np.inf, 2, 6, 10, 14, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype("float").fillna(-1).astype(np.int8)

    return out.sort_index()

