from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

TARGET = "fraud_bool"
EPS = 1e-6


@dataclass
class FeatureStats:
    q_curr_addr_short: float
    q_income_low: float
    q_income_high: float
    q_init_hi: float
    q_req6_hi: float
    q_req24_hi: float
    q_req4w_hi: float
    month_rate_mean: Dict[str, Dict[float, float]]
    month_rate_std: Dict[str, Dict[float, float]]
    age_dob_median: Dict[float, float]


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def fit_stats(train_df: pd.DataFrame) -> FeatureStats:
    t = train_df.copy()

    curr_addr = _safe_num(t["curr_addr_months"]).replace(-1, np.nan)
    yearly_income = _safe_num(t["yearly_income"])
    init_amt = _safe_num(t["init_transfer_amt"])

    req6 = _safe_num(t["req_rate_6h"])
    req24 = _safe_num(t["req_rate_24h"])
    req4w = _safe_num(t["req_rate_4w"])

    q_curr_addr_short = float(curr_addr.quantile(0.25))
    q_income_low = float(yearly_income.quantile(0.10))
    q_income_high = float(yearly_income.quantile(0.90))
    q_init_hi = float(init_amt.quantile(0.995))
    q_req6_hi = float(req6.quantile(0.99))
    q_req24_hi = float(req24.quantile(0.99))
    q_req4w_hi = float(req4w.quantile(0.99))

    month = _safe_num(t["month_idx"]).fillna(-1)
    month_rate_mean: Dict[str, Dict[float, float]] = {}
    month_rate_std: Dict[str, Dict[float, float]] = {}

    for col in ["req_rate_6h", "req_rate_24h", "req_rate_4w", "zip_req_count_4w"]:
        x = _safe_num(t[col])
        g = pd.DataFrame({"month": month, "x": x}).groupby("month")["x"]
        month_rate_mean[col] = g.mean().to_dict()
        std_map = g.std().fillna(0.0)
        std_map = std_map.where(std_map > 0, 1.0)
        month_rate_std[col] = std_map.to_dict()

    age = _safe_num(t["age_bucket"])
    dob = _safe_num(t["dob_email_count_4w"])
    age_dob_median = pd.DataFrame({"age": age, "dob": dob}).groupby("age")["dob"].median().to_dict()

    return FeatureStats(
        q_curr_addr_short=q_curr_addr_short,
        q_income_low=q_income_low,
        q_income_high=q_income_high,
        q_init_hi=q_init_hi,
        q_req6_hi=q_req6_hi,
        q_req24_hi=q_req24_hi,
        q_req4w_hi=q_req4w_hi,
        month_rate_mean=month_rate_mean,
        month_rate_std=month_rate_std,
        age_dob_median=age_dob_median,
    )


def _z_by_month(x: pd.Series, month: pd.Series, mean_map: Dict[float, float], std_map: Dict[float, float]) -> pd.Series:
    mu = month.map(mean_map).astype(float)
    sd = month.map(std_map).astype(float)
    sd = sd.fillna(1.0).replace(0.0, 1.0)
    return (x - mu) / (sd + EPS)


def add_eda_corr_features(df: pd.DataFrame, stats: FeatureStats) -> pd.DataFrame:
    out = df.copy()

    age = _safe_num(out["age_bucket"])
    month = _safe_num(out["month_idx"]).fillna(-1)

    curr_addr_raw = _safe_num(out["curr_addr_months"])
    prev_addr_raw = _safe_num(out["prev_addr_months"])
    curr_addr = curr_addr_raw.replace(-1, np.nan)
    prev_addr = prev_addr_raw.replace(-1, np.nan)

    req6 = _safe_num(out["req_rate_6h"])
    req24 = _safe_num(out["req_rate_24h"])
    req4w = _safe_num(out["req_rate_4w"])
    zip4w = _safe_num(out["zip_req_count_4w"])

    credit = _safe_num(out["credit_risk_score"])
    limit_ = _safe_num(out["req_credit_limit"])
    dob = _safe_num(out["dob_email_count_4w"])
    income = _safe_num(out["yearly_income"])
    init_amt = _safe_num(out["init_transfer_amt"])

    home_valid = _safe_num(out["is_home_phone_valid"]).fillna(0)
    mobile_valid = _safe_num(out["is_mobile_valid"]).fillna(0)
    free_email = _safe_num(out["is_free_email"]).fillna(0)
    foreign_req = _safe_num(out["is_foreign_req"]).fillna(0)

    payment = out["payment_type"].astype(str)
    housing = out["housing_status"].astype(str)
    employment = out["employment_status"].astype(str)

    # Month-driven request behavior interactions.
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    for c in ["req_rate_6h", "req_rate_24h", "req_rate_4w", "zip_req_count_4w"]:
        x = _safe_num(out[c])
        out[f"{c}_month_z"] = _z_by_month(
            x,
            month,
            stats.month_rate_mean[c],
            stats.month_rate_std[c],
        )

    out["req6_to_24_ratio"] = req6 / (req24 + EPS)
    out["req24_to_4w_ratio"] = req24 / (req4w + EPS)
    out["req_rate_6h_hi"] = (req6 > stats.q_req6_hi).astype(np.int8)
    out["req_rate_24h_hi"] = (req24 > stats.q_req24_hi).astype(np.int8)
    out["req_rate_4w_hi"] = (req4w > stats.q_req4w_hi).astype(np.int8)

    # Credit and limit interaction.
    out["credit_limit_ratio"] = credit / (limit_ + EPS)
    out["credit_limit_gap"] = limit_ - credit
    out["credit_x_limit"] = credit * np.log1p(np.clip(limit_, a_min=0, a_max=None))
    out["high_credit_high_limit"] = ((credit >= 200) & (limit_ >= 1500)).astype(np.int8)

    # Address duration and contact consistency.
    out["curr_addr_missing"] = curr_addr_raw.eq(-1).astype(np.int8)
    out["prev_addr_missing"] = prev_addr_raw.eq(-1).astype(np.int8)
    out["curr_addr_short"] = (curr_addr.fillna(stats.q_curr_addr_short) <= stats.q_curr_addr_short).astype(np.int8)
    out["addr_gap_curr_minus_prev"] = curr_addr.fillna(0) - prev_addr.fillna(0)
    out["addr_total_months"] = curr_addr.fillna(0) + prev_addr.fillna(0)

    out["phone_both_invalid"] = ((home_valid == 0) & (mobile_valid == 0)).astype(np.int8)
    out["phone_valid_mismatch"] = (home_valid != mobile_valid).astype(np.int8)
    out["contact_risk_score"] = (1 - home_valid) + (1 - mobile_valid) + free_email + foreign_req

    # Age-linked features from stack and cross tables.
    out["age_ge60"] = (age >= 60).astype(np.int8)
    out["age_ge70"] = (age >= 70).astype(np.int8)
    out["age_x_payment"] = age.astype("Int64").astype(str) + "_" + payment
    out["age_x_housing"] = age.astype("Int64").astype(str) + "_" + housing
    out["age_x_employment"] = age.astype("Int64").astype(str) + "_" + employment

    out["age_ge60_and_payment_ac"] = ((age >= 60) & payment.eq("AC")).astype(np.int8)
    out["age_ge60_and_housing_ba"] = ((age >= 60) & housing.eq("BA")).astype(np.int8)

    # dob_email_count_4w x age_bucket strong inverse relationship.
    age_median_dob = age.map(stats.age_dob_median)
    out["dob_age_resid"] = dob - age_median_dob
    out["dob_over_age"] = dob / (age + 1.0)
    out["dob_low_old_age"] = ((dob <= 4) & (age >= 60)).astype(np.int8)

    # Distribution-tail flags from EDA guidance.
    out["name_email_sim_low"] = (_safe_num(out["name_email_sim"]) <= 0.30).astype(np.int8)
    out["credit_score_negative"] = (credit < 0).astype(np.int8)
    out["init_transfer_hi"] = (init_amt > stats.q_init_hi).astype(np.int8)
    out["income_low_decile"] = (income <= stats.q_income_low).astype(np.int8)
    out["income_high_decile"] = (income >= stats.q_income_high).astype(np.int8)

    return out


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    stats = fit_stats(train_df)
    train_out = add_eda_corr_features(train_df, stats)
    test_out = add_eda_corr_features(test_df, stats)

    original_cols = set(train_df.columns)
    new_cols = [c for c in train_out.columns if c not in original_cols]
    return train_out, test_out, new_cols


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate EDA-corr-stack engineered features")
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--out-train", default="train_corrstack_features.csv")
    ap.add_argument("--out-test", default="test_corrstack_features.csv")
    ap.add_argument("--out-feature-list", default="corrstack_feature_list.txt")
    args = ap.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    train_out, test_out, new_cols = build_features(train_df, test_df)

    train_out.to_csv(args.out_train, index=False)
    test_out.to_csv(args.out_test, index=False)

    with open(args.out_feature_list, "w", encoding="utf-8") as f:
        for c in new_cols:
            f.write(c + "\n")

    print(f"[OK] new_features={len(new_cols)}")
    print(f"[OK] train_out={args.out_train}")
    print(f"[OK] test_out={args.out_test}")
    print(f"[OK] feature_list={args.out_feature_list}")


if __name__ == "__main__":
    main()
