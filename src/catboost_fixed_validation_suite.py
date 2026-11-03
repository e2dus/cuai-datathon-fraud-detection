import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import (
    GroupKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

warnings.filterwarnings("ignore")


SEED = 42
N_SPLITS = 5

# User-provided fixed CatBoost params (no Optuna tuning).
BEST_CAT_PARAMS = {
    "iterations": 2369,
    "learning_rate": 0.05156911709595948,
    "depth": 3,
    "l2_leaf_reg": 18.56686944170984,
    "random_strength": 3.125526917665803,
    "bagging_temperature": 2.1467430226178355,
    "border_count": 97,
    "bootstrap_type": "Bayesian",
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "random_seed": SEED,
    "auto_class_weights": "Balanced",
    "early_stopping_rounds": 80,
    "verbose": False,
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def merge_rare_categories(train_s: pd.Series, test_s: pd.Series, min_freq: float = 0.005):
    freq = train_s.value_counts(normalize=True)
    rare = set(freq[freq < min_freq].index)
    tr = train_s.where(~train_s.isin(rare), "Other")
    te = test_s.where(~test_s.isin(rare), "Other")
    return tr, te


def clip_log_transform(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    col: str,
    q_low=0.0,
    q_high=0.995,
):
    lo = float(train_df[col].quantile(q_low))
    hi = float(train_df[col].quantile(q_high))
    train_df[col] = np.log1p(train_df[col].clip(lo, hi))
    test_df[col] = np.log1p(test_df[col].clip(lo, hi))


def preprocess_guided(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train = train_df.copy()
    test = test_df.copy()

    drop_base = ["id", "device_fraud_history"]
    for c in drop_base:
        if c in train.columns:
            train = train.drop(columns=c)
        if c in test.columns:
            test = test.drop(columns=c)

    mi_top_bin = [
        "device_email_cnt_8w",
        "is_session_persistent",
        "is_mobile_valid",
        "is_free_email",
        "is_home_phone_valid",
    ]
    for c in mi_top_bin:
        if c in train.columns:
            train[c] = pd.to_numeric(train[c], errors="coerce")
            test[c] = pd.to_numeric(test[c], errors="coerce")

    for c in ["prev_addr_months", "bank_months_count"]:
        if c in train.columns:
            med = train.loc[train[c] != -1, c].median()
            train[f"is_missing_{c}"] = (train[c] == -1).astype(np.int8)
            test[f"is_missing_{c}"] = (test[c] == -1).astype(np.int8)
            train[c] = train[c].where(train[c] != -1, med)
            test[c] = test[c].where(test[c] != -1, med)

    if "device_email_cnt_8w" in train.columns:
        tr_dev = pd.to_numeric(train["device_email_cnt_8w"], errors="coerce")
        te_dev = pd.to_numeric(test["device_email_cnt_8w"], errors="coerce")
        train["flag_device_email_neq1"] = (tr_dev != 1).astype(np.int8)
        test["flag_device_email_neq1"] = (te_dev != 1).astype(np.int8)
        train["flag_device_email_missing"] = (tr_dev.isna() | (tr_dev == -1)).astype(np.int8)
        test["flag_device_email_missing"] = (te_dev.isna() | (te_dev == -1)).astype(np.int8)
        med_dev = tr_dev.replace(-1, np.nan).median()
        train["device_email_cnt_8w"] = tr_dev.replace(-1, np.nan).fillna(med_dev)
        test["device_email_cnt_8w"] = te_dev.replace(-1, np.nan).fillna(med_dev)

    if "name_email_sim" in train.columns:
        train["flag_name_email_low"] = (train["name_email_sim"] < 0.3).astype(np.int8)
        test["flag_name_email_low"] = (test["name_email_sim"] < 0.3).astype(np.int8)
        train["name_email_sim"] = train["name_email_sim"].clip(0, 1)
        test["name_email_sim"] = test["name_email_sim"].clip(0, 1)

    if "credit_risk_score" in train.columns:
        train["credit_risk_score"] = pd.to_numeric(train["credit_risk_score"], errors="coerce")
        test["credit_risk_score"] = pd.to_numeric(test["credit_risk_score"], errors="coerce")
        cr_med = train["credit_risk_score"].replace(-1, np.nan).median()
        train["credit_risk_score"] = train["credit_risk_score"].replace(-1, np.nan).fillna(cr_med)
        test["credit_risk_score"] = test["credit_risk_score"].replace(-1, np.nan).fillna(cr_med)

    if "req_credit_limit" in train.columns:
        train["req_credit_limit"] = pd.to_numeric(train["req_credit_limit"], errors="coerce")
        test["req_credit_limit"] = pd.to_numeric(test["req_credit_limit"], errors="coerce")
        rcl_med = train["req_credit_limit"].median()
        train["req_credit_limit"] = train["req_credit_limit"].fillna(rcl_med)
        test["req_credit_limit"] = test["req_credit_limit"].fillna(rcl_med)

    if "days_since_req" in train.columns:
        train["flag_days_since_req_lt7"] = (train["days_since_req"] < 7).astype(np.int8)
        test["flag_days_since_req_lt7"] = (test["days_since_req"] < 7).astype(np.int8)

    if "init_transfer_amt" in train.columns:
        threshold = float(train.loc[train["init_transfer_amt"] >= 0, "init_transfer_amt"].quantile(0.95))
        train["flag_init_transfer_high"] = (train["init_transfer_amt"] >= threshold).astype(np.int8)
        test["flag_init_transfer_high"] = (test["init_transfer_amt"] >= threshold).astype(np.int8)
        train["init_transfer_amt"] = train["init_transfer_amt"].clip(lower=0)
        test["init_transfer_amt"] = test["init_transfer_amt"].clip(lower=0)

    if "yearly_income" in train.columns:
        y_lo = float(train["yearly_income"].quantile(0.1))
        y_hi = float(train["yearly_income"].quantile(0.9))
        train["flag_income_low"] = (train["yearly_income"] <= y_lo).astype(np.int8)
        train["flag_income_high"] = (train["yearly_income"] >= y_hi).astype(np.int8)
        test["flag_income_low"] = (test["yearly_income"] <= y_lo).astype(np.int8)
        test["flag_income_high"] = (test["yearly_income"] >= y_hi).astype(np.int8)

    mapping_flags = [
        ("housing_status", "BA", "flag_housing_BA"),
        ("device_os", "windows", "flag_os_windows"),
        ("employment_status", "CC", "flag_employment_CC"),
        ("payment_type", "AC", "flag_payment_AC"),
        ("application_source", "TELEAPP", "flag_source_TELEAPP"),
    ]
    for col, val, out_col in mapping_flags:
        if col in train.columns:
            train[out_col] = (train[col].astype(str) == val).astype(np.int8)
            test[out_col] = (test[col].astype(str) == val).astype(np.int8)

    if "month_idx" in train.columns:
        nm = 8.0
        m_tr = pd.to_numeric(train["month_idx"], errors="coerce").fillna(0.0)
        m_te = pd.to_numeric(test["month_idx"], errors="coerce").fillna(0.0)
        train["month_sin"] = np.sin(2 * np.pi * m_tr / nm).astype(np.float32)
        train["month_cos"] = np.cos(2 * np.pi * m_tr / nm).astype(np.float32)
        test["month_sin"] = np.sin(2 * np.pi * m_te / nm).astype(np.float32)
        test["month_cos"] = np.cos(2 * np.pi * m_te / nm).astype(np.float32)
        train["month_idx_cat"] = train["month_idx"].astype(str)
        test["month_idx_cat"] = test["month_idx"].astype(str)
        train = train.drop(columns=["month_idx"])
        test = test.drop(columns=["month_idx"])

    req_cols = ["req_rate_6h", "req_rate_24h", "req_rate_4w"]
    if all(c in train.columns for c in req_cols):
        tr6_raw = train["req_rate_6h"].clip(lower=0)
        tr24_raw = train["req_rate_24h"].clip(lower=0)
        tr4w_raw = train["req_rate_4w"].clip(lower=0)
        te6_raw = test["req_rate_6h"].clip(lower=0)
        te24_raw = test["req_rate_24h"].clip(lower=0)
        te4w_raw = test["req_rate_4w"].clip(lower=0)

        train["feat_req24_over_4w"] = tr24_raw / (tr4w_raw + 1e-6)
        test["feat_req24_over_4w"] = te24_raw / (te4w_raw + 1e-6)
        train["feat_req6_over_24"] = tr6_raw / (tr24_raw + 1e-6)
        test["feat_req6_over_24"] = te6_raw / (te24_raw + 1e-6)
        train["feat_req6_over_4w"] = tr6_raw / (tr4w_raw + 1e-6)
        test["feat_req6_over_4w"] = te6_raw / (te4w_raw + 1e-6)

        q4 = np.unique(tr4w_raw.quantile([0.0, 1 / 3, 2 / 3, 1.0]).values)
        if len(q4) >= 4:
            train["req_rate_4w_bin3"] = pd.cut(tr4w_raw, bins=q4, include_lowest=True, labels=False)
            test["req_rate_4w_bin3"] = pd.cut(te4w_raw, bins=q4, include_lowest=True, labels=False)
        else:
            train["req_rate_4w_bin3"] = pd.qcut(tr4w_raw.rank(method="first"), 3, labels=False)
            q = np.quantile(tr4w_raw, [0, 1 / 3, 2 / 3, 1])
            test["req_rate_4w_bin3"] = pd.cut(te4w_raw, bins=q, include_lowest=True, labels=False)

        train["req_rate_4w_bin3"] = train["req_rate_4w_bin3"].fillna(0).astype(np.int8)
        test["req_rate_4w_bin3"] = test["req_rate_4w_bin3"].fillna(0).astype(np.int8)

        med24 = float(tr24_raw.median())
        train["req_rate_24h_bin2"] = (tr24_raw >= med24).astype(np.int8)
        test["req_rate_24h_bin2"] = (te24_raw >= med24).astype(np.int8)

    if "age_bucket" in train.columns:
        age_tr = train["age_bucket"].astype(str)
        age_te = test["age_bucket"].astype(str)
        rel_cols = [
            "credit_risk_score",
            "req_credit_limit",
            "curr_addr_months",
            "name_email_sim",
            "session_length_min",
        ]
        for c in rel_cols:
            if c in train.columns:
                tr_num = pd.to_numeric(train[c], errors="coerce")
                te_num = pd.to_numeric(test[c], errors="coerce")

                grp_mean = tr_num.groupby(age_tr).mean()
                grp_std = tr_num.groupby(age_tr).std().replace(0, np.nan)
                global_mean = float(tr_num.mean())
                std_val = float(tr_num.std())
                global_std = std_val if std_val > 1e-6 else 1.0

                tr_mu = age_tr.map(grp_mean).fillna(global_mean)
                te_mu = age_te.map(grp_mean).fillna(global_mean)
                tr_sd = age_tr.map(grp_std).fillna(global_std)
                te_sd = age_te.map(grp_std).fillna(global_std)

                train[f"age_adj_{c}_diff"] = (tr_num - tr_mu).astype(np.float32)
                test[f"age_adj_{c}_diff"] = (te_num - te_mu).astype(np.float32)
                train[f"age_adj_{c}_ratio"] = (tr_num / (tr_mu + 1e-6)).astype(np.float32)
                test[f"age_adj_{c}_ratio"] = (te_num / (te_mu + 1e-6)).astype(np.float32)
                train[f"age_adj_{c}_z"] = ((tr_num - tr_mu) / (tr_sd + 1e-6)).astype(np.float32)
                test[f"age_adj_{c}_z"] = ((te_num - te_mu) / (te_sd + 1e-6)).astype(np.float32)
                train[f"age_adj_{c}_absz_gt2"] = (np.abs(train[f"age_adj_{c}_z"]) > 2.0).astype(np.int8)
                test[f"age_adj_{c}_absz_gt2"] = (np.abs(test[f"age_adj_{c}_z"]) > 2.0).astype(np.int8)

    skew_cols = [
        "curr_addr_months",
        "session_length_min",
        "req_credit_limit",
        "branch_req_count_8w",
        "days_since_req",
        "dob_email_count_4w",
        "init_transfer_amt",
        "zip_req_count_4w",
        "req_rate_6h",
        "req_rate_24h",
        "req_rate_4w",
    ]
    for c in skew_cols:
        if c in train.columns:
            clip_log_transform(train, test, c, q_low=0.0, q_high=0.995)

    for c in ["req_rate_4w", "req_rate_24h"]:
        if c in train.columns and f"{c}_bin" not in train.columns:
            edges = np.unique(train[c].quantile([0.0, 0.33, 0.66, 1.0]).values)
            if len(edges) >= 3:
                train[f"{c}_bin"] = pd.cut(train[c], bins=edges, include_lowest=True, labels=False).astype("float")
                test[f"{c}_bin"] = pd.cut(test[c], bins=edges, include_lowest=True, labels=False).astype("float")
                train[f"{c}_bin"] = train[f"{c}_bin"].fillna(0).astype(np.int8)
                test[f"{c}_bin"] = test[f"{c}_bin"].fillna(0).astype(np.int8)

    if "req_credit_limit" in train.columns and "credit_risk_score" in train.columns:
        train["feat_limit_over_credit"] = train["req_credit_limit"] / (train["credit_risk_score"] + 1e-3)
        test["feat_limit_over_credit"] = test["req_credit_limit"] / (test["credit_risk_score"] + 1e-3)

    if "credit_risk_score" in train.columns and "age_bucket" in train.columns:
        age_tr_num = pd.to_numeric(train["age_bucket"], errors="coerce").fillna(0)
        age_te_num = pd.to_numeric(test["age_bucket"], errors="coerce").fillna(0)
        train["feat_credit_x_age"] = train["credit_risk_score"] * age_tr_num
        test["feat_credit_x_age"] = test["credit_risk_score"] * age_te_num

    if "curr_addr_months" in train.columns and "name_email_sim" in train.columns:
        train["feat_curr_x_name_sim"] = train["curr_addr_months"] * train["name_email_sim"]
        test["feat_curr_x_name_sim"] = test["curr_addr_months"] * test["name_email_sim"]

    if "name_email_sim" in train.columns and "is_mobile_valid" in train.columns:
        train["feat_name_x_mobile_valid"] = train["name_email_sim"] * train["is_mobile_valid"]
        test["feat_name_x_mobile_valid"] = test["name_email_sim"] * test["is_mobile_valid"]

    if "credit_risk_score" in train.columns and "is_session_persistent" in train.columns:
        train["feat_credit_x_session_persistent"] = train["credit_risk_score"] * train["is_session_persistent"]
        test["feat_credit_x_session_persistent"] = test["credit_risk_score"] * test["is_session_persistent"]

    for c in ["housing_status", "device_os", "employment_status", "payment_type", "application_source"]:
        if c in train.columns:
            train[c], test[c] = merge_rare_categories(train[c].astype(str), test[c].astype(str), min_freq=0.005)

    target_col = "fraud_bool"
    cat_cols = [
        c
        for c in train.columns
        if c != target_col
        and (
            pd.api.types.is_object_dtype(train[c])
            or pd.api.types.is_string_dtype(train[c])
            or pd.api.types.is_categorical_dtype(train[c])
        )
    ]

    if cat_cols:
        train = pd.get_dummies(train, columns=cat_cols, dummy_na=False)
        test = pd.get_dummies(test, columns=cat_cols, dummy_na=False)

    residual_cat_cols = [
        c
        for c in train.columns
        if c != target_col and not pd.api.types.is_numeric_dtype(train[c])
    ]

    if residual_cat_cols:
        for c in residual_cat_cols:
            train[c] = train[c].astype(str)
            test[c] = test[c].astype(str)
        train = pd.get_dummies(train, columns=residual_cat_cols, dummy_na=False)
        test = pd.get_dummies(test, columns=residual_cat_cols, dummy_na=False)

    y = train[target_col].astype(np.int8)
    X = train.drop(columns=[target_col], errors="ignore")
    X_test = test.copy()
    X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

    for df in (X, X_test):
        non_num = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        for c in non_num:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if non_num:
            df[non_num] = df[non_num].fillna(0)

    X = X.astype(np.float32)
    X_test = X_test.astype(np.float32)
    return X, y, X_test


def train_eval_cat(X_tr, y_tr, X_va, y_va, params):
    model = CatBoostClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    pred = model.predict_proba(X_va)[:, 1]
    ap = average_precision_score(y_va, pred)
    return float(ap)


def evaluate_cv_preprocessed(X, y, splitter, params, label: str):
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        ap = train_eval_cat(X_tr, y_tr, X_va, y_va, params)
        scores.append(ap)
        print(f"[{label}] fold {fold:02d} AP={ap:.6f}")
    arr = np.array(scores)
    print(f"[{label}] mean={arr.mean():.6f} std={arr.std():.6f} min={arr.min():.6f} max={arr.max():.6f}")
    return arr


def evaluate_cv_leakage_safe(train_raw, splitter, params, split_type: str = "stratified"):
    y_all = train_raw["fraud_bool"].astype(np.int8)
    scores = []

    if split_type == "group":
        groups = train_raw["_group_tmp_"].values
        splits = splitter.split(train_raw, y_all, groups)
    elif split_type == "time":
        splits = splitter.split(train_raw)
    else:
        splits = splitter.split(train_raw, y_all)

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        tr_df = train_raw.iloc[tr_idx].reset_index(drop=True)
        va_df = train_raw.iloc[va_idx].reset_index(drop=True)

        # Leakage-safe: preprocessing statistics are fitted only on tr_df.
        X_tr, y_tr, X_va = preprocess_guided(tr_df, va_df)
        y_va = va_df["fraud_bool"].astype(np.int8).reset_index(drop=True)

        ap = train_eval_cat(X_tr, y_tr, X_va, y_va, params)
        scores.append(ap)
        print(f"[LeakageSafe-{split_type}] fold {fold:02d} AP={ap:.6f}")

    arr = np.array(scores)
    print(
        f"[LeakageSafe-{split_type}] mean={arr.mean():.6f} std={arr.std():.6f} "
        f"min={arr.min():.6f} max={arr.max():.6f}"
    )
    return arr


def detect_group_column(df: pd.DataFrame):
    pats = ("customer", "account", "user", "device", "client", "member")
    candidates = [c for c in df.columns if any(p in c.lower() for p in pats)]
    candidates = [c for c in candidates if c != "fraud_bool"]
    for c in candidates:
        nunique = df[c].nunique(dropna=False)
        ratio = nunique / max(len(df), 1)
        if 0.001 <= ratio <= 0.95:
            return c
    return None


def detect_time_column(df: pd.DataFrame):
    pats = ("date", "time", "month", "day", "week")
    candidates = [c for c in df.columns if any(p in c.lower() for p in pats)]
    if "month_idx" in df.columns:
        return "month_idx"
    return candidates[0] if candidates else None


def run_suite(data_dir: Path, max_rows: int, repeats: int, seed_list):
    set_seed(SEED)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    if max_rows > 0 and len(train_raw) > max_rows:
        rs = np.random.RandomState(SEED)
        idx = rs.choice(len(train_raw), max_rows, replace=False)
        train_raw = train_raw.iloc[idx].reset_index(drop=True)

    print("train shape:", train_raw.shape)
    print("test shape:", test_raw.shape)
    print("fraud ratio:", float(train_raw["fraud_bool"].mean()))

    print("\n=== A) Leakage-Prone Baseline (global preprocessing before CV) ===")
    X_global, y_global, _ = preprocess_guided(train_raw, test_raw)
    cv_base = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    a_scores = evaluate_cv_preprocessed(X_global, y_global, cv_base, BEST_CAT_PARAMS, label="A-Global")

    print("\n=== B) Leakage-Safe CV (fold-local preprocessing) ===")
    cv_safe = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    b_scores = evaluate_cv_leakage_safe(train_raw, cv_safe, BEST_CAT_PARAMS, split_type="stratified")

    print("\n=== Leakage Gap (A - B) ===")
    print("A mean:", float(np.mean(a_scores)))
    print("B mean:", float(np.mean(b_scores)))
    print("gap   :", float(np.mean(a_scores) - np.mean(b_scores)))

    print("\n=== RepeatedStratifiedKFold Stability (leakage-safe) ===")
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=repeats, random_state=SEED)
    rep_scores = evaluate_cv_leakage_safe(train_raw, rskf, BEST_CAT_PARAMS, split_type="stratified")

    print("\n=== Multi-Seed Stability (leakage-safe) ===")
    multi_seed_rows = []
    y_all = train_raw["fraud_bool"].astype(np.int8)
    for s in seed_list:
        params_s = dict(BEST_CAT_PARAMS)
        params_s["random_seed"] = int(s)
        cv_s = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=int(s))
        fold_scores = []
        for tr_idx, va_idx in cv_s.split(train_raw, y_all):
            tr_df = train_raw.iloc[tr_idx].reset_index(drop=True)
            va_df = train_raw.iloc[va_idx].reset_index(drop=True)
            X_tr, y_tr, X_va = preprocess_guided(tr_df, va_df)
            y_va = va_df["fraud_bool"].astype(np.int8).reset_index(drop=True)
            ap = train_eval_cat(X_tr, y_tr, X_va, y_va, params_s)
            fold_scores.append(ap)

        arr = np.array(fold_scores)
        multi_seed_rows.append(
            {
                "seed": int(s),
                "mean_ap": float(arr.mean()),
                "std_ap": float(arr.std()),
                "min_ap": float(arr.min()),
                "max_ap": float(arr.max()),
            }
        )
        print(f"seed={s} mean={arr.mean():.6f} std={arr.std():.6f}")

    ms_df = pd.DataFrame(multi_seed_rows).sort_values("mean_ap", ascending=False)
    print("\nMulti-seed summary")
    print(ms_df)
    print("overall mean of seed means:", float(ms_df["mean_ap"].mean()))
    print("std across seed means:", float(ms_df["mean_ap"].std()))

    print("\n=== Optional Splitter Checks ===")
    group_col = detect_group_column(train_raw)
    if group_col is None:
        print("GroupKFold skipped: no reliable group column detected.")
    else:
        print("GroupKFold group column:", group_col)
        train_group = train_raw.copy()
        train_group["_group_tmp_"] = train_group[group_col].astype(str)
        gkf = GroupKFold(n_splits=N_SPLITS)
        _ = evaluate_cv_leakage_safe(train_group, gkf, BEST_CAT_PARAMS, split_type="group")

    time_col = detect_time_column(train_raw)
    if time_col is None:
        print("TimeSeriesSplit skipped: no time-like column detected.")
    else:
        print("TimeSeriesSplit time column:", time_col)
        train_time = train_raw.sort_values(time_col).reset_index(drop=True)
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        _ = evaluate_cv_leakage_safe(train_time, tscv, BEST_CAT_PARAMS, split_type="time")

    report = {
        "cat_params": BEST_CAT_PARAMS,
        "a_global_mean": float(np.mean(a_scores)),
        "b_leakage_safe_mean": float(np.mean(b_scores)),
        "ab_gap": float(np.mean(a_scores) - np.mean(b_scores)),
        "repeated_mean": float(np.mean(rep_scores)),
        "repeated_std": float(np.std(rep_scores)),
        "seed_summary": multi_seed_rows,
        "group_col_detected": group_col,
        "time_col_detected": time_col,
    }

    out_path = data_dir / "catboost_fixed_validation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nSaved:", out_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fixed-CatBoost validation suite (no Optuna), with leakage A/B and CV stability checks."
    )
    parser.add_argument("--data-dir", type=str, default=".", help="Directory containing train.csv/test.csv")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="If >0, sample this many train rows for faster diagnostics.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats for RepeatedStratifiedKFold.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="7,13,21,42,77",
        help="Comma-separated seeds for multi-seed stability.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    run_suite(
        data_dir=Path(args.data_dir),
        max_rows=int(args.max_rows),
        repeats=int(args.repeats),
        seed_list=seeds,
    )