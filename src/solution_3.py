from __future__ import annotations

import argparse
import itertools
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

TARGET = "fraud_bool"
ID_COL = "id"
AGE_CROSS_COLS = ["age_x_payment", "age_x_housing", "age_x_employment"]


@dataclass
class EvalResult:
    scenario: str
    n_features: int
    oof_ap: float
    fold_ap_mean: float
    fold_ap_std: float
    train_seconds: float


def build_catboost_params(profile: str, seed: int) -> Dict:
    if profile == "quick":
        return dict(
            loss_function="Logloss",
            eval_metric="PRAUC",
            auto_class_weights="Balanced",
            learning_rate=0.08,
            depth=7,
            l2_leaf_reg=6.0,
            iterations=1200,
            random_seed=seed,
            verbose=False,
        )
    if profile == "strong":
        return dict(
            loss_function="Logloss",
            eval_metric="PRAUC",
            auto_class_weights="Balanced",
            learning_rate=0.045,
            depth=8,
            l2_leaf_reg=8.0,
            iterations=4000,
            random_seed=seed,
            verbose=False,
        )
    raise ValueError(f"Unknown profile: {profile}")


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in (TARGET, ID_COL)]


def cat_feature_indices(df: pd.DataFrame, cols: List[str]) -> List[int]:
    cat_idx: List[int] = []
    for i, c in enumerate(cols):
        dtype = df[c].dtype
        if (
            pd.api.types.is_object_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(dtype)
            or not pd.api.types.is_numeric_dtype(dtype)
        ):
            cat_idx.append(i)
    return cat_idx


def eval_cv_catboost(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    *,
    n_folds: int,
    seed: int,
    profile: str,
) -> Tuple[EvalResult, np.ndarray, np.ndarray]:
    X = train_df[features]
    y = train_df[TARGET].astype(int)
    X_test = test_df[features]

    cat_idx = cat_feature_indices(train_df, features)
    params = build_catboost_params(profile=profile, seed=seed)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(train_df), dtype=np.float32)
    test_pred = np.zeros(len(test_df), dtype=np.float32)
    fold_scores: List[float] = []

    started = time.time()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y.iloc[va_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            cat_features=cat_idx,
            eval_set=(X_va, y_va),
            use_best_model=True,
            early_stopping_rounds=150,
            verbose=False,
        )

        p_va = model.predict_proba(X_va)[:, 1]
        p_te = model.predict_proba(X_test)[:, 1]

        oof[va_idx] = p_va.astype(np.float32)
        test_pred += (p_te.astype(np.float32) / n_folds)

        fold_ap = average_precision_score(y_va, p_va)
        fold_scores.append(float(fold_ap))
        print(f"  fold {fold}/{n_folds} AP={fold_ap:.6f}")

    oof_ap = float(average_precision_score(y, oof))
    elapsed = float(time.time() - started)

    result = EvalResult(
        scenario="",
        n_features=len(features),
        oof_ap=oof_ap,
        fold_ap_mean=float(np.mean(fold_scores)),
        fold_ap_std=float(np.std(fold_scores)),
        train_seconds=elapsed,
    )
    return result, oof, test_pred


def make_age_ablation_scenarios(full_features: List[str]) -> Dict[str, List[str]]:
    scenarios: Dict[str, List[str]] = {}
    present_age_cols = [c for c in AGE_CROSS_COLS if c in full_features]

    if len(present_age_cols) == 0:
        scenarios["fe_no_age_cols_found"] = full_features
        return scenarios

    for bits in itertools.product([0, 1], repeat=len(present_age_cols)):
        on_cols = [c for c, b in zip(present_age_cols, bits) if b == 1]
        off_cols = set(present_age_cols) - set(on_cols)
        feats = [c for c in full_features if c not in off_cols]

        key = "age_" + "".join(map(str, bits))
        label = []
        for c, b in zip(present_age_cols, bits):
            short = c.replace("age_x_", "")
            label.append(f"{short}:{'on' if b else 'off'}")
        scenarios[f"{key}__{'|'.join(label)}"] = feats

    return scenarios


def run_all(args: argparse.Namespace) -> None:
    train_raw = pd.read_csv(args.train_raw)
    test_raw = pd.read_csv(args.test_raw)
    train_fe = pd.read_csv(args.train_fe)
    test_fe = pd.read_csv(args.test_fe)

    if TARGET not in train_raw.columns:
        raise ValueError(f"{TARGET} not in raw train")
    if TARGET not in train_fe.columns:
        raise ValueError(f"{TARGET} not in FE train")

    results: List[Dict] = []
    predictions: Dict[str, np.ndarray] = {}

    print("\n[1/3] Baseline(raw) CV")
    base_features = feature_columns(train_raw)
    base_result, base_oof, base_test_pred = eval_cv_catboost(
        train_raw,
        test_raw,
        base_features,
        n_folds=args.n_folds,
        seed=args.seed,
        profile=args.profile,
    )
    base_result.scenario = "baseline_raw"
    results.append(base_result.__dict__)
    predictions[base_result.scenario] = base_test_pred

    if args.save_oof:
        pd.DataFrame({ID_COL: train_raw[ID_COL], "oof_pred": base_oof}).to_csv(
            os.path.join(args.out_dir, "oof_baseline_raw.csv"), index=False
        )

    print("\n[2/3] Full engineered CV")
    fe_features_full = feature_columns(train_fe)
    fe_result, fe_oof, fe_test_pred = eval_cv_catboost(
        train_fe,
        test_fe,
        fe_features_full,
        n_folds=args.n_folds,
        seed=args.seed,
        profile=args.profile,
    )
    fe_result.scenario = "corrstack_full"
    results.append(fe_result.__dict__)
    predictions[fe_result.scenario] = fe_test_pred

    if args.save_oof:
        pd.DataFrame({ID_COL: train_fe[ID_COL], "oof_pred": fe_oof}).to_csv(
            os.path.join(args.out_dir, "oof_corrstack_full.csv"), index=False
        )

    print("\n[3/3] age_x_* ablation CV")
    ablation_scenarios = make_age_ablation_scenarios(fe_features_full)
    for name, feats in ablation_scenarios.items():
        print(f"\n- scenario: {name}")
        r, oof, test_pred = eval_cv_catboost(
            train_fe,
            test_fe,
            feats,
            n_folds=args.n_folds,
            seed=args.seed,
            profile=args.profile,
        )
        r.scenario = name
        results.append(r.__dict__)
        predictions[name] = test_pred

        if args.save_oof:
            pd.DataFrame({ID_COL: train_fe[ID_COL], "oof_pred": oof}).to_csv(
                os.path.join(args.out_dir, f"oof_{name}.csv"), index=False
            )

    res_df = pd.DataFrame(results).sort_values("oof_ap", ascending=False).reset_index(drop=True)

    baseline_ap = float(res_df.loc[res_df["scenario"] == "baseline_raw", "oof_ap"].iloc[0])
    res_df["delta_vs_baseline"] = res_df["oof_ap"] - baseline_ap

    best_row = res_df.iloc[0]
    best_name = str(best_row["scenario"])

    os.makedirs(args.out_dir, exist_ok=True)

    results_path = os.path.join(args.out_dir, "solution_3_results.csv")
    res_df.to_csv(results_path, index=False)

    ablation_view = res_df[res_df["scenario"].str.startswith("age_")].copy()
    full_ap = float(res_df.loc[res_df["scenario"] == "corrstack_full", "oof_ap"].iloc[0])
    ablation_view["delta_vs_corrstack_full"] = ablation_view["oof_ap"] - full_ap
    ablation_path = os.path.join(args.out_dir, "solution_3_age_ablation.csv")
    ablation_view.to_csv(ablation_path, index=False)

    # Save submissions for baseline, full, and best scenario.
    def _save_submission(name: str, test_frame: pd.DataFrame) -> None:
        pred = predictions[name]
        sub = pd.DataFrame({ID_COL: test_frame[ID_COL], TARGET: pred})
        sub.to_csv(os.path.join(args.out_dir, f"submission_solution_3_{name}.csv"), index=False)

    _save_submission("baseline_raw", test_raw)
    _save_submission("corrstack_full", test_fe)
    if best_name not in ("baseline_raw", "corrstack_full"):
        _save_submission(best_name, test_fe)

    summary_path = os.path.join(args.out_dir, "solution_3_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Solution 3 summary\n")
        f.write(f"n_folds={args.n_folds}, profile={args.profile}, seed={args.seed}\n")
        f.write(f"baseline_raw_ap={baseline_ap:.6f}\n")
        f.write(f"best_scenario={best_name}\n")
        f.write(f"best_oof_ap={float(best_row['oof_ap']):.6f}\n")
        f.write(f"best_delta_vs_baseline={float(best_row['delta_vs_baseline']):.6f}\n")
        f.write(f"results_csv={results_path}\n")
        f.write(f"ablation_csv={ablation_path}\n")

    print("\n=== DONE ===")
    print(res_df[["scenario", "n_features", "oof_ap", "delta_vs_baseline"]].head(12).to_string(index=False))
    print(f"\n[OK] results: {results_path}")
    print(f"[OK] ablation: {ablation_path}")
    print(f"[OK] summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Solution 3: CatBoost CV + age_x_* ablation")
    ap.add_argument("--train-raw", default="train.csv")
    ap.add_argument("--test-raw", default="test.csv")
    ap.add_argument("--train-fe", default="train_corrstack_features.csv")
    ap.add_argument("--test-fe", default="test_corrstack_features.csv")
    ap.add_argument("--out-dir", default="solution_3_output")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--profile", choices=["quick", "strong"], default="quick")
    ap.add_argument("--save-oof", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    run_all(parse_args())
