from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class CurriculumResult:
    model: str
    oof_ap: float
    cv_mean: float
    cv_std: float


def _default_catboost_params(seed: int) -> Dict:
    return {
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 10,
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": seed,
        "early_stopping_rounds": 50,
        "auto_class_weights": "Balanced",
        "verbose": False,
    }


def fit_fold_negative_then_positive(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int,
    base_params: Optional[Dict] = None,
    stage1_iterations: int = 350,
    stage2_iterations: int = 850,
    stage1_positive_weight: float = 0.2,
    stage2_positive_weight: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2-stage curriculum training without undersampling:
    1) Negative-first: train on full data with low weight for label-1.
    2) Positive-boost: continue on full data with high weight for label-1.
    """
    y_tr = np.asarray(y_tr)
    if set(np.unique(y_tr)) != {0, 1}:
        raise ValueError("y_tr must contain both labels 0 and 1 for binary classification.")

    neg_idx = np.where(y_tr == 0)[0]
    pos_idx = np.where(y_tr == 1)[0]

    if len(neg_idx) == 0 or len(pos_idx) == 0:
        raise ValueError("Both classes must be present in each fold train split.")

    params = _default_catboost_params(seed)
    if base_params:
        params.update(base_params)

    params_stage1 = dict(params)
    params_stage2 = dict(params)
    params_stage1["iterations"] = stage1_iterations
    params_stage2["iterations"] = stage2_iterations

    w_stage1 = np.ones(len(y_tr), dtype=np.float32)
    w_stage2 = np.ones(len(y_tr), dtype=np.float32)
    w_stage1[pos_idx] = float(stage1_positive_weight)
    w_stage2[pos_idx] = float(stage2_positive_weight)

    m1 = CatBoostClassifier(**params_stage1)
    m1.fit(
        X_tr,
        y_tr,
        sample_weight=w_stage1,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Continue training from stage-1 model while increasing positive-class influence.
    m2 = CatBoostClassifier(**params_stage2)
    m2.fit(
        X_tr,
        y_tr,
        sample_weight=w_stage2,
        eval_set=[(X_val, y_val)],
        init_model=m1,
        verbose=False,
    )

    val_pred = m2.predict_proba(X_val)[:, 1]
    test_pred = m2.predict_proba(X_test)[:, 1]
    return val_pred, test_pred


def run_cv_negative_then_positive(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int = 42,
    n_splits: int = 5,
    base_params: Optional[Dict] = None,
    stage1_iterations: int = 350,
    stage2_iterations: int = 850,
    stage1_positive_weight: float = 0.2,
    stage2_positive_weight: float = 3.0,
    model_name: str = "CatBoost_negative_then_positive",
) -> Tuple[CurriculumResult, np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = np.zeros(len(y), dtype=np.float32)
    test_p = np.zeros(len(X_test), dtype=np.float32)
    fold_aps = []

    print(f"=== {model_name} ===")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        val_pred, te_pred = fit_fold_negative_then_positive(
            X_tr,
            y_tr,
            X_val,
            y_val,
            X_test,
            seed=seed + fold,
            base_params=base_params,
            stage1_iterations=stage1_iterations,
            stage2_iterations=stage2_iterations,
            stage1_positive_weight=stage1_positive_weight,
            stage2_positive_weight=stage2_positive_weight,
        )

        oof[val_idx] = val_pred.astype(np.float32)
        test_p += te_pred.astype(np.float32) / n_splits

        ap = float(average_precision_score(y_val, val_pred))
        fold_aps.append(ap)
        print(
            f"  Fold {fold} AP={ap:.5f} "
            f"| train_pos_rate={np.mean(y_tr):.4%} val_pos_rate={np.mean(y_val):.4%}"
        )

    oof_ap = float(average_precision_score(y, oof))
    cv_mean = float(np.mean(fold_aps))
    cv_std = float(np.std(fold_aps))
    print(f"  OOF AP={oof_ap:.5f} | CV mean±std={cv_mean:.5f} ± {cv_std:.5f}")

    result = CurriculumResult(
        model=model_name,
        oof_ap=oof_ap,
        cv_mean=cv_mean,
        cv_std=cv_std,
    )
    return result, oof, test_p


# Notebook usage example:
# from netgative_label_curriculum_catboost import run_cv_negative_then_positive
#
# cb_params = globals().get("CATBOOST_BEST_PARAMS", None)
# r, oof_pred, test_preds = run_cv_negative_then_positive(
#     X, y, X_test,
#     seed=SEED,
#     n_splits=5,
#     base_params=cb_params,
#     stage1_iterations=300,
#     stage2_iterations=900,
#     stage1_positive_weight=0.2,
#     stage2_positive_weight=3.0,
# )
# print(r)
