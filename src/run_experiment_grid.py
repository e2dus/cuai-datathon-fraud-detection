# -*- coding: utf-8 -*-
"""
전처리_가이드 / solution.ipynb / cuai_pipeline 조합 실험.

  python run_experiment_grid.py --preset wide --random-sample 64
  python run_experiment_grid.py --preset full --random-sample 200 --n-folds 5

결과: experiment_grid_results.csv — combo_description(한글 설명), cfg_key, 옵션별 컬럼.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from cuai_pipeline import PipelineConfig, TARGET, run_pipeline

try:
    import xgboost as xgb
except ImportError:
    print("pip install xgboost 필요", file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# 가이드·파이프라인 옵션별 짧은 한글 설명 (CSV combo_description / 콘솔)
# ---------------------------------------------------------------------------
OPTION_LABELS_KO: Dict[str, str] = {
    "woe_category_flags": "WOE전 고위험범주 플래그(AC/BA/TELEAPP/windows/CC)",
    "use_sentinel": "-1 센티넬→unknown+NaN(주소·신용·세션 등)",
    "kde_hotzones": "KDE 기반 사기 핫존 이진 피처",
    "guide_derived": "가이드 파생(이름-이메일·신용·초기이체·요일·교차 등)",
    "drop_device_fraud_history": "device_fraud_history 컬럼 제외(가이드: 분리력 낮음)",
    "similarity_filter": "정상/사기 분포 유사 피처 제거(KS+Wasserstein)",
    "similarity_ks_thr": "유사도 필터 KS 상한(작을수록 엄격)",
    "similarity_w_thr": "유사도 필터 Wasserstein 정규화 상한(작을수록 엄격)",
    "corr_prune": "피처 간 고상관(|r|) 시 타겟 약한 쪽 제거",
    "corr_threshold": "상관 pruning 임계 |r|",
    "init_amt_quantile": "init_transfer_amt 고액 플래그 분위수(가이드: 고액=사기 신호)",
}


def describe_combo_ko(cfg: PipelineConfig) -> str:
    """각 조합이 무엇인지 한 줄로 요약 (CSV combo_description)."""
    parts: List[str] = []
    d = asdict(cfg)
    d.pop("name", None)
    for key in sorted(d.keys()):
        val = d[key]
        label = OPTION_LABELS_KO.get(key, key)
        if isinstance(val, bool):
            parts.append(f"{label}: {'켜짐' if val else '끔'}")
        else:
            parts.append(f"{label}: {val}")
    return " / ".join(parts)


def short_combo_name(cfg: PipelineConfig) -> str:
    """파일명·로그용 짧은 이름 (해시)."""
    h = hashlib.md5(cfg.to_key().encode("utf-8")).hexdigest()[:10]
    return f"h{h}"


# ---------------------------------------------------------------------------
# 그리드 프리셋 (전처리_가이드.md + cuai_pipeline 스위치 최대 반영)
# ---------------------------------------------------------------------------
def grid_preset_narrow() -> Dict[str, List]:
    """스크리닝: 핵심 5스위치만 (32조합)."""
    return {
        "woe_category_flags": [True, False],
        "guide_derived": [True, False],
        "kde_hotzones": [True, False],
        "similarity_filter": [True, False],
        "corr_prune": [True, False],
    }


def grid_preset_wide() -> Dict[str, List]:
    """
    넓은 탐색: 파이프라인 불리언 전부 + init 분위.
    2^8 * 3 = 768 조합.
    """
    return {
        "woe_category_flags": [True, False],
        "use_sentinel": [True, False],
        "kde_hotzones": [True, False],
        "guide_derived": [True, False],
        "drop_device_fraud_history": [True, False],
        "similarity_filter": [True, False],
        "corr_prune": [True, False],
        "init_amt_quantile": [0.99, 0.995, 0.999],
    }


def grid_preset_full() -> Dict[str, List]:
    """
    최대 폭: wide + 상관/유사도 임계 스윕.
    768 * 3 * 3 * 3 = 20736 조합 — 반드시 --random-sample 권장.
    """
    g = grid_preset_wide()
    g["corr_threshold"] = [0.85, 0.90, 0.95]
    g["similarity_ks_thr"] = [0.02, 0.03, 0.05]
    g["similarity_w_thr"] = [0.03, 0.05, 0.08]
    return g


PRESETS = {
    "narrow": grid_preset_narrow,
    "wide": grid_preset_wide,
    "full": grid_preset_full,
}


def _defaults_for_pipeline_config() -> Dict[str, Any]:
    return asdict(PipelineConfig(name="_"))


def build_pipeline_config(grid_keys: Tuple[str, ...], combo: Tuple, name: str) -> PipelineConfig:
    d = _defaults_for_pipeline_config()
    d.pop("name", None)
    for k, v in zip(grid_keys, combo):
        d[k] = v
    d["name"] = name
    return PipelineConfig(**d)


def list_configs_from_grid(
    grid: Dict[str, List],
    *,
    random_sample: Optional[int],
    max_runs: Optional[int],
    seed: int,
) -> List[PipelineConfig]:
    keys = tuple(grid.keys())
    values = [grid[k] for k in keys]
    combos = list(itertools.product(*values))
    rng = np.random.RandomState(seed)
    if random_sample is not None:
        if random_sample < len(combos):
            pick = rng.choice(len(combos), size=random_sample, replace=False)
            combos = [combos[i] for i in pick]
        # random_sample >= len(combos) 이면 전체 조합 유지
    # random_sample 없으면 itertools 순서 그대로(재현 용이)
    if max_runs is not None:
        combos = combos[:max_runs]
    out: List[PipelineConfig] = []
    for i, combo in enumerate(combos):
        tmp = build_pipeline_config(keys, combo, "_")
        name = f"run{i+1:04d}_{short_combo_name(tmp)}"
        out.append(build_pipeline_config(keys, combo, name))
    return out


def train_xgb_cv(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_folds: int,
    seed: int,
    model_profile: str,
) -> Tuple[float, float, float, List[float]]:
    neg, pos = int((y == 0).sum()), int((y == 1).sum())
    spw = neg / max(pos, 1)

    # 경량 기본(실험용): 이전보다 트리 깊이 약간↑ (과적합 완화로 min_child_weight·정규화 유지)
    if model_profile == "light":
        params = dict(
            objective="binary:logistic",
            eval_metric="aucpr",
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=10,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=0.08,
            reg_lambda=1.2,
            n_estimators=128,
            max_bin=128,
            random_state=seed,
            verbosity=0,
            n_jobs=-1,
            tree_method="hist",
        )
        early = 18
    elif model_profile == "quick":
        params = dict(
            objective="binary:logistic",
            eval_metric="aucpr",
            learning_rate=0.15,
            max_depth=4,
            min_child_weight=14,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.0,
            reg_lambda=1.0,
            n_estimators=72,
            max_bin=96,
            random_state=seed,
            verbosity=0,
            n_jobs=-1,
            tree_method="hist",
        )
        early = 12
    else:  # heavy
        params = dict(
            objective="binary:logistic",
            eval_metric="aucpr",
            learning_rate=0.035,
            max_depth=10,
            min_child_weight=4,
            subsample=0.88,
            colsample_bytree=0.88,
            colsample_bylevel=0.9,
            reg_alpha=0.12,
            reg_lambda=2.5,
            gamma=0.08,
            n_estimators=3000,
            max_bin=512,
            random_state=seed,
            verbosity=0,
            n_jobs=-1,
            tree_method="hist",
        )
        early = 100

    params["scale_pos_weight"] = float(spw)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=np.float32)
    fold_aps: List[float] = []

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        X_tr = X_tr.fillna(X_tr.median(numeric_only=True))
        X_va = X_va.fillna(X_tr.median(numeric_only=True))

        model = xgb.XGBClassifier(**params, early_stopping_rounds=early)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        p = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = p.astype(np.float32)
        fold_aps.append(average_precision_score(y_va, p))

    oof_ap = average_precision_score(y, oof)
    return oof_ap, float(np.mean(fold_aps)), float(np.std(fold_aps)), fold_aps


def _coerce_cell(key: str, val: Any, default: Any) -> Any:
    if pd.isna(val):
        return default
    if isinstance(default, bool):
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes")
        return bool(val)
    if isinstance(default, float) and key in (
        "init_amt_quantile",
        "corr_threshold",
        "similarity_ks_thr",
        "similarity_w_thr",
    ):
        return float(val)
    return val


def _row_to_cfg_key(row: pd.Series) -> str:
    """CSV 한 행으로 PipelineConfig 복원 후 cfg_key (resume 매칭용)."""
    defaults = _defaults_for_pipeline_config()
    defaults.pop("name", None)
    for k in list(defaults.keys()):
        if k in row.index and pd.notna(row[k]):
            defaults[k] = _coerce_cell(k, row[k], defaults[k])
    cfg = PipelineConfig(name="resume", **defaults)
    return cfg.to_key()


def main():
    ap = argparse.ArgumentParser(description="CUAI 전처리 조합 그리드 실험")
    ap.add_argument("--data-dir", default=".", help="train.csv, test.csv 위치")
    ap.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="wide",
        help="narrow(32) / wide(768) / full(20736 근방)",
    )
    ap.add_argument(
        "--random-sample",
        type=int,
        default=None,
        help="전체 조합 중 무작위로 N개만 (큰 프리셋 필수 권장)",
    )
    ap.add_argument("--max-runs", type=int, default=None, help="실행 상한(샘플 후 앞에서부터 자름)")
    ap.add_argument("--n-folds", type=int, default=5, help="기본 5폴드")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--model",
        choices=["light", "quick", "heavy"],
        default="light",
        help="light=기본 실험용(빠름), quick=더 가벼움, heavy=느림",
    )
    ap.add_argument("--out", default="experiment_grid_results.csv")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="기존 CSV에 성공한 cfg_key는 건너뜀",
    )
    args = ap.parse_args()

    train_path = os.path.join(args.data_dir, "train.csv")
    test_path = os.path.join(args.data_dir, "test.csv")
    if not os.path.isfile(train_path):
        print(f"없음: {train_path}", file=sys.stderr)
        sys.exit(1)

    train_full = pd.read_csv(train_path)
    test_full = pd.read_csv(test_path) if os.path.isfile(test_path) else None

    grid = PRESETS[args.preset]()
    grid_keys_list = list(grid.keys())
    n_combos = int(np.prod([len(v) for v in grid.values()]))

    configs = list_configs_from_grid(
        grid,
        random_sample=args.random_sample,
        max_runs=args.max_runs,
        seed=args.seed,
    )

    print(f"[프리셋] {args.preset}")
    print(f"[그리드 키] {grid_keys_list}")
    print(f"[이론 조합 수] {n_combos} → 이번 실행 {len(configs)}회 (random_sample={args.random_sample}, max_runs={args.max_runs})")
    print(f"[모델] {args.model}, folds={args.n_folds}, seed={args.seed}, resume={args.resume}\n")

    done_keys: set = set()
    rows: List[dict] = []
    if args.resume and os.path.isfile(args.out):
        prev = pd.read_csv(args.out)
        if "oof_ap" in prev.columns:
            ok = prev["oof_ap"].notna()
            if "cfg_key" in prev.columns:
                done_keys = set(prev.loc[ok, "cfg_key"].astype(str).tolist())
            else:
                for _, row in prev.loc[ok].iterrows():
                    try:
                        done_keys.add(_row_to_cfg_key(row))
                    except Exception:
                        pass
            rows = prev.to_dict("records")
            print(f"[resume] 완료 스킵: {len(done_keys)}개\n")

    t0 = time.perf_counter()
    n_executed = 0

    for cfg in configs:
        cfg_key = cfg.to_key()
        if args.resume and cfg_key in done_keys:
            print(f"[SKIP] {cfg.name}")
            continue

        combo_desc = describe_combo_ko(cfg)
        t_run = time.perf_counter()
        try:
            train, _test, feats = run_pipeline(
                train_full.copy(),
                test_full.copy() if test_full is not None else train_full.iloc[:0].copy(),
                cfg,
            )
            X = train[feats].astype(np.float32)
            y = train[TARGET].astype(np.int8)
            oof_ap, cv_mean, cv_std, fold_aps = train_xgb_cv(
                X,
                y,
                n_folds=args.n_folds,
                seed=args.seed,
                model_profile=args.model,
            )
        except Exception as e:
            print(f"[FAIL] {cfg.name}  {e}")
            row_fail = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "cfg_key": cfg_key,
                "name": cfg.name,
                "combo_description": combo_desc,
                "preset": args.preset,
                "model": args.model,
                "oof_ap": np.nan,
                "cv_mean": np.nan,
                "cv_std": np.nan,
                "n_features": np.nan,
                "sec": round(time.perf_counter() - t_run, 2),
                "error": str(e),
            }
            for k in grid_keys_list:
                row_fail[k] = getattr(cfg, k)
            rows.append(row_fail)
            n_executed += 1
            pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")
            continue

        row_ok = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "cfg_key": cfg_key,
            "name": cfg.name,
            "combo_description": combo_desc,
            "preset": args.preset,
            "model": args.model,
            "oof_ap": round(oof_ap, 5),
            "cv_mean": round(cv_mean, 5),
            "cv_std": round(cv_std, 5),
            "n_features": len(feats),
            "sec": round(time.perf_counter() - t_run, 2),
            "error": "",
        }
        for k in grid_keys_list:
            row_ok[k] = getattr(cfg, k)
        rows.append(row_ok)
        n_executed += 1
        done_keys.add(cfg_key)
        print(f"[OK] {cfg.name}")
        print(f"     {combo_desc[:200]}{'...' if len(combo_desc) > 200 else ''}")
        print(
            f"     OOF={oof_ap:.5f}  CV={cv_mean:.5f}±{cv_std:.5f}  n_feat={len(feats)}  {row_ok['sec']}s\n"
        )

        pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")

    elapsed = time.perf_counter() - t0
    print(f"\n완료: 이번 실행 {n_executed}회, CSV 행 수 {len(rows)}, 총 {elapsed/60:.1f}분, 저장 {args.out}")


if __name__ == "__main__":
    main()
