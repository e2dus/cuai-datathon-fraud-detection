# -*- coding: utf-8 -*-
"""
baseline.ipynb 를 읽어
1) §2·§3 전처리(최상위 코드 상수 + 저장 출력)·§4 코드(AST)에서 하이퍼·설정 추출
2) §4 저장 출력의 [BASELINE_RUN_MANIFEST]로 실행 시점 _scale_pos·n_features 등 스냅샷 병합
3) 저장된 stdout에서 Optuna best params / CV 표 병합
4) Excel(및 JSON)로 저장 — JSON 최상단 `experiment_summary`에 전처리·seed·CV·모델설정을 한 번에 정리

사용: python export_baseline_experiment_excel.py
"""
from __future__ import annotations

import ast
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
NB_PATH = ROOT / "baseline.ipynb"
OUT_PRIMARY = ROOT / "baseline_experiment_log.xlsx"
OUT_FALLBACK = ROOT / "baseline_experiment_log_from_nb.xlsx"
JSON_OUT = ROOT / "baseline_run_hyperparams.json"


def _join_source(cell: dict) -> str:
    return "".join(cell.get("source") or [])


def _stream_outputs(cell: dict) -> str:
    parts: list[str] = []
    for o in cell.get("outputs") or []:
        if o.get("output_type") == "stream":
            parts.append("".join(o.get("text") or []))
    return "".join(parts)


def _expr_str(e: ast.expr | None) -> str:
    if e is None:
        return ""
    return ast.unparse(e)


def _dict_from_call_keywords(call: ast.Call) -> dict[str, str]:
    out: dict[str, str] = {}
    for kw in call.keywords:
        if kw.arg:
            out[kw.arg] = _expr_str(kw.value)
    return out


def _dict_from_ast_dict(d: ast.Dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in zip(d.keys, d.values):
        if k is None:
            continue
        key = ast.literal_eval(k) if isinstance(k, ast.Constant) else _expr_str(k)
        out[str(key)] = _expr_str(v)
    return out


def _find_assign_str(tree: ast.Module, name: str) -> str | None:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return _expr_str(node.value)
    return None


def _find_assign_int(tree: ast.Module, name: str) -> str | None:
    s = _find_assign_str(tree, name)
    return s


def parse_sec4_hyperparams(source: str) -> list[dict[str, str]]:
    """§4 셀에서 N_SPLITS, xgb_params, lgb_params, HGB, SGD, lgb.train 옵션 추출."""
    rows: list[dict[str, str]] = []
    tree = ast.parse(source)

    ns = _find_assign_str(tree, "N_SPLITS")
    if ns is not None:
        rows.append({"블록": "CV_설정", "이름": "N_SPLITS", "값": ns, "출처": "코드"})

    sp = _find_assign_str(tree, "_scale_pos")
    if sp is not None:
        rows.append({"블록": "CV_설정", "이름": "_scale_pos", "값": sp, "출처": "코드(실행 시 계산)"})

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if not isinstance(t, ast.Name):
                    continue
                if t.id == "xgb_params" and isinstance(node.value, ast.Call):
                    for k, v in _dict_from_call_keywords(node.value).items():
                        rows.append({"블록": "XGBoost", "이름": k, "값": v, "출처": "코드"})
                elif t.id == "lgb_params" and isinstance(node.value, ast.Dict):
                    for k, v in _dict_from_ast_dict(node.value).items():
                        rows.append({"블록": "LightGBM", "이름": k, "값": v, "출처": "코드"})
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "HistGradientBoostingClassifier":
                for k, v in _dict_from_call_keywords(node).items():
                    rows.append({"블록": "HistGradientBoosting", "이름": k, "값": v, "출처": "코드"})
            if isinstance(node.func, ast.Name) and node.func.id == "SGDClassifier":
                for k, v in _dict_from_call_keywords(node).items():
                    rows.append({"블록": "Linear_SGD", "이름": k, "값": v, "출처": "코드"})
            if isinstance(node.func, ast.Attribute) and node.func.attr == "train":
                base = node.func.value
                if not (isinstance(base, ast.Name) and base.id == "lgb"):
                    continue
                for kw in node.keywords:
                    if kw.arg and kw.arg not in ("params", "train_set", "valid_sets", "callbacks"):
                        rows.append(
                            {
                                "블록": "LightGBM_train",
                                "이름": kw.arg,
                                "값": _expr_str(kw.value),
                                "출처": "코드",
                            }
                        )

    return rows


def parse_optuna_settings(source: str) -> list[dict[str, str]]:
    """§4-b Optuna 셀에서 TUNE_N_ROWS 등 대입문 추출."""
    rows: list[dict[str, str]] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in (
                    "TUNE_N_ROWS",
                    "TUNE_CV",
                    "N_TRIALS",
                    "CATBOOST_ITER",
                    "EARLY_STOP",
                ):
                    rows.append(
                        {
                            "블록": "Optuna_튜닝_설정",
                            "이름": t.id,
                            "값": _expr_str(node.value),
                            "출처": "코드",
                        }
                    )
    return rows


def parse_baseline_run_manifest(text: str) -> list[dict[str, str]]:
    """§4 출력의 [BASELINE_RUN_MANIFEST] key=value ... 스냅샷."""
    rows: list[dict[str, str]] = []
    m = re.search(r"\[BASELINE_RUN_MANIFEST\]([^\n]+)", text)
    if not m:
        return rows
    for part in m.group(1).strip().split():
        if "=" not in part:
            continue
        k, _, v = part.partition("=")
        rows.append(
            {
                "블록": "실행_스냅샷",
                "이름": k.strip(),
                "값": v.strip(),
                "출처": "§4_노트북_저장_출력",
            }
        )
    return rows


# 데이터프레임·중간 결과물 등 재현에 불필요한 최상위 이름 (AST 대입 스킵)
_SKIP_PREPROC_TOP_NAMES = frozenset(
    {
        "train",
        "test",
        "_age_tr",
        "_age_te",
        "_ohe_tr",
        "_ohe_te",
        "cat_cols",
        "feature_cols",
        "X",
        "y",
        "X_test",
        "_g",
        "_ts",
        "_ss",
    }
)


def _is_skip_preproc_rhs(value: ast.expr) -> bool:
    """pd.read_csv / concat / get_dummies 등 데이터 적재·중간 테이블 생성은 제외."""
    if not isinstance(value, ast.Call):
        return False
    fn = value.func
    if isinstance(fn, ast.Attribute) and fn.attr in (
        "read_csv",
        "concat",
        "get_dummies",
        "reindex",
    ):
        return True
    return False


def _iter_top_level_assign_pairs(stmt: ast.stmt) -> list[tuple[str, ast.expr]]:
    """모듈 최상위 `a = ...` 또는 `a, b = ..., ...` 만 (함수·for 안은 제외)."""
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return []
    t = stmt.targets[0]
    if isinstance(t, ast.Name):
        return [(t.id, stmt.value)]
    if isinstance(t, ast.Tuple) and isinstance(stmt.value, ast.Tuple):
        if len(t.elts) != len(stmt.value.elts):
            return []
        out: list[tuple[str, ast.expr]] = []
        for a, b in zip(t.elts, stmt.value.elts):
            if isinstance(a, ast.Name):
                out.append((a.id, b))
        return out
    return []


def parse_preprocessing_top_level_assigns(
    source: str,
    block: str,
    extra_skip: frozenset[str] | None = None,
) -> list[dict[str, str]]:
    """§2·§3 셀: 모듈 최상위 대입문을 표로 (데이터 로드·중간 DF 제외)."""
    skip = _SKIP_PREPROC_TOP_NAMES | (extra_skip or frozenset())
    rows: list[dict[str, str]] = []
    if not source.strip():
        return rows
    tree = ast.parse(source)
    for stmt in tree.body:
        for name, val in _iter_top_level_assign_pairs(stmt):
            if name in skip:
                continue
            if _is_skip_preproc_rhs(val):
                continue
            rows.append(
                {
                    "블록": block,
                    "이름": name,
                    "값": _expr_str(val),
                    "출처": "코드",
                }
            )
    return rows


def parse_section_stdout_as_rows(text: str, block: str, source_label: str) -> list[dict[str, str]]:
    """노트북에 저장된 해당 섹션 stdout 한 줄씩 (실행 시점 요약)."""
    rows: list[dict[str, str]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        rows.append(
            {
                "블록": block,
                "이름": f"stdout_{i:03d}",
                "값": s,
                "출처": source_label,
            }
        )
    return rows


def parse_optuna_best_from_stdout(text: str) -> list[dict[str, str]]:
    """노트북 출력의 best trial params / Best mean AP."""
    rows: list[dict[str, str]] = []
    # "Best mean AP (튜닝용 3-fold, ...): 0.12345" — 첫 숫자가 3-fold의 3이 되지 않도록 ':' 뒤 점수만 캡처
    m = re.search(r"Best mean AP.*?:\s*([\d.]+)", text)
    if m:
        rows.append(
            {
                "블록": "Optuna_실행결과",
                "이름": "Best_mean_AP_튜닝셋",
                "값": m.group(1),
                "출처": "노트북_저장_출력",
            }
        )
    m = re.search(r"best trial params:\s*(\{.*\})", text, re.DOTALL)
    if m:
        raw = m.group(1).strip()
        try:
            d = ast.literal_eval(raw)
            if isinstance(d, dict):
                for k, v in d.items():
                    rows.append(
                        {
                            "블록": "Optuna_best_trial",
                            "이름": str(k),
                            "값": repr(v),
                            "출처": "노트북_저장_출력",
                        }
                    )
        except (SyntaxError, ValueError):
            rows.append(
                {
                    "블록": "Optuna_best_trial",
                    "이름": "raw",
                    "값": raw[:5000],
                    "출처": "노트북_저장_출력",
                }
            )
    return rows


def _rows_by_block(hp_rows: list[dict[str, str]], block: str) -> dict[str, str]:
    return {r["이름"]: r["값"] for r in hp_rows if r.get("블록") == block}


def build_experiment_summary(
    hp_rows: list[dict[str, str]],
    cv_df: pd.DataFrame | None,
    sec2_stdout: str,
    sec3_stdout: str,
) -> dict:
    """사람이 읽는 실험 요약: 전처리 무엇을 했는지 + seed + CV + 모델별 설정."""
    seed = next((r["값"] for r in hp_rows if r.get("블록") == "공통" and r.get("이름") == "SEED"), None)
    s2_lines = [ln.strip() for ln in sec2_stdout.splitlines() if ln.strip()]
    s3_lines = [ln.strip() for ln in sec3_stdout.splitlines() if ln.strip()]

    s2c = _rows_by_block(hp_rows, "전처리_§2_코드상수")
    s3c = dict(_rows_by_block(hp_rows, "전처리_§3_코드상수"))
    if "TARGET" in s3c and "DROP_COLS" in s3c:
        t = s3c["TARGET"].strip().strip("'\"")
        s3c["DROP_COLS_해석"] = f"['id', '{t}']"

    cv_rows: list[dict] = []
    best: dict | None = None
    if cv_df is not None and not cv_df.empty and "OOF_AP" in cv_df.columns:
        cv_sorted = cv_df.sort_values("OOF_AP", ascending=False)
        cv_rows = cv_sorted.to_dict(orient="records")
        best = cv_rows[0] if cv_rows else None

    train_blocks = (
        "CV_설정",
        "Optuna_튜닝_설정",
        "Optuna_실행결과",
        "XGBoost",
        "LightGBM",
        "LightGBM_train",
        "HistGradientBoosting",
        "Linear_SGD",
        "Optuna_best_trial",
        "CatBoost_최종추정_튜닝후",
        "CatBoost_기본_else분기",
        "실행_스냅샷",
    )
    models_hp: dict[str, dict[str, str]] = {}
    for b in train_blocks:
        d = _rows_by_block(hp_rows, b)
        if d:
            models_hp[b] = d

    one = ""
    if best is not None and seed is not None:
        one = (
            f"SEED={seed} | 최고 OOF AP: {best['model']} = {best['OOF_AP']:.6f} "
            f"(fold mean {best['CV_fold_mean_AP']:.6f} ± {best['CV_fold_std_AP']:.6f})"
        )
    elif best is not None:
        one = (
            f"최고 OOF AP: {best['model']} = {best['OOF_AP']:.6f} "
            f"(fold mean {best['CV_fold_mean_AP']:.6f} ± {best['CV_fold_std_AP']:.6f})"
        )

    return {
        "한줄요약": one,
        "seed": seed,
        "평가지표": "Average Precision (OOF) — CV 요약의 oof_ap",
        "무엇을_했는지": {
            "설명": (
                "§2: train/test CSV 로드 후 컬럼별 클립·분위수·파생 플래그·log1p 등 순차 적용. "
                "§3: age_bucket 원핫, object 범주형 원핫, 학습 피처 확정."
            ),
            "§2_실행로그_한줄씩": s2_lines,
            "§3_실행로그_한줄씩": s3_lines,
            "§2_코드에서_정한_상수_및_식": s2c,
            "§3_코드에서_정한_상수": s3c,
        },
        "검증_성능_OOF_AP_내림차순": cv_rows,
        "최고_모델": best,
        "모델_및_튜닝_설정_블록별": models_hp,
    }


def experiment_summary_to_dataframe(summary: dict) -> pd.DataFrame:
    """Excel `실험요약` 시트용 (긴 필드는 JSON 문자열)."""
    mp = summary.get("무엇을_했는지", {})
    rows = [
        {"항목": "한줄요약", "내용": summary.get("한줄요약", "")},
        {"항목": "seed", "내용": str(summary.get("seed", ""))},
        {"항목": "평가지표", "내용": summary.get("평가지표", "")},
        {"항목": "실험_무엇을_했는지_설명", "내용": mp.get("설명", "")},
        {"항목": "§2_전처리_실행요약_전체", "내용": "\n".join(mp.get("§2_실행로그_한줄씩", []))},
        {"항목": "§3_전처리_실행요약_전체", "내용": "\n".join(mp.get("§3_실행로그_한줄씩", []))},
        {
            "항목": "§2_코드상수_JSON",
            "내용": json.dumps(mp.get("§2_코드에서_정한_상수_및_식", {}), ensure_ascii=False, indent=2),
        },
        {
            "항목": "§3_코드상수_JSON",
            "내용": json.dumps(mp.get("§3_코드에서_정한_상수", {}), ensure_ascii=False, indent=2),
        },
        {
            "항목": "CV_성능_JSON",
            "내용": json.dumps(summary.get("검증_성능_OOF_AP_내림차순", []), ensure_ascii=False, indent=2),
        },
        {
            "항목": "최고_모델_JSON",
            "내용": json.dumps(summary.get("최고_모델"), ensure_ascii=False, indent=2),
        },
        {
            "항목": "모델별_설정_JSON",
            "내용": json.dumps(summary.get("모델_및_튜닝_설정_블록별", {}), ensure_ascii=False, indent=2),
        },
    ]
    return pd.DataFrame(rows)


def build_catboost_effective_params(
    opt_rows: list[dict[str, str]],
    optuna_stdout_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """코드의 CATBOOST_ITER + 출력 best trial → CatBoost 최종 추정 표."""
    cat_iter = None
    for r in opt_rows:
        if r["이름"] == "CATBOOST_ITER":
            try:
                cat_iter = int(ast.literal_eval(r["값"]))
            except Exception:
                cat_iter = r["값"]
    best: dict[str, str] = {}
    for r in optuna_stdout_rows:
        if r["블록"] == "Optuna_best_trial":
            best[r["이름"]] = r["값"]
    fixed = {
        "bootstrap_type": "Bernoulli",
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": "42",
        "early_stopping_rounds": "50",
        "auto_class_weights": "Balanced",
        "verbose": "False",
    }
    out: list[dict[str, str]] = []
    if cat_iter is not None:
        out.append(
            {
                "블록": "CatBoost_최종추정_튜닝후",
                "이름": "iterations",
                "값": str(cat_iter),
                "출처": "코드_CATBOOST_ITER",
            }
        )
    for k in ("learning_rate", "depth", "l2_leaf_reg", "random_strength", "border_count", "subsample"):
        if k in best:
            out.append(
                {
                    "블록": "CatBoost_최종추정_튜닝후",
                    "이름": k,
                    "값": best[k],
                    "출처": "노트북_출력_best_trial",
                }
            )
    for k, v in fixed.items():
        out.append(
            {
                "블록": "CatBoost_최종추정_튜닝후",
                "이름": k,
                "값": v,
                "출처": "코드_고정",
            }
        )
    return out


def _parse_cv_table(text: str) -> pd.DataFrame | None:
    rows: list[tuple[str, float, float, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
        if m:
            name, a, b, c = m.groups()
            if name == "model":
                continue
            try:
                rows.append((name, float(a), float(b), float(c)))
            except ValueError:
                continue
    if not rows:
        return None
    return pd.DataFrame(
        rows,
        columns=["model", "OOF_AP", "CV_fold_mean_AP", "CV_fold_std_AP"],
    )


def load_notebook() -> dict:
    with open(NB_PATH, encoding="utf-8") as f:
        return json.load(f)


def find_code_after_markdown(nb: dict, md_substr: str) -> tuple[int | None, dict | None]:
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "markdown" and md_substr in _join_source(c):
            for j in range(i + 1, len(nb["cells"])):
                if nb["cells"][j]["cell_type"] == "code":
                    return j, nb["cells"][j]
    return None, None


def find_cell_by_source_substr(nb: dict, substr: str) -> tuple[int | None, dict | None]:
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        if substr in _join_source(c):
            return i, c
    return None, None


def code_to_lines_df(title: str, code: str) -> pd.DataFrame:
    lines = code.splitlines()
    return pd.DataFrame(
        {"섹션": title, "line": range(1, len(lines) + 1), "내용": lines},
    )


def parse_seed_from_nb(nb: dict) -> str | None:
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = _join_source(c)
        if "SEED" in src and "=" in src:
            try:
                t = ast.parse(src)
                s = _find_assign_str(t, "SEED")
                if s:
                    return s
            except SyntaxError:
                pass
    return None


def main() -> None:
    nb = load_notebook()

    i2, sec2 = find_code_after_markdown(nb, "## 2. 데이터 로드")
    i3, sec3 = find_code_after_markdown(nb, "## 3. 전처리")
    idx4, sec4 = find_cell_by_source_substr(nb, "xgb_params = dict")
    idx_opt, opt_cell = find_cell_by_source_substr(nb, "study.optimize(objective")
    idx_cat, cat_cell = find_cell_by_source_substr(nb, "def fold_cat")

    sec2_code = _join_source(sec2) if sec2 else ""
    sec3_code = _join_source(sec3) if sec3 else ""
    sec4_code = _join_source(sec4) if sec4 else ""
    opt_code = _join_source(opt_cell) if opt_cell else ""
    cat_code = _join_source(cat_cell) if cat_cell else ""

    seed = parse_seed_from_nb(nb)
    hp_rows: list[dict[str, str]] = []
    if seed:
        hp_rows.append({"블록": "공통", "이름": "SEED", "값": seed, "출처": "코드"})
    if sec2_code:
        hp_rows.extend(parse_preprocessing_top_level_assigns(sec2_code, "전처리_§2_코드상수"))
    sec2_stdout = _stream_outputs(sec2) if sec2 else ""
    hp_rows.extend(
        parse_section_stdout_as_rows(sec2_stdout, "전처리_§2_실행출력", "§2_노트북_저장_출력")
    )
    if sec3_code:
        hp_rows.extend(parse_preprocessing_top_level_assigns(sec3_code, "전처리_§3_코드상수"))
    sec3_stdout = _stream_outputs(sec3) if sec3 else ""
    hp_rows.extend(
        parse_section_stdout_as_rows(sec3_stdout, "전처리_§3_실행출력", "§3_노트북_저장_출력")
    )
    if sec4_code:
        hp_rows.extend(parse_sec4_hyperparams(sec4_code))
    sec4_stdout = _stream_outputs(sec4) if sec4 else ""
    hp_rows.extend(parse_baseline_run_manifest(sec4_stdout))

    opt_rows: list[dict[str, str]] = []
    if opt_code:
        opt_rows = parse_optuna_settings(opt_code)
        hp_rows.extend(opt_rows)

    opt_stdout = _stream_outputs(opt_cell) if opt_cell else ""
    opt_out_parsed = parse_optuna_best_from_stdout(opt_stdout)
    hp_rows.extend(opt_out_parsed)
    hp_rows.extend(build_catboost_effective_params(opt_rows, opt_out_parsed))

    # CatBoost else 분기 기본값 (코드에서 추출)
    if cat_code:
        try:
            tree = ast.parse(cat_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id != "CatBoostClassifier":
                        continue
                    d = _dict_from_call_keywords(node)
                    if not d:
                        continue
                    for k, v in d.items():
                        hp_rows.append(
                            {
                                "블록": "CatBoost_기본_else분기",
                                "이름": k,
                                "값": v,
                                "출처": "코드_튜닝없을때",
                            }
                        )
                    break
        except SyntaxError:
            pass

    hp_df = pd.DataFrame(hp_rows)
    if hp_df.empty:
        hp_df = pd.DataFrame([{"비고": "하이퍼파라미터 파싱 실패 — baseline.ipynb 구조 확인"}])

    cv_df: pd.DataFrame | None = None
    cv_text_combined = ""
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        st = _stream_outputs(c)
        if "oof_ap" in st and "model" in st:
            t = _parse_cv_table(st)
            if t is not None and not t.empty:
                cv_df = t
            cv_text_combined += st + "\n---\n"

    if cv_df is None:
        cv_df = pd.DataFrame(
            [{"비고": "노트북을 실행해 출력을 저장한 뒤 이 스크립트를 다시 실행하세요."}]
        )

    experiment_summary = build_experiment_summary(hp_rows, cv_df, sec2_stdout, sec3_stdout)
    summary_df = experiment_summary_to_dataframe(experiment_summary)

    optuna_summary_df = pd.DataFrame(opt_out_parsed) if opt_out_parsed else pd.DataFrame(
        [{"비고": "Optuna 셀 출력 없음"}]
    )

    meta = pd.DataFrame(
        [
            ("노트북", NB_PATH.name),
            ("생성", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("시트_하이퍼파라미터_표", "§2·§3 전처리(AST+출력) + §4·Optuna + 실행 스냅샷"),
            ("§2 셀", str(i2)),
            ("§3 셀", str(i3)),
            ("§4 셀", str(idx4)),
            ("§4-b 셀", str(idx_opt)),
            ("§4-c 셀", str(idx_cat)),
            ("JSON", JSON_OUT.name),
        ],
        columns=["항목", "내용"],
    )

    df2 = code_to_lines_df("§2", sec2_code)
    df3 = code_to_lines_df("§3", sec3_code)
    df4 = code_to_lines_df("§4", sec4_code)
    dfo = code_to_lines_df("§4-b", opt_code)
    dfc = code_to_lines_df("§4-c", cat_code)

    log2 = _stream_outputs(sec2) if sec2 else ""
    log3 = _stream_outputs(sec3) if sec3 else ""

    # JSON 백업 (맨 위 experiment_summary 가 사람이 읽기 좋은 정리본)
    payload = {
        "experiment_summary": experiment_summary,
        "meta": meta.to_dict(orient="records"),
        "hyperparams_flat": hp_rows,
        "cv_performance": cv_df.to_dict(orient="records") if cv_df is not None else [],
    }
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    out_path = OUT_PRIMARY
    try:
        writer_cm = pd.ExcelWriter(out_path, engine="openpyxl")
    except PermissionError:
        out_path = OUT_FALLBACK
        writer_cm = pd.ExcelWriter(out_path, engine="openpyxl")

    with writer_cm as xl:
        meta.to_excel(xl, sheet_name="메타", index=False)
        summary_df.to_excel(xl, sheet_name="실험요약", index=False)
        hp_df.to_excel(xl, sheet_name="하이퍼파라미터_값", index=False)
        cv_df.to_excel(xl, sheet_name="모델_CV성능", index=False)
        optuna_summary_df.to_excel(xl, sheet_name="Optuna_출력요약", index=False)
        df2.to_excel(xl, sheet_name="전처리_§2_코드", index=False)
        df3.to_excel(xl, sheet_name="전처리_§3_코드", index=False)
        df4.to_excel(xl, sheet_name="하이퍼_§4_코드", index=False)
        dfo.to_excel(xl, sheet_name="하이퍼_§4b_코드", index=False)
        dfc.to_excel(xl, sheet_name="하이퍼_§4c_코드", index=False)
        pd.DataFrame({"§2_실행로그": log2.splitlines()}).to_excel(
            xl, sheet_name="실행로그_§2", index=False
        )
        pd.DataFrame({"§3_실행로그": log3.splitlines()}).to_excel(
            xl, sheet_name="실행로그_§3", index=False
        )
        pd.DataFrame({"CV_출력": cv_text_combined.splitlines()}).to_excel(
            xl, sheet_name="실행로그_CV", index=False
        )

    print(f"저장: {out_path}")
    print(f"JSON: {JSON_OUT}")
    print(f"하이퍼파라미터 행 수: {len(hp_df)}")
    print(f"CV 행 수: {len(cv_df)}")


if __name__ == "__main__":
    main()
