# 모델 시도 요약 및 CatBoost 선정 (`catboost.ipynb` 기준)

동일한 **Stratified 5-Fold**와 **동일 전처리 행렬(`X`, `y`, `X_test`)**으로 모델별 Out-of-Fold **Average Precision (AP)**을 비교한 뒤, OOF가 가장 높은 모델로 테스트 예측을 합칩니다.

---

## 1. 평가 방식

- **지표**: `average_precision_score` (AP)
- **CV**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- 각 폴드에서 검증 구간에 대한 예측을 모으면 **OOF AP**가 됩니다.
- 테스트 예측은 5개 폴드 예측의 **평균**입니다.

---

## 2. 시도한 모델 (§4)


| 모델                       | 구성 요약                                                                                    |
| ------------------------ | ---------------------------------------------------------------------------------------- |
| **XGBoost**              | `binary:logistic`, `tree_method='hist'`, `scale_pos_weight`, 조기 종료 50라운드 등               |
| **LightGBM**             | `binary`, metric `average_precision`, `scale_pos_weight`, 조기 종료                          |
| **HistGradientBoosting** | sklearn, `class_weight='balanced'`, 깊이·리프 수 등 부스팅 계열과 대략 맞춤                              |
| **Linear_SGD**           | `Pipeline`: `StandardScaler` + `SGDClassifier(loss='log_loss', class_weight='balanced')` |


노트북 출력에 따른 **OOF AP** (한 번의 실행 스냅샷):


| 모델                   | OOF AP  | 비고               |
| -------------------- | ------- | ---------------- |
| HistGradientBoosting | 0.16276 | CatBoost 추가 전 1위 |
| XGBoost              | 0.16262 | HGB와 근소한 차이      |
| LightGBM             | 0.15838 |                  |
| Linear_SGD           | 0.10782 | 선형 베이스라인으로 가장 낮음 |


`SUBMIT_MODEL = 'best'`이면 위 비교만 있을 때는 **HistGradientBoosting**이 자동 선택됩니다.

---

## 3. CatBoost (§4-b, §4-c)

### 3.1 선택적 하이퍼파라미터 튜닝 (§4-b)

- **Optuna** + **TPE** 샘플러로 AP 최대화.
- 튜닝 예산 예시: Train 서브샘플 **120,000행**, **3-Fold**, **25 trials** (노트북 기본값).
- 튜닝 결과가 `CATBOOST_BEST_PARAMS`에 저장되면 §4-c에서 그대로 사용합니다.
- 미실행 시 §4-c는 CatBoost **기본 하이퍼파라미터**로 학습합니다.

### 3.2 5-Fold 재평가 (§4-c)

CatBoost를 같은 5-Fold 루틴에 넣어 요약표를 **갱신**합니다. 동일 스냅샷 출력 예:


| 모델                   | OOF AP      |
| -------------------- | ----------- |
| **CatBoost**         | **0.17555** |
| HistGradientBoosting | 0.16276     |
| XGBoost              | 0.16262     |
| LightGBM             | 0.15838     |
| Linear_SGD           | 0.10782     |


---

## 4. 왜 CatBoost가 “가장 크게” 나왔는지 (요약)

- **OOF AP 기준**으로 CatBoost(약 **0.176**)가 부스팅 계열(XGBoost, LightGBM, HGB, 약 **0.158~0.163**)보다 **한 단계 위**입니다. 불균형 이진 분류에서 트리 부스터 간 차이는 종종 **학습 안정성·규제·분할 방식**에서 갈리는데, 이 노트북에서는 **동일 피처·동일 폴드**에서 CatBoost가 가장 높은 OOF를 기록했습니다.
- **선형 모델(SGD)**은 스케일링 후에도 AP가 크게 낮아, 이 문제에서는 **비선형·상호작용**이 중요한 것으로 해석할 수 있습니다.
- CatBoost는 (선택적으로) **Optuna로 PRAUC 지향 튜닝**을 거친 뒤 전체 5-Fold에 적용되어, 다른 모델 대비 **파라미터 쪽 이점**도 일부 기여했을 수 있습니다.

---

## 5. 제출

- `SUBMIT_MODEL = 'best'`이면 요약표 **1위 모델**의 테스트 평균 예측이 `test_preds`로 쓰이고, CatBoost까지 돌린 뒤에는 `**submission_baseline.csv`** 등으로 저장됩니다.
- 특정 모델만 고정하려면 `'XGBoost'`, `'LightGBM'` 등으로 변수를 바꿀 수 있습니다.

---

## 6. 주의

- 위 AP 숫자는 **노트북에 저장된 한 번의 실행 결과**입니다. 시드·라이브러리 버전·튜닝 유무에 따라 소수점 자리는 달라질 수 있습니다.
- 재현 시에는 `SEED = 42`, 동일 CSV, 동일 `CATBOOST_BEST_PARAMS` 여부를 맞추는 것이 좋습니다.

