# Phase 2 설계: 유전율 예측 + Dual-Objective 스크리닝

**날짜:** 2026-06-11  
**상태:** 승인됨

---

## 배경

Phase 1에서 XGBoost로 밴드갭(Eg) 예측 완료 (MAE 0.287 eV, R² 0.749).  
Phase 2 목표: 유전율(k) 예측 모델을 추가하고, 두 물성을 동시에 만족하는 DRAM high-k 후보를 스크리닝한다.

DRAM high-k 소재 요건:
- k > 20 (HfO₂ 기준 ~25)
- Eg > 3 eV (누설전류 방지)

---

## 데이터 전략

**수집 대상:** Materials Project `dielectric` 프로퍼티 → `total_dielectric` (ionic + electronic 합산, 실험값 근접).

**조인 전략:**
```
MP API (dielectric_total) — material_id로 inner join
    ↓
featurized_highk.csv (3,914개, 132 Magpie features 기보유)
    ↓
k 보유 서브셋 (~500–1,500개 예상) → XGBoost 학습
전체 3,914개 → k 예측 적용
```

데이터 부족 현상 자체를 "composition-only 예측이 필요한 이유"로 07_interpretation_k 에서 활용.

---

## 노트북 구조 (3개 신규)

### 06_dielectric_model.ipynb
1. MP API로 `dielectric_total` 수집
2. `featurized_highk.csv`와 material_id inner join
3. XGBoost 회귀 학습 (Phase 1과 동일 하이퍼파라미터 기준)
4. MAE / R² 평가, 예측 scatter plot
5. 모델 저장: `data/xgb_dielectric_model.json`
6. 전체 3,914개 후보에 k 예측 적용, 결과 저장: `data/dual_predictions.csv`

### 07_interpretation_k.ipynb
1. Feature importance Top 20 추출
2. 이온 분극 이론으로 해석:
   - 분극률 큰 양이온(Ba²⁺, Sr²⁺) → k ↑
   - Goldschmidt tolerance factor → 페로브스카이트 안정성
   - 저전기음성도 양이온 → 이온결합 강화 → ionic contribution ↑
3. Phase 1(d 오비탈 지배) vs Phase 2(이온 분극 지배) 대비 표
4. 면접 메시지: "같은 Magpie feature로 서로 다른 물리를 포착한다"

### 08_dual_screening.ipynb
1. `dual_predictions.csv` 로드 (ML Eg + ML k)
2. Pareto front 계산 (k ↑, Eg ↑ 동시 최적)
3. 시각화:
   - k vs Eg 산포도 (전체 후보)
   - Pareto 경계선 강조
   - HfO₂, ZrO₂, Al₂O₃ 레퍼런스 포인트 표시
4. Top 20 후보 테이블 (formula, predicted_k, predicted_Eg, on_pareto_front Y/N)
5. 면접 1분 스크립트 업데이트

---

## 면접 스토리 (완성형)

> "밴드갭 모델은 d 오비탈 점유 상태가 지배합니다. 유전율 모델은 이온 분극률이 지배합니다.  
> 같은 Magpie feature 132개로 학습했는데 서로 완전히 다른 물리를 포착합니다.  
> Pareto front에서 두 조건을 동시에 만족하는 소재를 HfO₂보다 유망한 후보로 제안했고,  
> 이 결과는 실험 우선순위 결정에 직접 연결될 수 있습니다."

---

## 기술 스택 추가 없음

기존 스택(matminer, XGBoost, mp-api, matplotlib) 그대로 사용.

---

## 성공 기준

- [ ] k 모델 R² > 0.6 (dielectric은 구조 의존성 강해 composition-only로 0.6+ 달성 시 충분)
- [ ] Pareto front 시각화에 HfO₂/ZrO₂ 레퍼런스 표시
- [ ] Top 20 후보 중 최소 1개가 기존 알려진 high-k 소재와 일치 (sanity check)
- [ ] 면접 스크립트 07, 08 각 1개씩 추가
