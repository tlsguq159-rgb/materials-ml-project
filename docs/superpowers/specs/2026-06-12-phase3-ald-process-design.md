# Phase 3 설계: ALD 공정 파라미터 → GPC 예측

**날짜:** 2026-06-12  
**상태:** 승인됨

---

## 배경

Phase 1: 밴드갭 예측 (소재 후보 발굴)  
Phase 2: k + Eg Pareto front (공정 후보 381개 선정)  
Phase 3 목표: 선정 소재(HfO₂, Al₂O₃, ZrO₂)의 ALD 공정 파라미터로 GPC 예측 → 공정 최적화 방향 제시

---

## 데이터 전략

### 수집 방법
문헌 수동 추출. Google Scholar / Web of Science 검색.

**추천 검색어:**
- `"HfO2 ALD growth per cycle" temperature`
- `"Al2O3 ALD" "TMA" OR "TDMAH" growth rate`
- `"ZrO2 ALD" "growth per cycle"`

**목표:** HfO₂ ~100, Al₂O₃ ~100, ZrO₂ ~50 → 총 150~300행

### CSV 스키마

| 컬럼 | 타입 | 예시 | 설명 |
|------|------|------|------|
| `material` | 범주 | HfO2 | 증착 소재 |
| `precursor` | 범주 | TDMAH | 전구체 이름 |
| `oxidant` | 범주 | H2O | 산화제 (H2O / O3 / O2_plasma) |
| `temperature` | 연속 | 250 | 증착 온도 (°C) |
| `pulse_time` | 연속 | 0.3 | 전구체 pulse 시간 (s) |
| `purge_time` | 연속 | 10 | purge 시간 (s) |
| `gpc` | 연속 | 1.12 | growth per cycle (Å/cycle) — **타겟** |
| `doi` | 문자열 | 10.1021/... | 출처 DOI (추적용) |

저장 위치: `data/ald_process_data.csv`

---

## 노트북 구조 (3개 신규)

### 09_process_data.ipynb
1. `data/ald_process_data.csv` 로드
2. EDA:
   - GPC 분포 (소재별 histogram)
   - temperature vs GPC scatter (소재별 색상 구분)
   - precursor / oxidant 분포
3. 결측치 확인, 이상치 처리:
   - GPC < 0 → 제거 (물리적 불가)
   - GPC > 4 Å/cycle → CVD 모드 의심으로 플래그 표시 (제거 전 논의)
4. 통계 요약 출력

### 10_process_model.ipynb
1. 범주형 인코딩: `material`, `precursor`, `oxidant` → Label Encoding
2. Feature matrix: `[material_enc, precursor_enc, oxidant_enc, temperature, pulse_time, purge_time]`
3. XGBoost 회귀 (소수 데이터 대응 파라미터):
   - `n_estimators=300, learning_rate=0.05, max_depth=4`
   - `min_child_weight=5, subsample=0.8, colsample_bytree=0.8`
4. Train/test split (80/20), MAE / R² 평가
5. 모델 저장: `data/xgb_process_model.json`

### 11_interpretation_process.ipynb
1. Feature importance → ALD 물리 해석:
   - `temperature`: ALD 윈도우 (너무 낮으면 미반응, 너무 높으면 CVD 분해)
   - `pulse_time`: 포화 거동 (self-limiting — 일정 이상에서 GPC 불변)
   - `material / precursor`: 소재·전구체별 기저 GPC 차이
2. 온도 vs GPC partial dependence plot (각 소재별)
3. Phase 1/2/3 비교 표 (세 모델의 feature importance 1위 비교)
4. 공정 최적화 제안: "이 온도 범위가 ALD 윈도우 — 실험 우선 탐색 구간"

---

## 면접 스토리 (완성형)

```
Phase 1: 소재 밴드갭 → d 오비탈 이론
Phase 2: 유전율 → 이온 분극 이론  
Phase 3: ALD 공정 GPC → ALD 자기 제한 반응 이론

세 모델 모두 XGBoost로 학습했는데, 각각 완전히 다른 물리를 포착했습니다.
이것이 Materials Informatics의 매력: 같은 도구가 소재 → 공정 → 수율까지 연결됩니다.
```

---

## 성공 기준

- [ ] 데이터: 150행 이상 수집 (소재 3종 포함)
- [ ] GPC 모델: R² > 0.5 (공정 데이터 이질성 감안, 0.5+ 달성 시 충분)
- [ ] 온도 vs GPC 그래프에서 ALD 윈도우 패턴 가시적 확인
- [ ] Phase 1/2/3 feature importance 1위 비교 표 완성
