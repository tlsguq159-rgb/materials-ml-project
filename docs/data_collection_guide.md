# ALD 공정 데이터 수집 가이드

## 목표
`data/ald_process_data.csv`에서 `doi='DUMMY_DATA'` 행을 실제 문헌 데이터로 교체.

## CSV 컬럼 설명

| 컬럼 | 허용값 예시 | 주의사항 |
|------|-----------|---------|
| material | HfO2, Al2O3, ZrO2 | 대소문자 통일 |
| precursor | TDMAH, HfCl4, TEMAHf, TMA, ZrCl4, TDMAZ | 약어 통일 |
| oxidant | H2O, O3, O2_plasma | 세 가지 중 하나로 통일 |
| temperature | 숫자 (°C) | 정수 입력 |
| pulse_time | 숫자 (초) | 소수점 허용 |
| purge_time | 숫자 (초) | 소수점 허용 |
| gpc | 숫자 (Å/cycle) | 소수점 3자리 |
| doi | 10.xxxx/... | 없으면 "unknown" |

## 추천 검색어

### HfO₂ ALD
- `"HfO2 ALD" "growth per cycle" temperature`
- `"hafnium oxide ALD" TDMAH OR HfCl4 GPC`
- Ritala 2000, Hausmann 2002, Cho 2005 등 landmark 논문 먼저

### Al₂O₃ ALD
- `"Al2O3 ALD" TMA H2O "growth per cycle"`
- `"aluminum oxide ALD" temperature window`
- George 2010 ALD review (Chem. Rev.) 참고

### ZrO₂ ALD
- `"ZrO2 ALD" "growth per cycle"`
- `"zirconium oxide ALD" ZrCl4 OR TDMAZ`

## 데이터 입력 방법

1. 논문에서 Table 또는 Figure의 GPC 값 추출
2. Figure의 경우 WebPlotDigitizer(무료) 사용하여 수치 추출
3. CSV에 한 행씩 추가
4. doi 컬럼에 DOI 입력 (추적성)

## 완료 후 재실행

```powershell
cd "C:\Users\USER\OneDrive\바탕 화면\materials-ml-project"
python scripts/run_process_model.py
python scripts/run_interpretation_process.py
```

더미 데이터 행 제거:
```python
import pandas as pd
df = pd.read_csv('data/ald_process_data.csv')
df_real = df[df['doi'] != 'DUMMY_DATA']
df_real.to_csv('data/ald_process_data.csv', index=False)
print(f"실데이터만 유지: {len(df_real)}행")
```
