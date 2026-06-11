# scripts/gen_dummy_ald.py
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def gpc_model(material, temp, pulse_time):
    """ALD 물리를 반영한 더미 GPC 생성."""
    params = {
        'HfO2':  {'window': (150, 300), 'base_gpc': 1.1},
        'Al2O3': {'window': (100, 280), 'base_gpc': 1.05},
        'ZrO2':  {'window': (150, 280), 'base_gpc': 1.0},
    }
    if material not in params:
        raise ValueError(f"Unknown material '{material}'. Expected: {list(params)}")
    p = params[material]
    lo, hi = p['window']
    gpc = p['base_gpc']
    if temp < lo:
        gpc *= (0.5 + 0.5 * (temp / lo))  # sub-window ramp: GPC → 50% at T=0
    elif temp > hi:
        gpc *= (1 + 0.01 * (temp - hi))   # CVD regime: ~1%/°C above window top
    if pulse_time < 0.1:                   # unsaturated regime: < 100ms pulse
        gpc *= (pulse_time / 0.1)
    return round(gpc + np.random.normal(0, 0.05), 3)  # ~5% measurement noise

records = []
configs = {
    'HfO2':  {'precursors': ['TDMAH', 'HfCl4', 'TEMAHf'], 'oxidants': ['H2O', 'O3'],  'temps': (100, 380)},
    'Al2O3': {'precursors': ['TMA'],                        'oxidants': ['H2O', 'O3'],  'temps': (80,  320)},
    'ZrO2':  {'precursors': ['ZrCl4', 'TDMAZ'],            'oxidants': ['H2O', 'O3'],  'temps': (120, 330)},
}

n_samples = {'HfO2': 70, 'Al2O3': 60, 'ZrO2': 40}
for material, cfg in configs.items():
    n = n_samples[material]
    for _ in range(n):
        temp      = np.random.randint(*cfg['temps'])
        pulse     = round(np.random.choice([0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]), 2)
        purge     = round(float(np.random.choice([5, 10, 15, 20, 30])), 1)
        precursor = np.random.choice(cfg['precursors'])
        oxidant   = np.random.choice(cfg['oxidants'])
        gpc       = gpc_model(material, temp, pulse)
        records.append({
            'material': material, 'precursor': precursor, 'oxidant': oxidant,
            'temperature': temp, 'pulse_time': pulse, 'purge_time': purge,
            'gpc': gpc, 'doi': 'DUMMY_DATA'
        })

df = pd.DataFrame(records)
print(f"더미 데이터: {len(df)}행")
print(df['material'].value_counts())
df.to_csv('data/ald_process_data.csv', index=False)
print("저장 완료: data/ald_process_data.csv")
print("\n[주의] 이 데이터는 더미입니다. doi='DUMMY_DATA' 행을 실제 문헌 데이터로 교체하세요.")
